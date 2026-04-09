import os, argparse, yaml, time, torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr
from src.model import build_model
from src.dataset import get_dataloaders


def safe_spearman(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    if len(preds) < 3:
        return 0.0
    if np.allclose(preds, preds[0]) or np.allclose(labels, labels[0]):
        return 0.0

    rho, _ = spearmanr(preds, labels)
    if np.isnan(rho):
        return 0.0
    return float(rho)


def train_epoch(model, loader, optimizer, scaler, device, grad_clip):
    model.train()
    total_loss, n = 0.0, 0
    criterion = nn.MSELoss()

    for batch in loader:
        acoustic = batch["acoustic"].to(device, non_blocking=True)
        text_emb = batch["text_emb"].to(device, non_blocking=True)
        wav_emb = batch["wav_emb"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device, enabled=(device != "cpu")):
            preds = model(acoustic, text_emb, wav_emb, mask)
            loss = criterion(preds, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0
    criterion = nn.MSELoss()

    for batch in loader:
        acoustic = batch["acoustic"].to(device, non_blocking=True)
        text_emb = batch["text_emb"].to(device, non_blocking=True)
        wav_emb = batch["wav_emb"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device, enabled=(device != "cpu")):
            preds = model(acoustic, text_emb, wav_emb, mask)
            loss = criterion(preds, labels)

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    if n == 0:
        return {"mse": float("nan"), "mae": float("nan"), "spearman": 0.0, "n": 0, "pred_std": 0.0, "label_std": 0.0}

    mse = total_loss / n
    mae = float(np.mean(np.abs(np.array(all_preds) - np.array(all_labels))))
    rho = safe_spearman(all_preds, all_labels)
    pred_std = float(np.std(all_preds))
    label_std = float(np.std(all_labels))
    return {"mse": mse, "mae": mae, "spearman": rho, "n": n, "pred_std": pred_std, "label_std": label_std}


def train_single_seed(cfg, seed, device, model_type="cross_modal"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n{'='*60}")
    print(f"Seed {seed}")
    print(f"{'='*60}")

    loaders = get_dataloaders(cfg)
    model = build_model(cfg, model_type).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_type} | Params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["training"]["lr"],
                                  weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    scaler = torch.amp.GradScaler(device, enabled=(device != "cpu"))

    best_val_mse = float("inf")
    patience_count = 0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    has_val_data = val_metrics["n"] > 0 if (val_metrics := evaluate(model, loaders["val"], device)) else False

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()
        train_loss = train_epoch(model, loaders["train"], optimizer, scaler,
                                 device, cfg["training"]["grad_clip"])
        val_metrics = evaluate(model, loaders["val"], device)
        scheduler.step()

        dt = time.time() - t0
        print(f"  E{epoch:03d} | train_loss={train_loss:.5f} | "
              f"val_mse={val_metrics['mse']:.5f} val_rho={val_metrics['spearman']:.4f} "
              f"val_pred_std={val_metrics['pred_std']:.5f} val_label_std={val_metrics['label_std']:.5f} | "
              f"{dt:.1f}s")

        if has_val_data and val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        elif has_val_data:
            patience_count += 1
        else:
            # No validation data, save current state
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if has_val_data and patience_count >= cfg["training"]["patience"]:
            print(f"  Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders["test"], device)
    print(f"  Test: mse={test_metrics['mse']:.5f} mae={test_metrics['mae']:.5f} "
          f"rho={test_metrics['spearman']:.4f}")

    return model, test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model_type", default="cross_modal",
                        choices=["cross_modal", "audio_only", "text_only", "early_fusion"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    n_seeds = args.seeds or cfg["training"]["n_seeds"]
    all_metrics = []

    for seed in range(n_seeds):
        model, metrics = train_single_seed(cfg, seed * 42, device, args.model_type)
        all_metrics.append(metrics)

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(),
                   f"checkpoints/{args.model_type}_seed{seed}.pt")

    print(f"\n{'='*60}")
    print(f"Ensemble Results ({n_seeds} seeds)")
    print(f"{'='*60}")
    for key in ["mse", "mae", "spearman"]:
        vals = [m[key] for m in all_metrics]
        print(f"  {key}: {np.mean(vals):.5f} ± {np.std(vals):.5f}")


if __name__ == "__main__":
    main()
