import os, argparse, yaml, time, torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr
from src.model import build_model
from src.dataset import get_dataloaders
import math


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


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class HuberRankLoss(nn.Module):
    """Combined Huber + differentiable rank-correlation loss."""

    def __init__(self, delta=0.1, rank_weight=0.5, multi_task=False,
                 primary_idx=1, primary_weight=2.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.rank_weight = rank_weight
        self.multi_task = multi_task
        self.primary_idx = primary_idx  # Index of primary target in multi-task
        self.primary_weight = primary_weight

    def _soft_rank_loss(self, preds, labels):
        """Differentiable approximation to (1 - Spearman ρ)."""
        if preds.shape[0] < 3:
            return torch.tensor(0.0, device=preds.device)
        # If constant predictions, return penalty
        if preds.std() < 1e-8:
            return torch.tensor(1.0, device=preds.device)

        # Soft ranking via sorting
        def _soft_ranks(x, temperature=0.1):
            """Convert values to soft ranks using pairwise sigmoid."""
            # pairwise differences: how many elements are smaller
            diff = x.unsqueeze(0) - x.unsqueeze(1)  # (N, N)
            ranks = torch.sigmoid(diff / temperature).sum(dim=1)
            return ranks

        pred_ranks = _soft_ranks(preds)
        label_ranks = _soft_ranks(labels)

        # Pearson correlation of ranks ≈ Spearman ρ
        pr_centered = pred_ranks - pred_ranks.mean()
        lr_centered = label_ranks - label_ranks.mean()

        num = (pr_centered * lr_centered).sum()
        den = torch.sqrt((pr_centered ** 2).sum() * (lr_centered ** 2).sum()).clamp(min=1e-8)
        soft_rho = num / den

        return 1.0 - soft_rho  # Minimise this → maximise correlation

    def forward(self, preds, labels):
        if self.multi_task and preds.dim() == 2:
            n_targets = preds.shape[1]
            total_loss = 0.0
            total_weight = 0.0

            for t in range(n_targets):
                w = self.primary_weight if t == self.primary_idx else 1.0
                h = self.huber(preds[:, t], labels[:, t])
                r = self._soft_rank_loss(preds[:, t], labels[:, t])
                total_loss += w * (h + self.rank_weight * r)
                total_weight += w

            return total_loss / total_weight
        else:
            # Single task
            h = self.huber(preds, labels)
            r = self._soft_rank_loss(preds, labels)
            return h + self.rank_weight * r


def _build_criterion(cfg):
    """Build loss function from config."""
    loss_type = cfg["training"].get("loss", "mse")
    multi_task = cfg["label"].get("multi_task", False)
    mt_targets = cfg["label"].get("multi_task_targets", [])
    primary_target = cfg["label"].get("primary_target", "abnormal_vol_3d")

    # Determine primary index in multi-task targets
    primary_idx = 1  # default: abnormal_vol_3d is index 1 in [1d, 3d, 7d]
    if multi_task and mt_targets and primary_target in mt_targets:
        primary_idx = mt_targets.index(primary_target)

    if loss_type == "huber_rank":
        return HuberRankLoss(
            delta=cfg["training"].get("huber_delta", 0.1),
            rank_weight=cfg["training"].get("rank_loss_weight", 0.5),
            multi_task=multi_task,
            primary_idx=primary_idx,
            primary_weight=cfg["label"].get("primary_task_weight", 2.0),
        )
    elif loss_type == "huber":
        return nn.HuberLoss(delta=cfg["training"].get("huber_delta", 0.1))
    else:
        return nn.MSELoss()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scaler, device, grad_clip, criterion,
                primary_idx=None):
    model.train()
    total_loss, n = 0.0, 0
    all_preds, all_labels_list = [], []

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
        
        # Collect for train Spearman
        with torch.no_grad():
            if preds.dim() == 2 and primary_idx is not None:
                p = preds[:, primary_idx]
                l = labels[:, primary_idx]
            elif preds.dim() == 2:
                p = preds[:, 0]
                l = labels[:, 0]
            else:
                p, l = preds, labels
            all_preds.extend(p.cpu().numpy().tolist())
            all_labels_list.extend(l.cpu().numpy().tolist())

    train_rho = safe_spearman(all_preds, all_labels_list)
    return total_loss / max(n, 1), train_rho


@torch.no_grad()
def evaluate(model, loader, device, primary_idx=None):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0
    criterion = nn.MSELoss()  # Always report MSE for comparability

    for batch in loader:
        acoustic = batch["acoustic"].to(device, non_blocking=True)
        text_emb = batch["text_emb"].to(device, non_blocking=True)
        wav_emb = batch["wav_emb"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device, enabled=(device != "cpu")):
            preds = model(acoustic, text_emb, wav_emb, mask)

        # Extract primary target for metrics if multi-task
        if preds.dim() == 2 and primary_idx is not None:
            p = preds[:, primary_idx]
            l = labels[:, primary_idx]
        elif preds.dim() == 2:
            p = preds[:, 0]
            l = labels[:, 0]
        else:
            p = preds
            l = labels

        loss = criterion(p, l)
        total_loss += loss.item() * l.size(0)
        n += l.size(0)
        all_preds.extend(p.cpu().numpy().tolist())
        all_labels.extend(l.cpu().numpy().tolist())

    if n == 0:
        return {"mse": float("nan"), "mae": float("nan"), "spearman": 0.0,
                "n": 0, "pred_std": 0.0, "label_std": 0.0}

    mse = total_loss / n
    mae = float(np.mean(np.abs(np.array(all_preds) - np.array(all_labels))))
    rho = safe_spearman(all_preds, all_labels)
    pred_std = float(np.std(all_preds))
    label_std = float(np.std(all_labels))
    return {"mse": mse, "mae": mae, "spearman": rho, "n": n,
            "pred_std": pred_std, "label_std": label_std}


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

    criterion = _build_criterion(cfg)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["training"]["lr"],
                                  weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    scaler = torch.amp.GradScaler(device, enabled=(device != "cpu"))

    # LR warmup
    warmup_epochs = cfg["training"].get("warmup_epochs", 0)

    # Determine primary target index for multi-task evaluation
    multi_task = cfg["label"].get("multi_task", False)
    mt_targets = cfg["label"].get("multi_task_targets", [])
    primary_target = cfg["label"].get("primary_target", "abnormal_vol_3d")
    primary_idx = None
    if multi_task and mt_targets and primary_target in mt_targets:
        primary_idx = mt_targets.index(primary_target)

    # Model selection: combined score = -mse + rho (higher is better)
    # This is more robust than pure Spearman on tiny val sets
    select_by = cfg["training"].get("model_selection", "spearman")

    best_val_score = -float("inf")
    patience_count = 0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Check if val has data
    val_metrics = evaluate(model, loaders["val"], device, primary_idx)
    has_val_data = val_metrics["n"] > 0
    # Track baseline MSE for normalization
    baseline_mse = val_metrics["mse"] if has_val_data else 1.0

    def _selection_score(metrics):
        """Combined score: normalized MSE improvement + Spearman ρ."""
        if select_by == "spearman":
            # Blend: MSE improvement (normalized) + spearman
            mse_score = (baseline_mse - metrics["mse"]) / max(baseline_mse, 1e-6)
            return mse_score + metrics["spearman"]
        else:
            return -metrics["mse"]  # Lower MSE = higher score

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()
        
        # Linear LR warmup
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = cfg["training"]["lr"] * warmup_factor
        
        train_loss, train_rho = train_epoch(
            model, loaders["train"], optimizer, scaler,
            device, cfg["training"]["grad_clip"], criterion, primary_idx)
        val_metrics = evaluate(model, loaders["val"], device, primary_idx)
        scheduler.step()

        dt = time.time() - t0
        print(f"  E{epoch:03d} | train_loss={train_loss:.5f} train_rho={train_rho:.4f} | "
              f"val_mse={val_metrics['mse']:.5f} val_rho={val_metrics['spearman']:.4f} "
              f"val_pred_std={val_metrics['pred_std']:.5f} | "
              f"{dt:.1f}s")

        if has_val_data:
            current_score = _selection_score(val_metrics)
            improved = current_score > best_val_score

            if improved:
                best_val_score = current_score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
        else:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if has_val_data and patience_count >= cfg["training"]["patience"]:
            print(f"  Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders["test"], device, primary_idx)
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
