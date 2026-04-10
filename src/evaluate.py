import argparse, yaml, torch, numpy as np, os
from scipy.stats import spearmanr, kendalltau
from src.model import build_model
from src.dataset import get_dataloaders


@torch.no_grad()
def run_evaluation(model, loader, device, primary_idx=None):
    model.eval()
    all_preds, all_labels, all_ids = [], [], []

    for batch in loader:
        acoustic = batch["acoustic"].to(device, non_blocking=True)
        text_emb = batch["text_emb"].to(device, non_blocking=True)
        wav_emb = batch["wav_emb"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device != "cpu")):
            preds = model(acoustic, text_emb, wav_emb, mask)

        # Extract primary target if multi-task
        if preds.dim() == 2 and primary_idx is not None:
            p = preds[:, primary_idx]
            l = batch["labels"][:, primary_idx]
        elif preds.dim() == 2:
            p = preds[:, 0]
            l = batch["labels"][:, 0]
        else:
            p = preds
            l = batch["labels"]

        all_preds.extend(p.cpu().numpy().tolist())
        all_labels.extend(l.numpy().tolist())
        all_ids.extend(batch["call_ids"])

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    rho, rho_p = spearmanr(preds, labels)
    tau, tau_p = kendalltau(preds, labels)
    mae = np.mean(np.abs(preds - labels))
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    mse = np.mean((preds - labels) ** 2)

    top_q_pred = preds >= np.percentile(preds, 75)
    top_q_true = labels >= np.percentile(labels, 75)
    hit_rate = np.mean(top_q_pred == top_q_true)

    return {
        "spearman_rho": rho,
        "spearman_p": rho_p,
        "kendall_tau": tau,
        "kendall_p": tau_p,
        "mae": mae,
        "rmse": rmse,
        "mse": mse,
        "top_quartile_hit": hit_rate,
        "n_samples": len(preds),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_type", default="cross_modal")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(cfg, args.model_type).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                     weights_only=True))

    loaders = get_dataloaders(cfg)
    loader = loaders[args.split]

    # Determine primary index for multi-task
    multi_task = cfg["label"].get("multi_task", False)
    mt_targets = cfg["label"].get("multi_task_targets", [])
    primary_target = cfg["label"].get("primary_target", "abnormal_vol_3d")
    primary_idx = None
    if multi_task and mt_targets and primary_target in mt_targets:
        primary_idx = mt_targets.index(primary_target)

    metrics = run_evaluation(model, loader, device, primary_idx)
    print(f"\n{'='*50}")
    print(f"Evaluation on {args.split} set ({metrics['n_samples']} samples)")
    print(f"{'='*50}")
    print(f"  Spearman ρ:       {metrics['spearman_rho']:.4f} (p={metrics['spearman_p']:.4e})")
    print(f"  Kendall τ:        {metrics['kendall_tau']:.4f} (p={metrics['kendall_p']:.4e})")
    print(f"  MAE:              {metrics['mae']:.5f}")
    print(f"  RMSE:             {metrics['rmse']:.5f}")
    print(f"  MSE:              {metrics['mse']:.5f}")
    print(f"  Top-quartile hit: {metrics['top_quartile_hit']:.4f}")

    # Run ablation across model variants
    print(f"\n{'='*50}")
    print("Ablation (if checkpoints exist)")
    print(f"{'='*50}")

    ablation_models = ["cross_modal", "audio_only", "text_only", "early_fusion"]
    for mtype in ablation_models:
        ckpt = f"checkpoints/{mtype}_seed0.pt"
        if not os.path.exists(ckpt):
            continue
        m = build_model(cfg, mtype).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        met = run_evaluation(m, loader, device, primary_idx)
        print(f"  {mtype:15s} | ρ={met['spearman_rho']:.4f} "
              f"MAE={met['mae']:.5f} RMSE={met['rmse']:.5f}")


if __name__ == "__main__":
    main()
