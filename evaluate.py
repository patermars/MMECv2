"""
Final model evaluation script.
Loads the best checkpoint, runs inference on the test set, and produces:
  - Regression metrics  (Spearman ρ, Kendall τ, MAE, RMSE, R²)
  - Directional accuracy (predicted vs actual above/below median)
  - Confusion matrix over volatility quartile buckets
  - Residual & scatter plots saved to data/evaluation/
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import yaml
from pathlib import Path
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import (
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)
from torch.utils.data import DataLoader

from src.models.hierarchical_encoder import HierarchicalMultimodalEncoder, collate_calls
from src.data.dataset import EarningsCallDataset

OUT_DIR       = Path("data/evaluation")
CHECKPOINT    = "data/processed/checkpoints/best_model.pt"
LABELS_CSV    = "data/processed/labels/labels.csv"
SPLITS_DIR    = "data/splits"
QUARTILE_LABELS = ["Low", "Medium", "High", "Very High"]


# ── helpers ──────────────────────────────────────────────────────────────────

def load_config(path="configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def to_quartile_bins(values: np.ndarray) -> np.ndarray:
    """Assign each value to a quartile bucket [0-3] based on the array's own percentiles."""
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    bins = np.zeros(len(values), dtype=int)
    bins[values >= q25] = 1
    bins[values >= q50] = 2
    bins[values >= q75] = 3
    return bins


# ── inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    for batch in loader:
        batch  = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        labels = batch.pop("labels")
        out    = model(**batch)
        preds.extend(out.cpu().numpy())
        targets.extend(labels.cpu().numpy())
    return np.array(preds), np.array(targets)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(QUARTILE_LABELS, rotation=30, ha="right")
    ax.set_yticklabels(QUARTILE_LABELS)
    ax.set_xlabel("Predicted Quartile"); ax.set_ylabel("True Quartile")
    ax.set_title("Volatility Quartile Confusion Matrix")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_scatter(preds, targets, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(targets, preds, alpha=0.5, s=20, edgecolors="none")
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("True Abnormal Volatility"); ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs True")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_residuals(preds, targets, out_path: Path):
    residuals = preds - targets
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(targets, residuals, alpha=0.5, s=20, edgecolors="none")
    axes[0].axhline(0, color="r", linestyle="--", linewidth=1)
    axes[0].set_xlabel("True Value"); axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs True")
    axes[1].hist(residuals, bins=20, edgecolor="white")
    axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_cumulative_gain(preds, targets, out_path: Path):
    """How much of the top-k% true volatility is captured by top-k% predictions."""
    order   = np.argsort(preds)[::-1]
    sorted_targets = targets[order]
    cumulative = np.cumsum(sorted_targets) / targets.sum()
    x = np.linspace(0, 1, len(cumulative))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, cumulative, label="Model")
    ax.plot([0, 1], [0, 1], "r--", label="Random")
    ax.set_xlabel("Fraction of Calls Selected"); ax.set_ylabel("Fraction of Total Volatility Captured")
    ax.set_title("Cumulative Gain Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--device",     default=None)
    parser.add_argument("--split",      default="test", choices=["test", "val", "train"])
    args = parser.parse_args()

    config = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── model ────────────────────────────────────────────────────────────────
    mc = config.get("model", {})
    model = HierarchicalMultimodalEncoder(
        audio_dim=mc.get("audio_dim", 88),
        text_dim=mc.get("text_dim", 768),
        hidden_dim=mc.get("hidden_dim", 256),
        n_heads=mc.get("n_heads", 4),
        n_structured=mc.get("n_structured", 5),
    ).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run `python run_training.py --mode train` first."
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {ckpt_path}")

    # ── data ─────────────────────────────────────────────────────────────────
    batch_size = config.get("training", {}).get("batch_size", 64)
    ds     = EarningsCallDataset(f"{SPLITS_DIR}/{args.split}.csv", "data/processed/checkpoints", LABELS_CSV)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_calls, num_workers=2, pin_memory=(device == "cuda"))
    print(f"Evaluating on {args.split} set: {len(ds)} samples")

    # ── inference ─────────────────────────────────────────────────────────────
    preds, targets = run_inference(model, loader, device)

    # ── regression metrics ────────────────────────────────────────────────────
    spearman, spearman_p = spearmanr(preds, targets)
    kendall,  kendall_p  = kendalltau(preds, targets)
    mae  = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2   = r2_score(targets, preds)

    # directional accuracy (above/below median)
    median        = np.median(targets)
    dir_acc       = np.mean((preds >= median) == (targets >= median))

    # top-quartile precision
    top_q_pred    = preds   >= np.percentile(preds,   75)
    top_q_true    = targets >= np.percentile(targets, 75)
    top_q_prec    = np.sum(top_q_pred & top_q_true) / np.sum(top_q_pred).clip(min=1)
    top_q_recall  = np.sum(top_q_pred & top_q_true) / np.sum(top_q_true).clip(min=1)

    print("\n" + "="*50)
    print("REGRESSION METRICS")
    print("="*50)
    metrics = {
        "Spearman ρ":          f"{spearman:.4f}  (p={spearman_p:.4f})",
        "Kendall τ":           f"{kendall:.4f}   (p={kendall_p:.4f})",
        "MAE":                 f"{mae:.6f}",
        "RMSE":                f"{rmse:.6f}",
        "R²":                  f"{r2:.4f}",
        "Directional Acc":     f"{dir_acc:.4f}",
        "Top-Q Precision":     f"{top_q_prec:.4f}",
        "Top-Q Recall":        f"{top_q_recall:.4f}",
    }
    for k, v in metrics.items():
        print(f"  {k:<22} {v}")

    # ── quartile confusion matrix ─────────────────────────────────────────────
    true_bins = to_quartile_bins(targets)
    pred_bins = to_quartile_bins(preds)
    cm = confusion_matrix(true_bins, pred_bins, labels=[0, 1, 2, 3])

    print("\n" + "="*50)
    print("VOLATILITY QUARTILE CLASSIFICATION")
    print("="*50)
    print(classification_report(true_bins, pred_bins,
                                 target_names=QUARTILE_LABELS, zero_division=0))

    # ── save metrics CSV ──────────────────────────────────────────────────────
    results_df = pd.DataFrame([{
        "split":            args.split,
        "n_samples":        len(targets),
        "spearman_rho":     spearman,
        "spearman_p":       spearman_p,
        "kendall_tau":      kendall,
        "kendall_p":        kendall_p,
        "mae":              mae,
        "rmse":             rmse,
        "r2":               r2,
        "directional_acc":  dir_acc,
        "top_q_precision":  top_q_prec,
        "top_q_recall":     top_q_recall,
    }])
    csv_path = OUT_DIR / f"metrics_{args.split}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Metrics saved → {csv_path}")

    # ── plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_confusion_matrix(cm,     OUT_DIR / f"confusion_matrix_{args.split}.png")
    plot_scatter(preds, targets,  OUT_DIR / f"scatter_{args.split}.png")
    plot_residuals(preds, targets, OUT_DIR / f"residuals_{args.split}.png")
    plot_cumulative_gain(preds, targets, OUT_DIR / f"cumulative_gain_{args.split}.png")

    print(f"\nAll outputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
