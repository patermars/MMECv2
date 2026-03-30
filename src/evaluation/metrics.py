import numpy as np
from scipy.stats import spearmanr, kendalltau


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    if len(preds) < 2:
        return {
            "spearman_rho": 0.0,
            "kendall_tau": 0.0,
            "mae": float("inf"),
            "rmse": float("inf"),
            "top_quartile_hit": 0.0,
        }

    spearman, _ = spearmanr(preds, targets)
    kendall, _  = kendalltau(preds, targets)
    mae         = np.mean(np.abs(preds - targets))
    rmse        = np.sqrt(np.mean((preds - targets) ** 2))

    top_q_preds = preds >= np.percentile(preds, 75)
    top_q_true  = targets >= np.percentile(targets, 75)
    hit_rate    = np.mean(top_q_preds == top_q_true)

    return {
        "spearman_rho":      float(spearman) if not np.isnan(spearman) else 0.0,
        "kendall_tau":       float(kendall) if not np.isnan(kendall) else 0.0,
        "mae":               float(mae),
        "rmse":              float(rmse),
        "top_quartile_hit":  float(hit_rate),
    }
