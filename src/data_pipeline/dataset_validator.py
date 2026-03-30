import numpy as np
import pandas as pd


def validate_dataset(records: list[dict], labels_df: pd.DataFrame) -> dict:
    report = {
        "n_records": len(records),
        "n_labels": len(labels_df),
        "issues": [],
    }

    if len(records) == 0:
        report["issues"].append("No records in dataset")
        return report

    n_utts = [r["n_utterances"] for r in records]
    report["utterance_stats"] = {
        "mean": np.mean(n_utts),
        "min":  int(np.min(n_utts)),
        "max":  int(np.max(n_utts)),
        "std":  np.std(n_utts),
    }

    audio_dims = set()
    text_dims  = set()
    for r in records:
        for u in r["utterances"]:
            if "audio_features" in u:
                audio_dims.add(u["audio_features"].shape[-1] if hasattr(u["audio_features"], "shape") else len(u["audio_features"]))
            if "text_embedding" in u:
                text_dims.add(u["text_embedding"].shape[-1] if hasattr(u["text_embedding"], "shape") else len(u["text_embedding"]))

    report["audio_feature_dims"] = list(audio_dims)
    report["text_embedding_dims"] = list(text_dims)

    if len(audio_dims) > 1:
        report["issues"].append(f"Inconsistent audio feature dims: {audio_dims}")
    if len(text_dims) > 1:
        report["issues"].append(f"Inconsistent text embedding dims: {text_dims}")

    nan_cols = labels_df.columns[labels_df.isna().any()].tolist()
    if nan_cols:
        nan_rates = {col: labels_df[col].isna().mean() for col in nan_cols}
        report["label_nan_rates"] = nan_rates
        for col, rate in nan_rates.items():
            if rate > 0.5:
                report["issues"].append(f"High NaN rate in {col}: {rate:.2%}")

    if "abnormal_vol_3d" in labels_df.columns:
        target = labels_df["abnormal_vol_3d"].dropna()
        report["target_stats"] = {
            "mean":   target.mean(),
            "std":    target.std(),
            "median": target.median(),
            "min":    target.min(),
            "max":    target.max(),
            "q25":    target.quantile(0.25),
            "q75":    target.quantile(0.75),
        }

    section_counts = {}
    role_counts = {}
    for r in records:
        for u in r["utterances"]:
            sec = u.get("section", "unknown")
            section_counts[sec] = section_counts.get(sec, 0) + 1
            role = u.get("speaker_role", "UNKNOWN")
            role_counts[role] = role_counts.get(role, 0) + 1

    report["section_distribution"] = section_counts
    report["role_distribution"] = role_counts

    drop_rates = [r["drop_stats"]["drop_rate_pct"] for r in records if "drop_stats" in r]
    if drop_rates:
        report["avg_utterance_drop_rate_pct"] = np.mean(drop_rates)

    print(f"Validation: {len(records)} records, {len(report['issues'])} issues")
    for issue in report["issues"]:
        print(f"  ISSUE: {issue}")

    return report
