import opensmile
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
from sklearn.decomposition import PCA

smile_egemaps = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

smile_compare = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

MIN_DURATION_SECONDS = 1.5


def extract_utterance_features(
    audio: np.ndarray,
    utt_start: float,
    utt_end: float,
    sr: int = 16000,
    feature_set: str = "egemaps"
) -> np.ndarray | None:
    segment = audio[int(utt_start * sr): int(utt_end * sr)]

    if len(segment) < int(MIN_DURATION_SECONDS * sr):
        return None

    smile = smile_egemaps if feature_set == "egemaps" else smile_compare

    try:
        features = smile.process_signal(segment, sr)
    except Exception:
        return None

    if features is None or features.empty:
        return None

    vals = features.values[0]

    if np.any(np.isnan(vals)):
        return None

    return vals


def extract_all_utterances_with_tracking(
    aligned_utterances: list[dict],
    audio: np.ndarray,
    sr: int = 16000,
    feature_set: str = "egemaps"
) -> tuple[list[dict], dict]:
    results  = []
    n_total  = len(aligned_utterances)
    n_short  = 0
    n_nan    = 0

    for utt in aligned_utterances:
        duration = utt["duration_s"]

        if duration < MIN_DURATION_SECONDS:
            n_short += 1
            continue

        segment = audio[int(utt["start"] * sr): int(utt["end"] * sr)]
        smile   = smile_egemaps if feature_set == "egemaps" else smile_compare

        try:
            feats = smile.process_signal(segment, sr)
            vals  = feats.values[0] if feats is not None and not feats.empty else None
        except Exception:
            vals = None

        if vals is None or np.any(np.isnan(vals)):
            n_nan += 1
            continue

        utt_out = {**utt, "audio_features": vals}
        results.append(utt_out)

    drop_stats = {
        "n_total":      n_total,
        "n_kept":       len(results),
        "n_dropped_short": n_short,
        "n_dropped_nan":   n_nan,
        "drop_rate_pct":   (n_total - len(results)) / max(n_total, 1) * 100,
    }
    print(f"Feature extraction: kept {len(results)}/{n_total} utterances "
          f"({drop_stats['drop_rate_pct']:.1f}% dropped - "
          f"{n_short} too short, {n_nan} NaN)")

    return results, drop_stats


def apply_pca_if_compare(audio_features: np.ndarray,
                          n_components: int = 64) -> tuple[np.ndarray, object]:
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(audio_features)
    variance_retained = pca.explained_variance_ratio_.sum()
    print(f"PCA: {audio_features.shape[1]} -> {n_components} features, "
          f"{variance_retained:.3f} variance retained")
    return reduced, pca


def compare_maec_lld_vs_reextracted(
    maec_features_csv: str,
    reextracted_features: np.ndarray
) -> dict:
    maec_df   = pd.read_csv(maec_features_csv)
    maec_means = maec_df.drop(columns=["frameTime"]).mean().values
    your_means = reextracted_features.mean(axis=0)

    min_len = min(len(maec_means), len(your_means))
    rho, _  = spearmanr(maec_means[:min_len], your_means[:min_len])

    return {
        "spearman_correlation": rho,
        "interpretation": "Good" if rho > 0.5 else "Check pipeline - low consistency with MAEC"
    }
