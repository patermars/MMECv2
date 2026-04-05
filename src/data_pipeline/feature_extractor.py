"""
Feature extractor for ACL19 sentence-level MP3 files.

ACL19 provides one MP3 per CEO sentence — extract eGeMAPS directly
from each file. No waveform slicing needed.
"""

import opensmile
import numpy as np
import librosa
from pathlib import Path

MIN_DURATION_SECONDS = 1.5
TARGET_SR = 16000

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)


def get_audio_duration(mp3_path: str) -> float:
    try:
        return librosa.get_duration(path=mp3_path)
    except Exception:
        return 0.0


def extract_features_from_file(mp3_path: str) -> np.ndarray | None:
    """
    Load one ACL19 sentence MP3 and extract 88-dim eGeMAPS features.
    Returns None if too short, silent, or openSMILE fails.
    """
    try:
        audio, _ = librosa.load(mp3_path, sr=TARGET_SR, mono=True)
    except Exception as e:
        print(f"    [load error] {Path(mp3_path).name}: {e}")
        return None

    if len(audio) < int(MIN_DURATION_SECONDS * TARGET_SR):
        return None

    try:
        feats = smile.process_signal(audio, TARGET_SR)
    except Exception as e:
        print(f"    [opensmile error] {Path(mp3_path).name}: {e}")
        return None

    if feats is None or feats.empty:
        return None

    vals = feats.values[0]
    if np.any(np.isnan(vals)):
        return None

    return vals.astype(np.float32)


def extract_features_for_call(utterances: list[dict]) -> tuple[list[dict], dict]:
    """
    Extract eGeMAPS for all utterances in a call.
    Each utterance must have an 'audio_path' key pointing to its MP3.
    Returns (enriched_utterances, drop_stats). Failures are dropped, never imputed.
    """
    results = []
    n_total = len(utterances)
    n_short = n_fail = 0

    for utt in utterances:
        mp3_path = utt.get("audio_path", "")
        duration = get_audio_duration(mp3_path)

        if duration < MIN_DURATION_SECONDS:
            n_short += 1
            continue

        feats = extract_features_from_file(mp3_path)
        if feats is None:
            n_fail += 1
            continue

        results.append({**utt, "audio_features": feats, "duration_s": duration})

    drop_stats = {
        "n_total":         n_total,
        "n_kept":          len(results),
        "n_dropped_short": n_short,
        "n_dropped_fail":  n_fail,
        "drop_rate_pct":   (n_total - len(results)) / max(n_total, 1) * 100,
    }
    return results, drop_stats
