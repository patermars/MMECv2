import os
import re
import json
import csv
from pathlib import Path
import pandas as pd
import numpy as np

MANAGEMENT_TITLES = {
    "ceo", "chief executive", "president", "cfo", "chief financial",
    "chief operating", "coo", "chairman", "vice president", "vp"
}
ANALYST_MARKERS = {
    "analyst", "research", "capital", "securities", "partners",
    "asset management", "goldman", "morgan", "citigroup", "jp morgan"
}

PRAAT_FEATURE_COLUMNS = [
    "Mean pitch", "Standard deviation", "Minimum pitch", "Maximum pitch",
    "Mean intensity", "Minimum intensity", "Maximum intensity",
    "Number of pulses", "Number of periods", "Mean period",
    "Standard deviation of period", "Fraction of unvoiced",
    "Number of voice breaks", "Degree of voice breaks",
    "Jitter local", "Jitter local absolute", "Jitter rap",
    "Jitter ppq5", "Jitter ddp", "Shimmer local", "Shimmer local dB",
    "Shimmer apq3", "Shimmer apq5", "Shimmer apq11", "Shimmer dda",
    "Mean autocorrelation", "Mean NHR", "Mean HNR", "Audio Length"
]


def classify_speaker(speaker_raw: str) -> str:
    s = speaker_raw.lower().strip()
    if "operator" in s:
        return "OPERATOR"
    for marker in ANALYST_MARKERS:
        if marker in s:
            return "ANALYST"
    for title in MANAGEMENT_TITLES:
        if title in s:
            if "chief executive" in s or s == "ceo":
                return "CEO"
            if "chief financial" in s or s == "cfo":
                return "CFO"
            return "MANAGEMENT"
    return "UNKNOWN"


def parse_person_label_csv(csv_path: str, call_id: str) -> list[dict]:
    segments = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            person = row.get("Person", "").strip()
            sentence = row.get("Sentence", "").strip()
            if not sentence or len(sentence) < 5:
                continue
            sentence = re.sub(r'\s+', ' ', sentence).strip()
            segments.append({
                "call_id": call_id,
                "utterance_index": i,
                "speaker_raw": person,
                "speaker_role": "UNKNOWN",
                "high_confidence": False,
                "text": sentence,
                "char_count": len(sentence),
                "word_count": len(sentence.split()),
            })
    return segments


def parse_maec_transcript_txt(txt_path: str, call_id: str,
                               speaker_name_map: dict = None) -> list[dict]:
    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    segments = []
    for i, line in enumerate(lines):
        text = line.strip()
        if not text or len(text) < 10:
            continue
        text = re.sub(r'\s+', ' ', text)
        segments.append({
            "call_id": call_id,
            "utterance_index": i,
            "speaker_raw": "UNKNOWN",
            "speaker_role": "UNKNOWN",
            "high_confidence": False,
            "text": text,
            "char_count": len(text),
            "word_count": len(text.split()),
        })
    return segments


def parse_maec_features_csv(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    vals = df[numeric_cols].copy()
    vals = vals.replace("--undefined--", np.nan)
    vals = vals.apply(pd.to_numeric, errors="coerce")
    return vals.values


def parse_maec_features_summary(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = {}
    for col in numeric_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        summary[f"{col}_mean"] = float(vals.mean())
        summary[f"{col}_std"] = float(vals.std())
        summary[f"{col}_max"] = float(vals.max())
        summary[f"{col}_min"] = float(vals.min())
    return summary


def ingest_maec_dataset(maec_root: str, use_person_label: bool = True,
                        speaker_name_map: dict = None) -> list[dict]:
    root = Path(maec_root)
    person_label_dir = root / "MAEC_Dataset_Person_Label"
    main_dataset_dir = root / "MAEC_Dataset"

    calls = []
    source_dir = person_label_dir if use_person_label else main_dataset_dir

    if not source_dir.exists():
        print(f"WARNING: {source_dir} does not exist")
        return calls

    for call_dir in sorted(source_dir.iterdir()):
        if not call_dir.is_dir():
            continue

        parts = call_dir.name.split("_", 1)
        if len(parts) != 2:
            continue
        date_str, ticker = parts

        if len(date_str) < 8:
            continue

        call_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        call_id = f"{ticker}_{date_str}"

        call_meta = {
            "call_id": call_id,
            "ticker": ticker,
            "call_date": call_date,
            "maec_dir": str(call_dir),
            "has_audio_features": False,
        }

        if use_person_label:
            csv_path = call_dir / "text.csv"
            if csv_path.exists():
                call_meta["utterances"] = parse_person_label_csv(str(csv_path), call_id)
            else:
                continue

            main_call_dir = main_dataset_dir / call_dir.name
            feat_path = main_call_dir / "features.csv"
            if feat_path.exists():
                call_meta["has_audio_features"] = True
                call_meta["maec_lld_summary"] = parse_maec_features_summary(str(feat_path))
                call_meta["features_csv_path"] = str(feat_path)
        else:
            txt_path = call_dir / "text.txt"
            feat_path = call_dir / "features.csv"
            if not txt_path.exists():
                continue

            call_meta["utterances"] = parse_maec_transcript_txt(
                str(txt_path), call_id, speaker_name_map
            )
            if feat_path.exists():
                call_meta["has_audio_features"] = True
                call_meta["maec_lld_summary"] = parse_maec_features_summary(str(feat_path))
                call_meta["features_csv_path"] = str(feat_path)

        calls.append(call_meta)

    print(f"Ingested {len(calls)} calls from MAEC")
    return calls


def validate_maec_parse(calls: list[dict]) -> dict:
    stats = {
        "total_calls": len(calls),
        "calls_with_ceo": 0,
        "calls_with_high_conf_ceo": 0,
        "calls_with_unknown_speakers": 0,
        "calls_with_audio_features": 0,
        "avg_utterances_per_call": 0,
        "avg_words_per_call": 0,
        "speaker_role_distribution": {},
        "unknown_rate_pct": 0,
    }

    if len(calls) == 0:
        return stats

    role_counts = {}
    total_utts = 0
    total_words = 0
    total_unknown = 0
    unique_speakers_per_call = []

    for call in calls:
        utts = call.get("utterances", [])
        roles = [u["speaker_role"] for u in utts]
        speakers = set(u["speaker_raw"] for u in utts)
        unique_speakers_per_call.append(len(speakers))

        if "CEO" in roles:
            stats["calls_with_ceo"] += 1
        if any(u["speaker_role"] == "CEO" and u.get("high_confidence") for u in utts):
            stats["calls_with_high_conf_ceo"] += 1
        if "UNKNOWN" in roles:
            stats["calls_with_unknown_speakers"] += 1
        if call.get("has_audio_features"):
            stats["calls_with_audio_features"] += 1

        for role in roles:
            role_counts[role] = role_counts.get(role, 0) + 1
        total_unknown += roles.count("UNKNOWN")

        total_utts += len(utts)
        total_words += sum(u["word_count"] for u in utts)

    stats["avg_utterances_per_call"] = total_utts / len(calls)
    stats["avg_words_per_call"] = total_words / len(calls)
    stats["pct_calls_with_ceo"] = stats["calls_with_ceo"] / len(calls) * 100
    stats["speaker_role_distribution"] = role_counts
    stats["unknown_rate_pct"] = total_unknown / max(total_utts, 1) * 100
    stats["avg_unique_speakers_per_call"] = np.mean(unique_speakers_per_call) if unique_speakers_per_call else 0

    return stats
