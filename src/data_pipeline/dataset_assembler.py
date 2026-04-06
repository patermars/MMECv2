"""
Dataset assembler for ACL19.

ACL19 gives us pre-segmented sentence-level MP3s with 1:1 text alignment,
so the pipeline is:
  parse -> extract features -> encode text -> build labels -> assemble

No diarization, no forced alignment, no audio scraping.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from .feature_extractor import extract_features_for_call
from .text_encoder import encode_batch
from .linguistic_features import extract_lm_features
from .label_builder import build_labels_for_acl19_calls

# ACL19 is all 2017 — single-year dataset, so splits are by company not time
ACL19_SPLITS = {
    "train": 0.7,
    "val":   0.15,
    "test":  0.15,
}

CHECKPOINT_DIR = Path("data/processed/checkpoints")


def _checkpoint_path(call_id: str) -> Path:
    return CHECKPOINT_DIR / f"{call_id}.json"


def _save_checkpoint(record: dict):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(record["call_id"])
    # numpy arrays → lists for JSON serialisation
    serialisable = {**record}
    serialisable["utterances"] = [
        {**u,
         "text_embedding": u["text_embedding"].tolist(),
         "audio_features": u["audio_features"].tolist(),
         "ling_features":  u["ling_features"].tolist()}
        for u in record["utterances"]
    ]
    with open(path, "w") as f:
        json.dump(serialisable, f)


def _load_checkpoint(call_id: str) -> dict | None:
    path = _checkpoint_path(call_id)
    if not path.exists():
        return None
    with open(path) as f:
        record = json.load(f)
    # lists → numpy arrays
    for u in record["utterances"]:
        u["text_embedding"] = np.array(u["text_embedding"], dtype=np.float32)
        u["audio_features"] = np.array(u["audio_features"], dtype=np.float32)
        u["ling_features"]  = np.array(u["ling_features"],  dtype=np.float32)
    return record



def assemble_acl19_dataset(
    calls: list[dict],
    labels_df: pd.DataFrame,
    device: str = "cpu",
) -> list[dict]:
    """
    Run the full per-call feature extraction pipeline and merge with labels.
    Checkpoints each call to data/processed/checkpoints/ so crashes resume.
    """
    attrition = {"started": len(calls), "no_label": 0, "no_utts": 0, "final": 0}
    total = len(calls)
    records = []

    for idx, call in enumerate(calls, 1):
        call_id = call["call_id"]

        cached = _load_checkpoint(call_id)
        if cached is not None:
            print(f"[{idx}/{total}] RESUME {call_id}")
            records.append(cached)
            continue

        label_row = labels_df[labels_df["call_id"] == call_id]
        if label_row.empty or pd.isna(label_row["abnormal_vol_3d"].values[0]):
            print(f"[{idx}/{total}] SKIP {call_id}: no label")
            attrition["no_label"] += 1
            continue

        print(f"[{idx}/{total}] Processing {call_id} ({call['company']})")
        try:
            enriched_utts, drop_stats = extract_features_for_call(call["utterances"])
            if not enriched_utts:
                print(f"  SKIP: no valid utterances")
                attrition["no_utts"] += 1
                continue

            texts     = [u["text"] for u in enriched_utts]
            text_embs = encode_batch(texts, device=device)
            ling_feats  = np.stack([
                np.array(list(extract_lm_features(u["text"]).values()))
                for u in enriched_utts
            ])
            audio_feats = np.stack([u["audio_features"] for u in enriched_utts])

            record = {
                "call_id":      call_id,
                "ticker":       call["ticker"],
                "company":      call["company"],
                "call_date":    call["call_date"],
                "speaker":      call["speaker"],
                "n_utterances": len(enriched_utts),
                "utterances": [
                    {
                        "utterance_index": u["utterance_index"],
                        "text":            u["text"],
                        "speaker_role":    u["speaker_role"],
                        "duration_s":      u["duration_s"],
                        "audio_path":      u["audio_path"],
                        "text_embedding":  text_embs[i],
                        "audio_features":  audio_feats[i],
                        "ling_features":   ling_feats[i],
                    }
                    for i, u in enumerate(enriched_utts)
                ],
                "labels":     label_row.iloc[0].to_dict(),
                "drop_stats": drop_stats,
            }
            _save_checkpoint(record)
            records.append(record)
            print(f"  OK: {len(enriched_utts)} utterances")
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    attrition["final"] = len(records)
    print(f"\n=== Assembly complete: {len(records)}/{total} calls ===")
    print(f"    Attrition: {attrition}")
    return records


def create_splits(
    records: list[dict],
    output_dir: str,
    split_ratios: dict = None,
    seed: int = 42,
) -> dict:
    """
    Random train/val/test split (ACL19 is single-year, no temporal ordering).
    Saves call_id CSVs to output_dir.
    """
    import random
    random.seed(seed)

    split_ratios = split_ratios or ACL19_SPLITS
    shuffled = records.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * split_ratios["train"])
    n_val   = int(n * split_ratios["val"])

    splits = {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
        df = pd.DataFrame([{
            "call_id":   r["call_id"],
            "ticker":    r["ticker"],
            "company":   r["company"],
            "call_date": r["call_date"],
            "n_utterances": r["n_utterances"],
            "abnormal_vol_3d": r["labels"].get("abnormal_vol_3d"),
        } for r in subset])
        df.to_csv(f"{output_dir}/{name}.csv", index=False)
        print(f"{name}: {len(subset)} calls")

    return splits
