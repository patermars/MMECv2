"""
Dataset assembler for ACL19.

ACL19 gives us pre-segmented sentence-level MP3s with 1:1 text alignment,
so the pipeline is:
  parse -> extract features -> encode text -> build labels -> assemble

No diarization, no forced alignment, no audio scraping.
"""

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


def assemble_acl19_dataset(
    calls: list[dict],
    labels_df: pd.DataFrame,
    device: str = "cuda",
) -> list[dict]:
    """
    Run the full per-call feature extraction pipeline and merge with labels.

    Pipeline per call:
      1. Extract eGeMAPS from each sentence MP3
      2. Encode text with FinBERT
      3. Extract LM linguistic features
      4. Merge with volatility labels
    """
    records = []
    attrition = {
        "started":    len(calls),
        "no_label":   0,
        "no_utts":    0,
        "final":      0,
    }

    total = len(calls)
    for idx, call in enumerate(calls, 1):
        call_id = call["call_id"]
        print(f"\n[{idx}/{total}] Processing {call_id} ({call['company']})")

        # Labels
        label_row = labels_df[labels_df["call_id"] == call_id]
        if label_row.empty or pd.isna(label_row["abnormal_vol_3d"].values[0]):
            print(f"  [skip] No label for {call_id}")
            attrition["no_label"] += 1
            continue

        # Audio features
        print(f"  [audio] Extracting eGeMAPS from {call['n_utterances']} sentences...")
        enriched_utts, drop_stats = extract_features_for_call(call["utterances"])
        print(f"  [audio] Kept {drop_stats['n_kept']}/{drop_stats['n_total']} "
              f"({drop_stats['drop_rate_pct']:.1f}% dropped)")

        if not enriched_utts:
            print(f"  [skip] No valid utterances after feature extraction")
            attrition["no_utts"] += 1
            continue

        # Text embeddings
        texts = [u["text"] for u in enriched_utts]
        print(f"  [text] Encoding {len(texts)} sentences with FinBERT...")
        text_embs = encode_batch(texts, device=device)

        # Linguistic features
        ling_feats = np.stack([
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
            "utterances":   [
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
            "labels":      label_row.iloc[0].to_dict(),
            "drop_stats":  drop_stats,
        }
        records.append(record)
        print(f"  [ok] {len(enriched_utts)} utterances assembled")

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
