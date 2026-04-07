"""
PyTorch Dataset for ACL19 earnings call checkpoints.
Loads pre-processed call records and prepares tensors for HierarchicalMultimodalEncoder.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


STRUCTURED_COLS = ["estimation_beta", "earnings_surprise", "market_cap_log", "vix_at_call", "hist_vol_30d"]


def _safe_float(val):
    """Return float(val) or 0.0 if val is NaN/None."""
    try:
        v = float(val)
        return 0.0 if np.isnan(v) else v
    except (TypeError, ValueError):
        return 0.0


def _load_checkpoint(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    for u in record["utterances"]:
        u["text_embedding"] = np.array(u["text_embedding"], dtype=np.float32)
        u["audio_features"] = np.array(u["audio_features"], dtype=np.float32)
        u["ling_features"]  = np.array(u["ling_features"],  dtype=np.float32)
    return record


def _split_utterances(utterances):
    mid = max(1, len(utterances) // 2)
    return utterances[:mid], utterances[mid:]


def _utt_to_arrays(utts):
    audio = np.stack([u["audio_features"] for u in utts])
    text  = np.stack([u["text_embedding"]  for u in utts])
    return audio, text


class EarningsCallDataset(Dataset):
    def __init__(self, split_csv: str, checkpoint_dir: str, labels_csv: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.labels_df = pd.read_csv(labels_csv)

        split_df = pd.read_csv(split_csv)
        valid = []
        for call_id in split_df["call_id"].tolist():
            ckpt = self.checkpoint_dir / f"{call_id}.json"
            row  = self.labels_df[self.labels_df["call_id"] == call_id]
            if not ckpt.exists() or row.empty or pd.isna(row.iloc[0]["abnormal_vol_3d"]):
                continue
            try:
                with open(ckpt, encoding="utf-8") as f:
                    json.load(f)
                valid.append(call_id)
            except (PermissionError, OSError, json.JSONDecodeError):
                print(f"  [dataset] Skipping unreadable checkpoint: {call_id}")
        self.call_ids = valid

        self._structured_mean, self._structured_std = self._compute_stats()

    def _compute_stats(self):
        vals = []
        for call_id in self.call_ids:
            row = self.labels_df[self.labels_df["call_id"] == call_id]
            if row.empty:
                continue
            vals.append([_safe_float(row.iloc[0][c]) for c in STRUCTURED_COLS])
        if not vals:
            return np.zeros(len(STRUCTURED_COLS), dtype=np.float32), np.ones(len(STRUCTURED_COLS), dtype=np.float32)
        arr = np.array(vals, dtype=np.float32)
        return arr.mean(0), arr.std(0).clip(min=1e-6)

    def __len__(self):
        return len(self.call_ids)

    def __getitem__(self, idx):
        call_id   = self.call_ids[idx]
        try:
            record = _load_checkpoint(self.checkpoint_dir / f"{call_id}.json")
        except (PermissionError, OSError, json.JSONDecodeError) as e:
            # Fall back to a neighbouring sample rather than crashing the epoch
            print(f"  [dataset] Read error on {call_id}: {e}, using idx 0")
            return self.__getitem__(0)
        utts      = record["utterances"]

        remarks_utts, qa_utts = _split_utterances(utts)
        if not remarks_utts:
            remarks_utts = qa_utts
        if not qa_utts:
            qa_utts = remarks_utts

        remarks_audio, remarks_text = _utt_to_arrays(remarks_utts)
        qa_audio,      qa_text      = _utt_to_arrays(qa_utts)

        label_row  = self.labels_df[self.labels_df["call_id"] == call_id].iloc[0]
        label      = _safe_float(label_row["abnormal_vol_3d"])
        structured = np.array(
            [_safe_float(label_row[c]) for c in STRUCTURED_COLS], dtype=np.float32
        )
        structured = (structured - self._structured_mean) / self._structured_std

        return {
            "remarks_audio": remarks_audio,
            "remarks_text":  remarks_text,
            "qa_audio":      qa_audio,
            "qa_text":       qa_text,
            "structured":    structured,
            "label":         label,
        }
