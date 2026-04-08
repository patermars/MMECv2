"""
PyTorch Dataset for ACL19 earnings call checkpoints.
Loads pre-processed call records and prepares tensors for HierarchicalMultimodalEncoder.
Supports multi-window targets (1d, 3d, 7d) to 3× the dataset, rank-based normalization,
and training-time feature augmentation.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


STRUCTURED_COLS = ["estimation_beta", "earnings_surprise", "market_cap_log", "vix_at_call", "hist_vol_30d"]
EVENT_WINDOWS = ["abnormal_vol_1d", "abnormal_vol_3d", "abnormal_vol_7d"]


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


class RankTransformer:
    """Maps raw labels to uniform [0, 1] via rank-based quantile normalization."""

    def __init__(self):
        self.sorted_values = None

    def fit(self, values: np.ndarray):
        self.sorted_values = np.sort(values)
        return self

    def transform(self, value: float) -> float:
        if self.sorted_values is None:
            return value
        rank = np.searchsorted(self.sorted_values, value, side="right")
        return rank / len(self.sorted_values)

    def inverse_transform(self, rank: float) -> float:
        if self.sorted_values is None:
            return rank
        idx = int(np.clip(rank * len(self.sorted_values), 0, len(self.sorted_values) - 1))
        return float(self.sorted_values[idx])


class EarningsCallDataset(Dataset):
    def __init__(self, split_csv: str, checkpoint_dir: str, labels_csv: str,
                 augment: bool = False, aug_config: dict = None,
                 rank_transformer: RankTransformer = None,
                 target_transform: str = "rank",
                 multi_window: bool = False):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.labels_df = pd.read_csv(labels_csv)
        self.augment = augment
        self.aug_config = aug_config or {}
        self.target_transform = target_transform
        self.multi_window = multi_window

        split_df = pd.read_csv(split_csv)

        # Build sample list: (call_id, target_col) pairs
        self.samples = []
        target_cols = EVENT_WINDOWS if multi_window else ["abnormal_vol_3d"]

        for call_id in split_df["call_id"].tolist():
            ckpt = self.checkpoint_dir / f"{call_id}.json"
            row  = self.labels_df[self.labels_df["call_id"] == call_id]
            if not ckpt.exists() or row.empty:
                continue
            try:
                with open(ckpt, encoding="utf-8") as f:
                    json.load(f)
            except (PermissionError, OSError, json.JSONDecodeError):
                print(f"  [dataset] Skipping unreadable checkpoint: {call_id}")
                continue

            for target_col in target_cols:
                if not pd.isna(row.iloc[0].get(target_col, np.nan)):
                    self.samples.append((call_id, target_col))

        if multi_window:
            window_counts = {}
            for _, tc in self.samples:
                window_counts[tc] = window_counts.get(tc, 0) + 1
            print(f"  [dataset] Multi-window samples: {window_counts}")

        self._structured_mean, self._structured_std = self._compute_stats()

        # Label transformation
        if rank_transformer is not None:
            self.rank_transformer = rank_transformer
        else:
            self.rank_transformer = self._fit_rank_transformer()

    def _compute_stats(self):
        seen = set()
        vals = []
        for call_id, _ in self.samples:
            if call_id in seen:
                continue
            seen.add(call_id)
            row = self.labels_df[self.labels_df["call_id"] == call_id]
            if row.empty:
                continue
            vals.append([_safe_float(row.iloc[0][c]) for c in STRUCTURED_COLS])
        if not vals:
            return np.zeros(len(STRUCTURED_COLS), dtype=np.float32), np.ones(len(STRUCTURED_COLS), dtype=np.float32)
        arr = np.array(vals, dtype=np.float32)
        return arr.mean(0), arr.std(0).clip(min=1e-6)

    def _fit_rank_transformer(self):
        """Fit rank transformer on ALL labels across all windows (for multi-window)."""
        labels = []
        for _, target_col in self.samples:
            # Get raw labels from all windows combined
            pass
        # Collect all labels
        labels = []
        for call_id, target_col in self.samples:
            row = self.labels_df[self.labels_df["call_id"] == call_id]
            if not row.empty:
                labels.append(_safe_float(row.iloc[0][target_col]))
        rt = RankTransformer()
        if labels:
            rt.fit(np.array(labels, dtype=np.float32))
            print(f"  [dataset] Rank transformer fitted on {len(labels)} labels "
                  f"(range [{min(labels):.6f}, {max(labels):.6f}])")
        return rt

    @property
    def call_ids(self):
        """Unique call_ids for backward compatibility."""
        return list(dict.fromkeys(cid for cid, _ in self.samples))

    def __len__(self):
        return len(self.samples)

    def _augment_features(self, features: np.ndarray) -> np.ndarray:
        """Apply Gaussian noise and random feature dropout during training."""
        noise_std = self.aug_config.get("feature_noise_std", 0.05)
        drop_p = self.aug_config.get("feature_dropout_p", 0.15)
        features = features + np.random.randn(*features.shape).astype(np.float32) * noise_std
        mask = np.random.rand(*features.shape) > drop_p
        features = features * mask.astype(np.float32)
        return features

    def __getitem__(self, idx):
        call_id, target_col = self.samples[idx]
        try:
            record = _load_checkpoint(self.checkpoint_dir / f"{call_id}.json")
        except (PermissionError, OSError, json.JSONDecodeError) as e:
            print(f"  [dataset] Read error on {call_id}: {e}, using idx 0")
            return self.__getitem__(0)
        utts = record["utterances"]

        remarks_utts, qa_utts = _split_utterances(utts)
        if not remarks_utts:
            remarks_utts = qa_utts
        if not qa_utts:
            qa_utts = remarks_utts

        remarks_audio, remarks_text = _utt_to_arrays(remarks_utts)
        qa_audio,      qa_text      = _utt_to_arrays(qa_utts)

        # Apply augmentation during training only
        if self.augment:
            remarks_audio = self._augment_features(remarks_audio)
            qa_audio = self._augment_features(qa_audio)

        label_row  = self.labels_df[self.labels_df["call_id"] == call_id].iloc[0]
        raw_label  = _safe_float(label_row[target_col])

        # Apply rank transform
        if self.target_transform == "rank":
            label = self.rank_transformer.transform(raw_label)
        else:
            label = raw_label

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
