import os, torch, numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader


class EarningsCallDataset(Dataset):
    def __init__(self, manifest_path, labels_path, processed_dir, split_range=None,
                 target_col="abnormal_vol_3d", max_utts=200,
                 filter_degenerate=False, filter_zeros=False, rank_transform=False,
                 multi_task=False, multi_task_targets=None,
                 train_quantiles=None):
        manifest = pd.read_csv(manifest_path)
        labels = pd.read_csv(labels_path)

        # Determine which columns to keep from labels
        if multi_task and multi_task_targets:
            label_cols = ["call_id"] + multi_task_targets
            # Also need r2 and beta for filtering
            if filter_degenerate:
                label_cols = list(set(label_cols + ["r2", "beta"]))
        else:
            label_cols = ["call_id", target_col]
            if filter_degenerate:
                label_cols = list(set(label_cols + ["r2", "beta"]))

        # Only keep columns that exist
        label_cols = [c for c in label_cols if c in labels.columns]
        merged = manifest.merge(labels[label_cols], on="call_id", how="inner")

        # Filter degenerate CAPM estimates (placeholders, not real measurements)
        if filter_degenerate and "r2" in merged.columns and "beta" in merged.columns:
            before = len(merged)
            # Only remove exact CAPM placeholders (both conditions together)
            merged = merged[~(
                (merged["r2"] >= 0.999) &
                (merged["beta"] == 1.0)
            )]
            after = len(merged)
            if before != after:
                print(f"    Filtered {before - after} degenerate labels (r²≥0.999 AND β=1.0)")
            # Drop the helper columns
            merged = merged.drop(columns=["r2", "beta"], errors="ignore")

        # Drop rows missing the primary target
        merged = merged.dropna(subset=[target_col])

        # Filter exact-zero volatility (from broken CAPM fits)
        if filter_zeros:
            before = len(merged)
            if multi_task and multi_task_targets:
                # Remove rows where ALL multi-task targets are zero
                zero_mask = True
                for col in multi_task_targets:
                    if col in merged.columns:
                        zero_mask = zero_mask & (merged[col] == 0.0)
                merged = merged[~zero_mask]
            else:
                merged = merged[merged[target_col] != 0.0]
            after = len(merged)
            if before != after:
                print(f"    Filtered {before - after} zero-volatility samples")

        if split_range:
            start, end = split_range
            merged = merged[(merged["call_date"] >= start) & (merged["call_date"] <= end)]

        merged = merged.reset_index(drop=True)

        # Multi-task targets setup
        self.multi_task = multi_task and multi_task_targets is not None
        self.multi_task_targets = multi_task_targets or []
        self.target_col = target_col

        # Rank-transform (quantile normalization)
        self.rank_transform = rank_transform
        self.train_quantiles = train_quantiles  # Dict of {col: sorted_values} from train set

        if self.rank_transform:
            if self.multi_task:
                target_cols = self.multi_task_targets
            else:
                target_cols = [target_col]

            if train_quantiles is None:
                # This is the train set — compute and store quantiles
                self.train_quantiles = {}
                for col in target_cols:
                    if col in merged.columns:
                        vals = merged[col].dropna().values
                        self.train_quantiles[col] = np.sort(vals)
                        merged[col] = _rank_transform(merged[col].values, self.train_quantiles[col])
            else:
                # This is val/test — use train quantiles
                for col in target_cols:
                    if col in merged.columns and col in train_quantiles:
                        merged[col] = _rank_transform(merged[col].values, train_quantiles[col])

        self.records = merged
        self.processed_dir = processed_dir
        self.max_utts = max_utts

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        pt_path = os.path.join(self.processed_dir, f"{row['call_id']}.pt")
        data = torch.load(pt_path, weights_only=True)

        n = min(data["n_utterances"], self.max_utts)
        acoustic = data["acoustic"][:n]
        text_emb = data["text_emb"][:n]
        wav_emb = data["wav_emb"][:n]

        if self.multi_task and self.multi_task_targets:
            labels = []
            for col in self.multi_task_targets:
                val = row.get(col, float("nan"))
                labels.append(float(val) if not pd.isna(val) else 0.0)
            label_tensor = torch.tensor(labels, dtype=torch.float32)
        else:
            label_tensor = torch.tensor(float(row[self.target_col]), dtype=torch.float32)

        return {
            "acoustic": acoustic,
            "text_emb": text_emb,
            "wav_emb": wav_emb,
            "label": label_tensor,
            "n_utts": n,
            "call_id": row["call_id"],
        }


def _log_transform(values):
    """Log-transform to spread right-skewed volatility values."""
    return np.log1p(np.abs(values) * 100).astype(np.float32)


def _rank_transform(values, reference_sorted):
    """Quantile-normalize values using reference distribution (train set).
    Applies log1p(val*100) before ranking to improve resolution in the skewed tail."""
    # Log-transform both values and reference before ranking
    log_values = _log_transform(values)
    log_ref = np.sort(_log_transform(reference_sorted))
    
    result = np.empty_like(values, dtype=np.float32)
    n_ref = len(log_ref)
    for i, v in enumerate(log_values):
        if np.isnan(v):
            result[i] = 0.5  # Default for missing
        else:
            # Find the rank of v in the reference distribution
            rank = np.searchsorted(log_ref, v, side="right")
            result[i] = rank / n_ref  # Normalize to [0, 1]
    return result


def collate_fn(batch):
    max_n = max(b["n_utts"] for b in batch)
    B = len(batch)
    a_dim = batch[0]["acoustic"].shape[1]

    acoustic = torch.zeros(B, max_n, a_dim)
    text_emb = torch.zeros(B, max_n, 768)
    wav_emb = torch.zeros(B, max_n, 768)
    mask = torch.ones(B, max_n, dtype=torch.bool)
    call_ids = []

    # Handle both multi-task (2D) and single-task (1D) labels
    sample_label = batch[0]["label"]
    if sample_label.dim() == 0:
        labels = torch.zeros(B)
    else:
        labels = torch.zeros(B, sample_label.shape[0])

    for i, b in enumerate(batch):
        n = b["n_utts"]
        acoustic[i, :n] = b["acoustic"]
        text_emb[i, :n] = b["text_emb"]
        wav_emb[i, :n] = b["wav_emb"]
        mask[i, :n] = False
        labels[i] = b["label"]
        call_ids.append(b["call_id"])

    return {
        "acoustic": acoustic,
        "text_emb": text_emb,
        "wav_emb": wav_emb,
        "mask": mask,
        "labels": labels,
        "call_ids": call_ids,
    }


def get_dataloaders(cfg):
    manifest = os.path.join(cfg["data"]["processed_dir"], "manifest.csv")
    labels = cfg["data"]["labels_path"]
    proc_dir = cfg["data"]["processed_dir"]
    target = cfg["label"]["primary_target"]
    max_utts = cfg["data"]["max_utterances"]
    bs = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 0)
    pin_memory = cfg["training"].get("pin_memory", torch.cuda.is_available())

    filter_deg = cfg["label"].get("filter_degenerate", False)
    rank_tf = cfg["label"].get("rank_transform", False)
    multi_task = cfg["label"].get("multi_task", False)
    mt_targets = cfg["label"].get("multi_task_targets", None)

    filter_zeros = cfg["label"].get("filter_zeros", False)

    # Build train set first (to get quantiles for rank transform)
    train_ds = EarningsCallDataset(
        manifest, labels, proc_dir,
        split_range=cfg["splits"]["train"],
        target_col=target, max_utts=max_utts,
        filter_degenerate=filter_deg, filter_zeros=filter_zeros,
        rank_transform=rank_tf,
        multi_task=multi_task, multi_task_targets=mt_targets,
        train_quantiles=None,  # Train computes its own
    )
    train_quantiles = train_ds.train_quantiles if rank_tf else None

    loaders = {}
    loaders["train"] = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory,
    )
    print(f"  train: {len(train_ds)} calls")

    for split_name in ["val", "test"]:
        ds = EarningsCallDataset(
            manifest, labels, proc_dir,
            split_range=cfg["splits"][split_name],
            target_col=target, max_utts=max_utts,
            filter_degenerate=filter_deg, filter_zeros=filter_zeros,
            rank_transform=rank_tf,
            multi_task=multi_task, multi_task_targets=mt_targets,
            train_quantiles=train_quantiles,
        )
        loaders[split_name] = DataLoader(
            ds, batch_size=bs, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory,
        )
        print(f"  {split_name}: {len(ds)} calls")

    return loaders
