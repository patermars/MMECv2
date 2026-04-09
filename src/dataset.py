import os, torch, numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader


class EarningsCallDataset(Dataset):
    def __init__(self, manifest_path, labels_path, processed_dir, split_range=None,
                 target_col="abnormal_vol_3d", max_utts=200):
        manifest = pd.read_csv(manifest_path)
        labels = pd.read_csv(labels_path)
        merged = manifest.merge(labels[["call_id", target_col]], on="call_id", how="inner")
        merged = merged.dropna(subset=[target_col])

        if split_range:
            start, end = split_range
            merged = merged[(merged["call_date"] >= start) & (merged["call_date"] <= end)]

        self.records = merged.reset_index(drop=True)
        self.processed_dir = processed_dir
        self.target_col = target_col
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
        label = float(row[self.target_col])

        return {
            "acoustic": acoustic,
            "text_emb": text_emb,
            "wav_emb": wav_emb,
            "label": torch.tensor(label, dtype=torch.float32),
            "n_utts": n,
            "call_id": row["call_id"],
        }


def collate_fn(batch):
    max_n = max(b["n_utts"] for b in batch)
    B = len(batch)
    a_dim = batch[0]["acoustic"].shape[1]

    acoustic = torch.zeros(B, max_n, a_dim)
    text_emb = torch.zeros(B, max_n, 768)
    wav_emb = torch.zeros(B, max_n, 768)
    mask = torch.ones(B, max_n, dtype=torch.bool)
    labels = torch.zeros(B)
    call_ids = []

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

    loaders = {}
    for split_name, date_range in cfg["splits"].items():
        ds = EarningsCallDataset(manifest, labels, proc_dir,
                                 split_range=date_range, target_col=target,
                                 max_utts=max_utts)
        shuffle = (split_name == "train")
        loaders[split_name] = DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        print(f"  {split_name}: {len(ds)} calls")

    return loaders
