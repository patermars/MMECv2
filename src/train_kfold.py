import os, argparse, yaml, torch
import numpy as np
import pandas as pd
from src.train import train_single_seed, _build_criterion
from src.model import build_model
from src.dataset import EarningsCallDataset, collate_fn
from torch.utils.data import DataLoader


def create_temporal_folds(manifest_path, labels_path, k=5):
    """Create K temporal folds from the dataset."""
    manifest = pd.read_csv(manifest_path)
    labels = pd.read_csv(labels_path)
    # Manifest already has call_date, just use it directly
    merged = manifest.merge(labels[["call_id"]], on="call_id", how="inner")
    merged = merged.sort_values("call_date").reset_index(drop=True)
    
    dates = merged["call_date"].unique()
    dates = np.sort(dates)
    
    fold_size = len(dates) // k
    folds = []
    
    for i in range(k):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < k - 1 else len(dates)
        fold_dates = dates[start_idx:end_idx]
        folds.append([fold_dates[0], fold_dates[-1]])
    
    # Verify temporal ordering - no overlaps between folds
    for i in range(len(folds) - 1):
        assert folds[i][1] < folds[i+1][0], f"Temporal overlap between folds {i} and {i+1}: {folds[i][1]} >= {folds[i+1][0]}"
    
    return folds


def get_kfold_dataloaders(cfg, train_folds, val_fold, test_fold):
    """Create dataloaders for specific fold configuration."""
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
    
    # Combine train folds
    df = pd.read_csv(manifest)
    train_mask = pd.Series([False] * len(df))
    for fold in train_folds:
        train_mask |= (df["call_date"] >= fold[0]) & (df["call_date"] <= fold[1])
    
    train_dates = df[train_mask]["call_date"]
    train_range = [train_dates.min(), train_dates.max()]
    
    train_ds = EarningsCallDataset(
        manifest, labels, proc_dir,
        split_range=train_range,
        target_col=target, max_utts=max_utts,
        filter_degenerate=filter_deg, filter_zeros=filter_zeros,
        rank_transform=rank_tf,
        multi_task=multi_task, multi_task_targets=mt_targets,
        train_quantiles=None,
    )
    train_quantiles = train_ds.train_quantiles if rank_tf else None
    
    loaders = {}
    loaders["train"] = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory,
    )
    
    val_ds = EarningsCallDataset(
        manifest, labels, proc_dir,
        split_range=val_fold,
        target_col=target, max_utts=max_utts,
        filter_degenerate=filter_deg, filter_zeros=filter_zeros,
        rank_transform=rank_tf,
        multi_task=multi_task, multi_task_targets=mt_targets,
        train_quantiles=train_quantiles,
    )
    loaders["val"] = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory,
    )
    
    test_ds = EarningsCallDataset(
        manifest, labels, proc_dir,
        split_range=test_fold,
        target_col=target, max_utts=max_utts,
        filter_degenerate=filter_deg, filter_zeros=filter_zeros,
        rank_transform=rank_tf,
        multi_task=multi_task, multi_task_targets=mt_targets,
        train_quantiles=train_quantiles,
    )
    loaders["test"] = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory,
    )
    
    print(f"  train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")
    return loaders


def train_fold(cfg, fold_idx, train_folds, val_fold, test_fold, seed, device, model_type):
    """Train a single fold."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx} | Seed {seed}")
    print(f"  Train: {[f'{f[0]} to {f[1]}' for f in train_folds]}")
    print(f"  Val: {val_fold[0]} to {val_fold[1]}")
    print(f"  Test: {test_fold[0]} to {test_fold[1]}")
    print(f"{'='*60}")
    
    loaders = get_kfold_dataloaders(cfg, train_folds, val_fold, test_fold)
    model = build_model(cfg, model_type).to(device)
    
    criterion = _build_criterion(cfg)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["training"]["lr"],
                                  weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    scaler = torch.amp.GradScaler(device, enabled=(device != "cpu"))
    
    multi_task = cfg["label"].get("multi_task", False)
    mt_targets = cfg["label"].get("multi_task_targets", [])
    primary_target = cfg["label"].get("primary_target", "abnormal_vol_3d")
    primary_idx = None
    if multi_task and mt_targets and primary_target in mt_targets:
        primary_idx = mt_targets.index(primary_target)
    
    from src.train import train_epoch, evaluate
    
    best_val_score = -float("inf")
    patience_count = 0
    best_state = None
    
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss, train_rho = train_epoch(
            model, loaders["train"], optimizer, scaler,
            device, cfg["training"]["grad_clip"], criterion, primary_idx)
        val_metrics = evaluate(model, loaders["val"], device, primary_idx)
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  E{epoch:03d} | train_rho={train_rho:.4f} | "
                  f"val_rho={val_metrics['spearman']:.4f}")
        
        if val_metrics["n"] > 0:
            current_score = val_metrics["spearman"]
            if current_score > best_val_score:
                best_val_score = current_score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
        
        if patience_count >= cfg["training"]["patience"]:
            print(f"  Early stop at epoch {epoch}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    test_metrics = evaluate(model, loaders["test"], device, primary_idx)
    print(f"  Test: rho={test_metrics['spearman']:.4f} mae={test_metrics['mae']:.5f}")
    
    return test_metrics


def run_kfold_validation(cfg, args, device):
    """Run K-fold cross-validation."""
    manifest = os.path.join(cfg["data"]["processed_dir"], "manifest.csv")
    labels = cfg["data"]["labels_path"]
    folds = create_temporal_folds(manifest, labels, k=args.k)
    
    print(f"\nTemporal folds:")
    for i, fold in enumerate(folds):
        print(f"  Fold {i}: {fold[0]} to {fold[1]}")
    
    all_fold_metrics = []
    
    for fold_idx in range(args.k):
        test_fold = folds[fold_idx]
        # Use only 1 fold for validation (standard K-fold)
        val_idx = (fold_idx + 1) % args.k
        val_fold = folds[val_idx]
        train_folds = [folds[i] for i in range(args.k) 
                      if i not in [fold_idx, val_idx]]
        
        fold_metrics = []
        for seed_idx in range(args.seeds):
            seed = seed_idx * 42
            metrics = train_fold(cfg, fold_idx, train_folds, val_fold, test_fold,
                               seed, device, args.model_type)
            fold_metrics.append(metrics)
        
        avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0].keys()}
        all_fold_metrics.append(avg_metrics)
        print(f"\nFold {fold_idx} avg: rho={avg_metrics['spearman']:.4f} mae={avg_metrics['mae']:.5f}")
    
    print(f"\n{'='*60}")
    print(f"K-Fold CV Results (K={args.k})")
    print(f"{'='*60}")
    for key in ["spearman", "mae", "mse"]:
        vals = [m[key] for m in all_fold_metrics]
        print(f"  {key}: {np.mean(vals):.5f} ± {np.std(vals):.5f}")


def run_holdout_validation(cfg, args, device):
    """Run holdout validation (70% train, 15% val, 15% test)."""
    manifest = os.path.join(cfg["data"]["processed_dir"], "manifest.csv")
    labels = cfg["data"]["labels_path"]
    
    # Get all dates and split temporally
    df = pd.read_csv(manifest)
    labels_df = pd.read_csv(labels)
    merged = df.merge(labels_df[["call_id"]], on="call_id", how="inner")
    merged = merged.sort_values("call_date").reset_index(drop=True)
    
    dates = merged["call_date"].unique()
    dates = np.sort(dates)
    
    n_dates = len(dates)
    train_end = int(0.7 * n_dates)
    val_end = int(0.85 * n_dates)
    
    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]
    
    train_range = [train_dates[0], train_dates[-1]]
    val_range = [val_dates[0], val_dates[-1]]
    test_range = [test_dates[0], test_dates[-1]]
    
    print(f"\nHoldout split:")
    print(f"  Train: {train_range[0]} to {train_range[1]} ({len(train_dates)} dates)")
    print(f"  Val: {val_range[0]} to {val_range[1]} ({len(val_dates)} dates)")
    print(f"  Test: {test_range[0]} to {test_range[1]} ({len(test_dates)} dates)")
    
    all_metrics = []
    for seed_idx in range(args.seeds):
        seed = seed_idx * 42
        metrics = train_holdout_seed(cfg, train_range, val_range, test_range, 
                                   seed, device, args.model_type)
        all_metrics.append(metrics)
    
    print(f"\n{'='*60}")
    print(f"Holdout Validation Results ({args.seeds} seeds)")
    print(f"{'='*60}")
    for key in ["spearman", "mae", "mse"]:
        vals = [m[key] for m in all_metrics]
        print(f"  {key}: {np.mean(vals):.5f} ± {np.std(vals):.5f}")


def run_original_split(cfg, args, device):
    """Run with original fixed temporal split."""
    # Use the original train/val/test split from your config
    train_range = ["2017-04-24", "2017-10-31"]
    val_range = ["2017-11-01", "2017-12-31"]
    test_range = ["2018-01-01", "2018-06-21"]
    
    print(f"\nOriginal split:")
    print(f"  Train: {train_range[0]} to {train_range[1]}")
    print(f"  Val: {val_range[0]} to {val_range[1]}")
    print(f"  Test: {test_range[0]} to {test_range[1]}")
    
    all_metrics = []
    for seed_idx in range(args.seeds):
        seed = seed_idx * 42
        metrics = train_holdout_seed(cfg, train_range, val_range, test_range, 
                                   seed, device, args.model_type)
        all_metrics.append(metrics)
    
    print(f"\n{'='*60}")
    print(f"Original Split Results ({args.seeds} seeds)")
    print(f"{'='*60}")
    for key in ["spearman", "mae", "mse"]:
        vals = [m[key] for m in all_metrics]
        print(f"  {key}: {np.mean(vals):.5f} ± {np.std(vals):.5f}")


def train_holdout_seed(cfg, train_range, val_range, test_range, seed, device, model_type):
    """Train a single seed with holdout validation."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"\n{'='*60}")
    print(f"Seed {seed}")
    print(f"{'='*60}")
    
    # Create single fold with train/val/test ranges
    train_folds = [train_range]
    val_fold = val_range
    test_fold = test_range
    
    loaders = get_kfold_dataloaders(cfg, train_folds, val_fold, test_fold)
    model = build_model(cfg, model_type).to(device)
    
    criterion = _build_criterion(cfg)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["training"]["lr"],
                                  weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    scaler = torch.amp.GradScaler(device, enabled=(device != "cpu"))
    
    multi_task = cfg["label"].get("multi_task", False)
    mt_targets = cfg["label"].get("multi_task_targets", [])
    primary_target = cfg["label"].get("primary_target", "abnormal_vol_3d")
    primary_idx = None
    if multi_task and mt_targets and primary_target in mt_targets:
        primary_idx = mt_targets.index(primary_target)
    
    from src.train import train_epoch, evaluate
    
    best_val_score = -float("inf")
    patience_count = 0
    best_state = None
    
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss, train_rho = train_epoch(
            model, loaders["train"], optimizer, scaler,
            device, cfg["training"]["grad_clip"], criterion, primary_idx)
        val_metrics = evaluate(model, loaders["val"], device, primary_idx)
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  E{epoch:03d} | train_rho={train_rho:.4f} | "
                  f"val_rho={val_metrics['spearman']:.4f}")
        
        if val_metrics["n"] > 0:
            current_score = val_metrics["spearman"]
            if current_score > best_val_score:
                best_val_score = current_score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
        
        if patience_count >= cfg["training"]["patience"]:
            print(f"  Early stop at epoch {epoch}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    test_metrics = evaluate(model, loaders["test"], device, primary_idx)
    print(f"  Test: rho={test_metrics['spearman']:.4f} mae={test_metrics['mae']:.5f}")
    
    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model_type", default="cross_modal",
                        choices=["cross_modal", "audio_only", "text_only", "early_fusion"])
    parser.add_argument("--k", type=int, default=5, help="Number of folds")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=1, help="Seeds per fold")
    parser.add_argument("--validation_strategy", default="kfold", 
                        choices=["kfold", "holdout", "original"],
                        help="Validation strategy: kfold, holdout (70/15/15), or original (fixed temporal split)")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if args.validation_strategy == "original":
        print("Using original fixed temporal split (2017-2018)")
        run_original_split(cfg, args, device)
    elif args.validation_strategy == "holdout":
        print("Using holdout validation (70% train, 15% val, 15% test)")
        run_holdout_validation(cfg, args, device)
    else:
        print(f"K-fold temporal CV: K={args.k}, seeds={args.seeds}")
        run_kfold_validation(cfg, args, device)


if __name__ == "__main__":
    main()
