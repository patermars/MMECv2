import argparse
import yaml
import torch
import numpy as np
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from src.models.hierarchical_encoder import HierarchicalMultimodalEncoder, collate_calls
from src.data.dataset import EarningsCallDataset

load_dotenv()

CHECKPOINT_DIR = "data/processed/checkpoints"
LABELS_CSV     = "data/processed/labels/labels.csv"
SPLITS_DIR     = "data/splits"


def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_datasets(config: dict):
    aug_config = config.get("augmentation", {})
    tc = config.get("training", {})
    target_transform = tc.get("target_transform", "rank")
    multi_window = tc.get("multi_window", False)

    train_ds = EarningsCallDataset(
        f"{SPLITS_DIR}/train.csv", CHECKPOINT_DIR, LABELS_CSV,
        augment=True, aug_config=aug_config,
        target_transform=target_transform,
        multi_window=multi_window,
    )
    val_ds = EarningsCallDataset(
        f"{SPLITS_DIR}/val.csv", CHECKPOINT_DIR, LABELS_CSV,
        augment=False,
        rank_transformer=train_ds.rank_transformer,
        target_transform=target_transform,
        multi_window=multi_window,
    )
    test_ds = EarningsCallDataset(
        f"{SPLITS_DIR}/test.csv", CHECKPOINT_DIR, LABELS_CSV,
        augment=False,
        rank_transformer=train_ds.rank_transformer,
        target_transform=target_transform,
        multi_window=multi_window,
    )
    print(f"Dataset sizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


def build_dataloaders(config: dict):
    batch_size = config.get("training", {}).get("batch_size", 16)
    train_ds, val_ds, test_ds = build_datasets(config)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_calls, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_calls, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_calls, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader


def build_model(config: dict):
    mc = config.get("model", {})
    return HierarchicalMultimodalEncoder(
        audio_dim=mc.get("audio_dim", 88),
        text_dim=mc.get("text_dim", 768),
        hidden_dim=mc.get("hidden_dim", 128),
        n_heads=mc.get("n_heads", 2),
        n_structured=mc.get("n_structured", 5),
        dropout=mc.get("dropout", 0.4),
        n_layers=mc.get("n_layers", 1),
    )


def run_training(config: dict, device: str):
    from src.training.trainer import Trainer

    print("=== Training Full Model (EXP-08) ===")
    model = build_model(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,} total")

    train_loader, val_loader, _ = build_dataloaders(config)
    trainer = Trainer(model, train_loader, val_loader, config, device=device)
    results = trainer.train()
    print(f"\nTraining complete. Best Spearman ρ: {results['best_spearman']:.4f}")


def run_ablation(config: dict, device: str, experiments: list = None):
    from src.training.ablation_runner import AblationRunner
    print(f"Running ablation study on device: {device}")
    train_ds, val_ds, test_ds = build_datasets(config)
    runner = AblationRunner(train_ds, val_ds, test_ds, config, device)
    runner.run_all(experiments)


def run_single_experiment(config: dict, exp_id: str, device: str):
    from src.training.ablation_runner import AblationRunner
    print(f"Running experiment: {exp_id}")
    train_ds, val_ds, test_ds = build_datasets(config)
    runner = AblationRunner(train_ds, val_ds, test_ds, config, device)
    runner.run_experiment(exp_id)


def main():
    parser = argparse.ArgumentParser(description="MMEC v2 Training")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "ablation", "single"])
    parser.add_argument("--exp-id", type=str, default="EXP-08")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--experiments", nargs="+", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.get("training", {}).get("seed", 42))

    if args.mode == "train":
        run_training(config, device)
    elif args.mode == "ablation":
        run_ablation(config, device, args.experiments)
    elif args.mode == "single":
        run_single_experiment(config, args.exp_id, device)


if __name__ == "__main__":
    main()
