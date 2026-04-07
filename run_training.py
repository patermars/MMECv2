import argparse
import yaml
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from src.models.hierarchical_encoder import collate_calls
from src.data.dataset import EarningsCallDataset

load_dotenv()

CHECKPOINT_DIR = "data/processed/checkpoints"
LABELS_CSV     = "data/processed/labels/labels.csv"
SPLITS_DIR     = "data/splits"


def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_dataloaders(config: dict):
    batch_size = config.get("training", {}).get("batch_size", 16)
    train_ds = EarningsCallDataset(f"{SPLITS_DIR}/train.csv", CHECKPOINT_DIR, LABELS_CSV)
    val_ds   = EarningsCallDataset(f"{SPLITS_DIR}/val.csv",   CHECKPOINT_DIR, LABELS_CSV)
    test_ds  = EarningsCallDataset(f"{SPLITS_DIR}/test.csv",  CHECKPOINT_DIR, LABELS_CSV)
    print(f"Dataset sizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_calls, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_calls, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_calls, num_workers=0)
    return train_loader, val_loader, test_loader


def run_training(config: dict, device: str):
    print("=== Training Full Model (EXP-08) ===")
    from src.models.hierarchical_encoder import HierarchicalMultimodalEncoder
    from src.training.trainer import Trainer

    mc = config.get("model", {})
    model = HierarchicalMultimodalEncoder(
        audio_dim=mc.get("audio_dim", 88),
        text_dim=mc.get("text_dim", 768),
        hidden_dim=mc.get("hidden_dim", 256),
        n_heads=mc.get("n_heads", 4),
        n_structured=mc.get("n_structured", 5),
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,} total")

    train_loader, val_loader, _ = build_dataloaders(config)
    trainer = Trainer(model, train_loader, val_loader, config, device=device)
    results = trainer.train()
    print(f"\nTraining complete. Best Spearman ρ: {results['best_spearman']:.4f}")


def run_ablation(config: dict, device: str, experiments: list = None):
    from src.training.ablation_runner import AblationRunner
    print(f"Running ablation study on device: {device}")
    train_loader, val_loader, test_loader = build_dataloaders(config)
    runner = AblationRunner(config, device)
    runner.run_all(train_loader, val_loader, test_loader, experiments=experiments)


def run_single_experiment(config: dict, exp_id: str, device: str):
    from src.training.ablation_runner import AblationRunner
    print(f"Running experiment: {exp_id}")
    train_loader, val_loader, test_loader = build_dataloaders(config)
    runner = AblationRunner(config, device)
    runner.run_experiment(exp_id, train_loader, val_loader, test_loader)


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
