import argparse
import yaml
import torch
from dotenv import load_dotenv
import os

load_dotenv()


def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_single_experiment(config: dict, exp_id: str, device: str):
    from src.training.ablation_runner import AblationRunner
    print(f"Running experiment: {exp_id}")
    print(f"Device: {device}")
    print("NOTE: You must provide train/val/test datasets as PyTorch datasets.")
    print("This requires running the data pipeline first (run_pipeline.py).")


def run_ablation(config: dict, device: str, experiments: list = None):
    from src.training.ablation_runner import AblationRunner
    print(f"Running ablation study on device: {device}")
    print("NOTE: You must provide train/val/test datasets as PyTorch datasets.")
    print("This requires running the data pipeline first (run_pipeline.py).")


def run_training(config: dict, device: str):
    print("=== Training Full Model (EXP-08) ===")
    from src.models.hierarchical_encoder import HierarchicalMultimodalEncoder

    mc = config.get("model", {})
    model = HierarchicalMultimodalEncoder(
        audio_dim=mc.get("audio_dim", 88),
        text_dim=mc.get("text_dim", 768),
        hidden_dim=mc.get("hidden_dim", 256),
        n_heads=mc.get("n_heads", 4),
        n_structured=mc.get("n_structured", 5),
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print("NOTE: Provide DataLoaders to Trainer class to begin training.")
    print("See README.md for usage instructions.")


def main():
    parser = argparse.ArgumentParser(description="MMEC v2 Training")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "ablation", "single"],
                        help="Training mode")
    parser.add_argument("--exp-id", type=str, default="EXP-08",
                        help="Experiment ID (for single mode)")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Config file path")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Specific experiments to run (for ablation mode)")
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
