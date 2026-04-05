"""
MMECv2 Data Pipeline — ACL19 edition.

Phases:
  0  Parse ACL19 dataset (text + audio paths)
  1  Build volatility labels from yfinance
  2  Assemble full dataset (features + labels)

Usage (on Colab):
  python run_pipeline.py --phase 0 --acl19-root /content/ACL19_Release
  python run_pipeline.py --phase 1
  python run_pipeline.py --phase 2
"""

import argparse
import yaml
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Phase 0: Parse ACL19 ──────────────────────────────────────────────────────

def run_phase_0(config: dict, acl19_root: str = None):
    print("=== Phase 0: ACL19 Ingestion ===")
    from src.data_pipeline.acl19_parser import (
        ingest_acl19_dataset, validate_acl19_parse
    )

    root = acl19_root or config["paths"]["acl19_root"]

    # Optional ticker map — place at data/raw/ticker_map.json if you have one
    ticker_map = {}
    ticker_map_path = Path("data/raw/ticker_map.json")
    if ticker_map_path.exists():
        with open(ticker_map_path) as f:
            ticker_map = json.load(f)
        print(f"Loaded ticker map: {len(ticker_map)} entries")

    calls = ingest_acl19_dataset(root, ticker_map=ticker_map)

    stats = validate_acl19_parse(calls)
    print("\nParse stats:")
    print(json.dumps(stats, indent=2, default=str))

    out_dir = Path(config["paths"]["transcripts"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for call in calls:
        out_path = out_dir / f"{call['call_id']}_utterances.json"
        with open(out_path, "w") as f:
            json.dump(call, f, indent=2)

    print(f"\nSaved {len(calls)} call records to {out_dir}")
    return calls


# ── Phase 1: Build Labels ─────────────────────────────────────────────────────

def run_phase_1(config: dict):
    print("=== Phase 1: Label Construction ===")
    from src.data_pipeline.label_builder import build_labels_for_acl19_calls

    transcripts_dir = Path(config["paths"]["transcripts"])
    calls = []
    for f in sorted(transcripts_dir.glob("*_utterances.json")):
        with open(f) as fp:
            calls.append(json.load(fp))

    print(f"Building labels for {len(calls)} calls...")
    labels_df = build_labels_for_acl19_calls(calls)
    print(labels_df[["call_id", "ticker", "call_date",
                      "abnormal_vol_3d", "earnings_surprise"]].to_string())
    return labels_df


# ── Phase 2: Assemble Dataset ─────────────────────────────────────────────────

def run_phase_2(config: dict):
    print("=== Phase 2: Dataset Assembly ===")
    import pandas as pd
    import torch
    from src.data_pipeline.dataset_assembler import assemble_acl19_dataset, create_splits

    transcripts_dir = Path(config["paths"]["transcripts"])
    calls = []
    for f in sorted(transcripts_dir.glob("*_utterances.json")):
        with open(f) as fp:
            calls.append(json.load(fp))

    labels_path = Path(config["paths"]["labels"]) / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels not found at {labels_path}. Run --phase 1 first."
        )
    labels_df = pd.read_csv(labels_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    records = assemble_acl19_dataset(calls, labels_df, device=device)

    splits = create_splits(records, config["paths"]["splits"])
    print(f"\nDataset ready: {len(records)} calls across "
          f"train/val/test = "
          f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    return records


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MMECv2 Data Pipeline (ACL19)")
    parser.add_argument("--phase", type=int, required=True,
                        choices=[0, 1, 2],
                        help="0=parse, 1=labels, 2=assemble")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--acl19-root", type=str, default=None,
                        help="Path to ACL19_Release folder "
                             "(overrides config; e.g. /content/ACL19_Release)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.dry_run:
        print("Config OK:")
        print(yaml.dump(config, default_flow_style=False))
        return

    phase_map = {
        0: lambda c: run_phase_0(c, args.acl19_root),
        1: run_phase_1,
        2: run_phase_2,
    }
    phase_map[args.phase](config)


if __name__ == "__main__":
    main()
