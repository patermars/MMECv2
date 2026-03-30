import argparse
import yaml
import json
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()


def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_phase_0(config: dict):
    print("=== Phase 0: MAEC Ingestion ===")
    from src.data_pipeline.maec_parser import ingest_maec_dataset, validate_maec_parse

    maec_root = config["paths"]["maec_root"]
    calls = ingest_maec_dataset(maec_root, use_person_label=True)

    stats = validate_maec_parse(calls)
    print(json.dumps(stats, indent=2, default=str))

    out_dir = Path(config["paths"]["transcripts"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for call in calls:
        out_path = out_dir / f"{call['call_id']}_utterances.json"
        serializable = {k: v for k, v in call.items() if k != "maec_lld_summary"}
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)

    print(f"Saved {len(calls)} parsed transcripts to {out_dir}")
    return calls


def run_phase_1(config: dict):
    print("=== Phase 1: Audio Download ===")
    from src.data_pipeline.maec_audio_scraper import batch_download_maec_audio

    transcripts_dir = Path(config["paths"]["transcripts"])
    calls = []
    for f in sorted(transcripts_dir.glob("*_utterances.json")):
        with open(f) as fp:
            calls.append(json.load(fp))

    audio_dir = config["paths"]["audio_raw"]
    results = batch_download_maec_audio(calls, audio_dir)
    print(f"Downloaded: {len(results['success'])}, Failed: {len(results['failed'])}")
    return results


def run_phase_4(config: dict):
    print("=== Phase 4: Label Construction ===")
    from src.data_pipeline.label_builder import build_labels_for_maec_calls

    transcripts_dir = Path(config["paths"]["transcripts"])
    calls = []
    for f in sorted(transcripts_dir.glob("*_utterances.json")):
        with open(f) as fp:
            calls.append(json.load(fp))

    labels_df = build_labels_for_maec_calls(calls)
    print(f"Built labels for {len(labels_df)} calls")
    print(labels_df.describe())
    return labels_df


def run_phase_5(config: dict):
    print("=== Phase 5: Dataset Assembly ===")
    from src.data_pipeline.dataset_assembler import create_temporal_splits
    import pandas as pd

    labels_path = Path(config["paths"]["labels"]) / "labels.csv"
    labels_df = pd.read_csv(labels_path)

    splits = create_temporal_splits(labels_df, config["splits"]["maec"],
                                    config["paths"]["splits"])
    return splits


def main():
    parser = argparse.ArgumentParser(description="MMEC v2 Data Pipeline")
    parser.add_argument("--phase", type=int, required=True,
                        help="Pipeline phase to run (0-5)")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Config file path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load config and validate without running")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.dry_run:
        print("Config loaded successfully:")
        print(yaml.dump(config, default_flow_style=False))
        return

    phase_map = {
        0: run_phase_0,
        1: run_phase_1,
        4: run_phase_4,
        5: run_phase_5,
    }

    if args.phase not in phase_map:
        print(f"Phase {args.phase} not implemented as standalone. "
              f"Available: {list(phase_map.keys())}")
        return

    phase_map[args.phase](config)


if __name__ == "__main__":
    main()
