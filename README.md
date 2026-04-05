# MMEC v2 — Multi-Modal Earnings Call Analysis

Predicts post-earnings abnormal volatility from multi-modal features (audio + text + structured) extracted from the ACL19 earnings call dataset (Qin & Yang, ACL 2019).

## Quick Start

### 1. Setup Environment

```bash
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

pip install -r requirements.txt

cp .env.example .env
# Edit .env with your HuggingFace token and (optional) W&B key
```

### 2. Get ACL19 Dataset

Download from [Google Drive](https://drive.google.com/drive/folders/1BKCANORbcmUJKkOkBOghw6uNHPqS_az1), then:

```bash
zip -s0 ACL19_Release.zip --out ACL19_Release_All.zip
unzip -q ACL19_Release_All.zip
# Results in: ACL19_Release/{CompanyName_YYYYMMDD}/CEO/*.mp3 + TextSequence.txt
```

Optionally place a `{company: ticker}` JSON map at `data/raw/ticker_map.json` for accurate ticker resolution.

### 3. Run the Pipeline

```bash
# Phase 0 — Parse ACL19 dataset (text + audio paths)
python run_pipeline.py --phase 0 --acl19-root /path/to/ACL19_Release

# Phase 1 — Build volatility labels from yfinance
python run_pipeline.py --phase 1

# Phase 2 — Assemble full dataset (features + labels + splits)
python run_pipeline.py --phase 2
```

### 4. Train Models

```bash
# Train the full hierarchical model (EXP-08)
python run_training.py --mode train

# Run full ablation study (all 12 experiments)
python run_training.py --mode ablation

# Run a specific experiment
python run_training.py --mode single --exp-id EXP-06
```

### 5. Dry Run (verify config)

```bash
python run_pipeline.py --phase 0 --dry-run
```

## Project Structure

```
MMECv2/
├── configs/
│   └── default.yaml              # All hyperparameters and paths
├── src/
│   ├── data_pipeline/
│   │   ├── acl19_parser.py       # Parse ACL19 folders → call records
│   │   ├── label_builder.py      # Event-study OLS + earnings surprise
│   │   ├── dataset_assembler.py  # Merge features + labels + temporal splits
│   │   ├── dataset_validator.py  # Shape/NaN/distribution checks
│   │   ├── feature_extractor.py  # eGeMAPS (88-dim) utterance features
│   │   ├── text_encoder.py       # FinBERT CLS embeddings (768-dim)
│   │   └── linguistic_features.py# Loughran-McDonald sentiment (7-dim)
│   ├── models/
│   │   ├── hierarchical_encoder.py   # 3-level cross-modal attention
│   │   ├── volatility_head.py        # Regression head
│   │   └── fusion/
│   │       ├── early_fusion.py       # Concatenate + MLP baseline
│   │       ├── late_fusion.py        # Separate towers baseline
│   │       └── cross_modal_attn.py   # Attention module
│   ├── training/
│   │   ├── losses.py             # MSE + Pairwise ranking loss
│   │   ├── trainer.py            # Training loop + early stopping
│   │   └── ablation_runner.py    # 12-experiment ablation suite
│   └── evaluation/
│       ├── metrics.py            # Spearman, Kendall, MAE, RMSE
│       └── interpretability.py   # SHAP + visualization
├── run_pipeline.py               # Data pipeline CLI
├── run_training.py               # Training CLI
├── requirements.txt
├── .env.example
└── solution.md                   # Full solution document
```

## Key Design Decisions

- **ACL19 dataset** — CEO-only sentence-level audio aligned with text; no diarization needed
- **eGeMAPS (88 features)** over ComParE (6,373) — better suited for ~500-call dataset
- **Event-study OLS** with proper α + β estimation over 200-day window
- **Earnings surprise** included as confounder in all experiment variants
- **Primary metric**: Spearman rank correlation (ρ)
- **Loss**: 0.7 × MSE + 0.3 × Pairwise Ranking

## Requirements

- Python 3.10+
- CUDA GPU (recommended for FinBERT)
- ffmpeg
- HuggingFace token (for pyannote model access if needed)

## Citation

```bibtex
@inproceedings{qin-yang-2019-say,
  author    = {Qin, Yu and Yang, Yi},
  title     = {What You Say and How You Say It Matters: Predicting Financial Risk Using Verbal and Vocal Cues},
  booktitle = {ACL 2019},
  year      = {2019},
}
```
