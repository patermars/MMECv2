# MMEC v2 — Multi-Modal Earnings Call Analysis

Predicts post-earnings abnormal volatility from multi-modal features (audio + text + structured) extracted from the MAEC dataset.

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Copy and fill environment variables
copy .env.example .env
# Edit .env with your HuggingFace token and (optional) W&B key
```

### 2. Get MAEC Dataset

```bash
# Clone into data/raw/maec/
git clone https://github.com/Earnings-Call-Dataset/MAEC-A-Multimodal-Aligned-Earnings-Conference-Call-Dataset-for-Financial-Risk-Prediction.git data/raw/maec
```

### 3. Run the Pipeline

Each phase runs independently. Run them in order:

```bash
# Phase 0 — Parse MAEC transcripts
python run_pipeline.py --phase 0

# Phase 1 — Download audio from IR pages (slow, expect ~60% success)
python run_pipeline.py --phase 1

# Phase 4 — Build volatility labels from yfinance
python run_pipeline.py --phase 4

# Phase 5 — Create temporal train/val/test splits
python run_pipeline.py --phase 5
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
│   │   ├── maec_parser.py        # Parse MAEC transcripts + LLD features
│   │   ├── maec_audio_scraper.py # Download audio from IR pages
│   │   ├── audio_preprocessor.py # 16kHz mono, EBU R128 normalization
│   │   ├── diarization.py        # Speaker-first pyannote + WhisperX alignment
│   │   ├── whisperx_aligner.py   # Forced alignment (text → timestamps)
│   │   ├── feature_extractor.py  # eGeMAPS (88-dim) utterance features
│   │   ├── maec_text_cleaner.py  # OCR cleanup + section segmentation
│   │   ├── text_encoder.py       # FinBERT CLS embeddings (768-dim)
│   │   ├── linguistic_features.py# Loughran-McDonald sentiment (7-dim)
│   │   ├── label_builder.py      # Event-study OLS + earnings surprise
│   │   ├── dataset_assembler.py  # Merge all outputs + temporal splits
│   │   └── dataset_validator.py  # Shape/NaN/distribution checks
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

- **eGeMAPS (88 features)** over ComParE (6,373) — better suited for ~900-call dataset
- **Speaker-first diarization** — pyannote runs first, then WhisperX aligns text within each speaker segment
- **Forced alignment** — MAEC text is ground truth; WhisperX assigns timestamps, not re-transcribe
- **Event-study OLS** with proper α + β estimation over 200-day window
- **Earnings surprise** included as confounder in all experiment variants
- **Primary metric**: Spearman rank correlation (ρ)
- **Loss**: 0.7 × MSE + 0.3 × Pairwise Ranking

## Requirements

- Python 3.10+
- CUDA GPU (recommended for pyannote, WhisperX, FinBERT)
- ffmpeg (required by whisperx/yt-dlp)
- HuggingFace token with pyannote model access

## Citation

```bibtex
@inproceedings{CIKM2020MAEC,
  author    = {Li, Jiazheng and Yang, Linyi and Smyth, Barry and Dong, Ruihai},
  title     = {MAEC: A Multimodal Aligned Earnings Conference Call Dataset},
  booktitle = {CIKM '20},
  year      = {2020},
}
```
