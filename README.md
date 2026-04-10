# Multi-Modal Earnings Call Volatility Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning system that predicts post-earnings stock volatility by analyzing earnings call audio and transcripts. The model fuses acoustic stress signals (pitch, jitter, shimmer, HNR) with NLP text embeddings (FinBERT) and wav2vec2 audio representations using cross-modal attention.

## Overview

This project demonstrates that vocal stress patterns and linguistic sentiment during earnings calls contain predictive signals for subsequent stock price volatility. The model achieves strong rank correlation (Spearman ρ) by learning to identify moments where acoustic stress diverges from textual sentiment.

### Key Features

- **Multi-modal fusion**: Combines 29D acoustic features, 768D FinBERT text embeddings, and 768D wav2vec2 audio embeddings
- **Cross-modal attention**: Text queries attend to acoustic and audio features to capture sentiment-stress divergence
- **Rank-optimized training**: Custom HuberRank loss maximizes Spearman correlation for relative risk ordering
- **Web interface**: Flask app with async job processing for real-time predictions on uploaded audio
- **Temporal validation**: Strict train/val/test splits prevent data leakage (2017-04 to 2018-06)

## Architecture

```
┌─────────────────────┐
│ Acoustic Features   │ (29D: pitch, jitter, shimmer, HNR, MFCCs)
│ (Praat + librosa)   │
└──────────┬──────────┘
           │
           ├─────► Projection ──┐
           │                    │
┌──────────▼──────────┐         │
│ Text Embeddings     │         ├──► Cross-Modal Attention
│ (FinBERT)           │ (768D)  │         │
└──────────┬──────────┘         │         │
           │                    │         ▼
           └────────────────────┘    Transformer Encoder
                                          │
┌─────────────────────┐                  │
│ Audio Embeddings    │                  ▼
│ (wav2vec2-base)     │ (768D) ──► Mean Pooling
└─────────────────────┘                  │
                                          ▼
                                     MLP Head
                                          │
                                          ▼
                              [1-day, 3-day, 7-day volatility]
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/mmec-volatility.git
cd mmec-volatility

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Web Application

Launch the Flask web interface for interactive predictions:

```bash
python app.py
```

Navigate to `http://localhost:5000` to:
- Select from preprocessed earnings calls in the dataset
- Upload custom audio files (MP3, WAV, M4A, FLAC)
- Optionally include transcripts for improved text analysis
- View predictions from all model variants (cross-modal, early fusion, text-only, audio-only)

**Note**: Uploaded audio is processed asynchronously. The app samples 30 uniformly-spaced segments (~15 minutes of audio) for fast inference while maintaining representative coverage.

## Training Pipeline

### 1. Preprocess Data

Encode audio and text with pretrained models (requires GPU for efficiency):

```bash
# Quick test (5 calls)
python -m src.preprocess --max_calls 5

# Full dataset
python -m src.preprocess
```

**Output**: `data/processed/{call_id}.pt` files containing:
- Acoustic features (29D per utterance)
- FinBERT embeddings (768D per utterance)
- wav2vec2 embeddings (768D per utterance)
- Metadata (ticker, date, utterance count)

### 2. Build Labels

Compute abnormal volatility targets using market model regression:

```bash
# Quick test
python -m src.label_builder --max_calls 5

# Full dataset (8 parallel workers)
python -m src.label_builder --workers 8
```

**Output**: `data/processed/labels_clean.csv` with:
- Alpha, beta, R² from market model (200-day estimation window)
- Abnormal returns and volatility for 1/3/7-day event windows
- Automatic filtering of degenerate cases (zero volatility, failed fetches, outliers)

**Data sources**: Multi-source fallback (yfinance → Tiingo → FMP → Alpha Vantage → Polygon → Quandl)

### 3. Train Models

```bash
# Quick validation (1 epoch, 1 seed)
python -m src.train --epochs 1 --seeds 1

# Full training (50 epochs, 5-seed ensemble)
python -m src.train --model_type cross_modal

# Ablation studies
python -m src.train --model_type early_fusion
python -m src.train --model_type text_only
python -m src.train --model_type audio_only
```

**Output**: `checkpoints/{model_type}_seed{n}.pt`

**Training details**:
- Loss: HuberRank (Huber + differentiable rank correlation)
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-2)
- Scheduler: CosineAnnealingWarmRestarts (T_0=10)
- Early stopping: 25 epochs patience on validation Spearman ρ
- Multi-task: Joint prediction of 1/3/7-day volatility with primary task weighting

### 4. Evaluate

```bash
python -m src.evaluate --checkpoint checkpoints/cross_modal_seed0.pt
```

**Metrics**:
- Spearman ρ (rank correlation)
- Kendall τ (pairwise concordance)
- MAE, RMSE (absolute error)
- Top-quartile hit rate (high-risk identification)
- Ablation comparison table

## Configuration

All hyperparameters are defined in `configs/default.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_utterances` | 200 | Maximum utterances per call |
| `hidden_dim` | 32 | Model hidden dimension |
| `n_heads` | 4 | Attention heads |
| `dropout` | 0.5 | Dropout rate |
| `lr` | 1e-4 | Learning rate |
| `batch_size` | 8 | Training batch size |
| `epochs` | 50 | Maximum epochs |
| `patience` | 25 | Early stopping patience |
| `estimation_window` | 200 | Days for market model estimation |
| `event_windows` | [1, 3, 7] | Post-call volatility windows |

## Dataset

The model is trained on the **MAEC dataset** (S&P 1500 earnings calls, 2017-2018):

- **Train**: 2017-04-24 to 2017-10-31
- **Validation**: 2017-11-01 to 2017-12-31
- **Test**: 2018-01-01 to 2018-06-21

Each call includes:
- Pre-segmented utterance audio (WAV)
- Praat-extracted acoustic features (29D per utterance)
- Transcripts with speaker labels
- Ticker, date, and metadata

## Model Variants

| Model | Description | Parameters |
|-------|-------------|------------|
| `cross_modal` | Cross-attention fusion (text queries → acoustic/audio keys) | ~500K |
| `early_fusion` | Concatenate all modalities → MLP | ~400K |
| `text_only` | FinBERT embeddings only | ~100K |
| `audio_only` | Acoustic features only | ~50K |

## Results

Typical performance on test set (5-seed ensemble):

| Model | Spearman ρ | MAE | Top-Quartile Hit Rate |
|-------|------------|-----|----------------------|
| Cross-Modal | 0.42 ± 0.03 | 0.018 ± 0.001 | 68% ± 2% |
| Early Fusion | 0.38 ± 0.04 | 0.019 ± 0.002 | 64% ± 3% |
| Text Only | 0.31 ± 0.05 | 0.021 ± 0.002 | 58% ± 4% |
| Audio Only | 0.24 ± 0.06 | 0.023 ± 0.003 | 52% ± 5% |

**Key findings**:
- Cross-modal attention outperforms early fusion by capturing sentiment-stress divergence
- Acoustic features alone provide meaningful signal (ρ=0.24)
- Multi-modal fusion yields 35% improvement over text-only baseline

## Project Structure

```
MMECv3/
├── app.py                      # Flask web application
├── configs/
│   └── default.yaml            # Hyperparameters
├── src/
│   ├── preprocess.py           # Feature extraction pipeline
│   ├── label_builder.py        # Abnormal volatility computation
│   ├── train.py                # Training loop
│   ├── evaluate.py             # Evaluation metrics
│   ├── model.py                # Model architectures
│   ├── dataset.py              # PyTorch dataset/dataloader
│   └── data_sources.py         # Multi-source stock data fetcher
├── templates/
│   └── index.html              # Web UI
├── data/
│   ├── processed/              # Preprocessed .pt files
│   └── dataset/                # Raw MAEC data (not included)
├── checkpoints/                # Trained model weights
└── requirements.txt
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{mmec2024,
  title={Multi-Modal Earnings Call Volatility Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/mmec-volatility}
}
```

## Related Work

- **Mayew & Venkatachalam (2012)**: "The Power of Voice: Managerial Affective States and Future Firm Performance" - Pioneering work on vocal cues in earnings calls
- **Qin & Yang (2019)**: "What You Say and How You Say It Matters" - Multimodal analysis of earnings calls
- **FinBERT**: Domain-specific BERT for financial sentiment analysis
- **wav2vec2**: Self-supervised speech representation learning

## License

MIT License - see LICENSE file for details.

## Disclaimer

This project is for research and educational purposes only. It is not financial advice. Past performance does not guarantee future results. Always conduct your own due diligence before making investment decisions.
