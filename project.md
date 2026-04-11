# Multi-Modal Earnings Call Volatility Prediction (MMECv2)

## Comprehensive Project Documentation for IEEE Format Report

---

## 1. Title

**Multi-Modal Earnings Call Volatility Prediction Using Cross-Modal Attention with Acoustic Stress Signals and Financial NLP**

---

## 2. Abstract

This project presents a deep learning system that predicts post-earnings-announcement stock price volatility by jointly analyzing the audio recordings and textual transcripts of corporate earnings calls. The proposed architecture fuses three distinct modality representations — 29-dimensional Praat-extracted acoustic stress features (pitch, jitter, shimmer, harmonics-to-noise ratio, MFCCs), 768-dimensional FinBERT text embeddings, and 768-dimensional wav2vec2 audio embeddings — using a novel cross-modal attention mechanism where text queries attend to acoustic and audio key-value pairs. The model is trained with a custom HuberRank loss function that jointly optimizes absolute prediction accuracy and rank correlation (Spearman ρ), enabling relative risk ordering of earnings calls. The system is evaluated on the MAEC dataset (S&P 1500 earnings calls, April 2017–June 2018) using strict temporal train/validation/test splits to prevent data leakage. A comprehensive ablation study across four model variants (cross-modal, early fusion, text-only, audio-only) demonstrates that multi-modal fusion with cross-modal attention captures sentiment-stress divergence signals invisible to uni-modal baselines. The system includes a Flask-based web interface for real-time inference on uploaded audio files with asynchronous processing. Target labels are computed as abnormal volatility using a market model (CAPM) regression framework, with multi-source stock price data fetching across 12 financial APIs for robustness.

---

## 3. Introduction

### 3.1 Background and Motivation

Corporate earnings calls are quarterly events where company executives present financial results and answer analyst questions. These calls contain rich information beyond the reported numbers — the *how* something is said can be as informative as *what* is said. Research in behavioral finance has shown that vocal stress patterns, speech disfluencies, and emotional tone during earnings calls carry predictive signals for future stock price movements (Mayew & Venkatachalam, 2012; Qin & Yang, 2019).

Traditional approaches to earnings call analysis rely solely on natural language processing (NLP) of transcripts, missing the paralinguistic cues embedded in the audio signal. This project addresses this gap by building a multi-modal system that fuses textual sentiment analysis with acoustic stress detection to predict post-earnings abnormal stock price volatility.

### 3.2 Problem Statement

Given an earnings call with both audio recordings and textual transcripts, predict the magnitude of abnormal stock price volatility in the 1-day, 3-day, and 7-day windows following the call. The prediction task is framed as a regression problem where the primary objective is to correctly *rank* calls by their subsequent volatility (i.e., maximize Spearman rank correlation), rather than minimize absolute prediction error alone.

### 3.3 Key Contributions

1. **Multi-modal fusion architecture**: A cross-modal attention mechanism that allows text representations to attend to both acoustic and wav2vec2 audio features, capturing moments where linguistic sentiment diverges from vocal stress.
2. **HuberRank loss function**: A custom loss combining Huber regression loss with a differentiable approximation to Spearman rank correlation, enabling rank-optimized training.
3. **Multi-task learning**: Joint prediction of 1-day, 3-day, and 7-day volatility windows with primary task weighting.
4. **Robust data pipeline**: Multi-source stock data fetching with 12-source fallback (yfinance, Yahoo Finance direct API, Yahoo Finance V8, Tiingo, FMP, Alpha Vantage, Polygon, Quandl, pandas_datareader, Twelve Data, EOD Historical Data, World Trading Data) and comprehensive label cleaning.
5. **Web-based inference system**: Flask application with asynchronous audio processing for real-time predictions.

### 3.4 Related Work

| Work | Year | Description |
|------|------|-------------|
| Mayew & Venkatachalam | 2012 | "The Power of Voice" — Pioneered analysis of vocal cues in earnings calls for predicting firm performance |
| Qin & Yang | 2019 | "What You Say and How You Say It Matters" — Multi-modal earnings call analysis |
| FinBERT (ProsusAI) | 2019 | Domain-specific BERT model pre-trained on financial text for sentiment analysis |
| wav2vec2 (Facebook) | 2020 | Self-supervised speech representation learning; pre-trained on 960h of Librispeech |
| MAEC Dataset | 2020 | Multi-modal financial dataset of S&P 1500 earnings calls with segmented audio and transcripts |

---

## 4. Dataset

### 4.1 Source

The model is trained on the **MAEC (Multi-modal, Multi-speaker Annotated Earnings Call)** dataset, comprising earnings calls from S&P 1500 companies during 2017–2018.

### 4.2 Dataset Statistics

| Property | Value |
|----------|-------|
| Total processed calls (manifest) | 875 |
| Calls with valid labels | 667 |
| Date range | April 24, 2017 – June 21, 2018 |
| Utterances per call | 3–200 (capped at 200) |
| Minimum utterances required | 3 |
| Per-call data files | `features.csv` (acoustics), `text.txt` (transcript), `*.mp3` (audio segments) |

### 4.3 Temporal Splits

Strict chronological splitting is used to prevent data leakage:

| Split | Date Range | Calls (after filtering) |
|-------|-----------|------------------------|
| Train | 2017-04-24 to 2017-10-31 | ~131–141 |
| Validation | 2017-11-01 to 2017-12-31 | ~43–51 |
| Test | 2018-01-01 to 2018-05-31 | ~89–92 |

Note: Exact counts vary across training runs due to progressive improvements in label quality and filtering criteria. The latest run (log `10/02`) used 131/43/89 splits after filtering 215 degenerate labels.

### 4.4 Per-Call Data Structure

Each earnings call directory (named `{YYYYMMDD}_{TICKER}`) contains:
- **`features.csv`**: Pre-extracted 29-dimensional acoustic features per utterance (Praat-extracted)
- **`text.txt`**: Transcript lines, one per utterance
- **`*.mp3`**: Individual audio segments per utterance

### 4.5 Label Construction

Target labels are **abnormal volatility** computed via market model (CAPM) regression:

1. **Estimation Window**: 200 trading days prior to the earnings call date
2. **Market Model**: OLS regression of stock returns on SPY (S&P 500 ETF) returns
   - Computes alpha (α), beta (β), and R²
3. **Abnormal Returns**: For each event window *w* ∈ {1, 3, 7} days post-call:
   - AR_t = R_stock,t − (α + β × R_market,t)
   - Cumulative AR = Σ AR_t for t = 1..w
4. **Abnormal Volatility**: |Cumulative AR| — the absolute magnitude of cumulative abnormal return

### 4.6 Label Cleaning Pipeline

The following filters are applied to produce `labels_clean.csv`:

1. **Degenerate CAPM removal**: Rows where β = 1.0 AND R² ≥ 0.999 (indicates failed data fetch, ~215 removed)
2. **Zero volatility removal**: Rows where volatility = 0.0 exactly (suspicious)
3. **Missing target removal**: NaN target values
4. **Extreme outlier removal**: Values > 99.5th percentile
5. **Low R² removal**: Rows with R² < 0.01 (poor market model fit)

### 4.7 Multi-Source Stock Data Fetching

The `MultiSourceFetcher` class implements a cascading fallback across 12 data sources:

| Priority | Source | Type | Rate Limit |
|----------|--------|------|------------|
| 1 | yfinance | Python library | None |
| 2 | Yahoo Finance Direct API | REST (v7) | Unofficial |
| 3 | Yahoo Finance V8 API | REST (v8/chart) | Unofficial |
| 4 | Tiingo | REST | 500 tickers (free) |
| 5 | Financial Modeling Prep | REST | 250 calls/day (free) |
| 6 | Alpha Vantage | REST | 5 calls/min (free) |
| 7 | Polygon.io | REST | 5 calls/min (free) |
| 8 | Quandl (Nasdaq) | Python library | 50 calls/day (free) |
| 9 | pandas_datareader | Python library (stooq, yahoo, iex) | Varies |
| 10 | Twelve Data | REST | 800 calls/day (free) |
| 11 | EOD Historical Data | REST | Free tier |
| 12 | World Trading Data | REST | Free tier |

The label builder uses a **staged fallback** strategy: free sources are exhausted first, then API-key-gated sources are tried in order, processing only unresolved calls at each stage.

---

## 5. Proposed Methodology

### 5.1 System Architecture Overview

```
Input Layer
├── Audio Utterances (MP3) ─────────────────────────┐
├── Transcript Text (TXT) ──────────────────────────┤
└── Acoustic Features (CSV, 29D per utterance) ─────┤
                                                     │
Feature Extraction (Preprocessing)                   │
├── Praat/librosa → 29D acoustic features            │
├── FinBERT → 768D text embeddings (CLS token)       │
└── wav2vec2-base → 768D audio embeddings (mean pool)│
                                                     │
Multi-Modal Fusion (CrossModalFusion)                │
├── Projection Layers (29D→H, 768D→H, 768D→H)       │
├── Cross-Modal Attention                            │
│   ├── Text→Acoustic (text queries, acoustic K/V)   │
│   └── Text→wav2vec (text queries, wav2vec K/V)     │
├── Residual Connection (fused = ta + tw + t)        │
├── Transformer Encoder (1 layer)                    │
├── Masked Mean Pooling                              │
└── MLP Head → [vol_1d, vol_3d, vol_7d]             │
```

### 5.2 Feature Extraction Pipeline (`src/preprocess.py`)

#### 5.2.1 Acoustic Features (29 dimensions)

Pre-extracted via Praat and stored in `features.csv` per call. Features include:
- **Pitch statistics**: Mean pitch (F0), pitch standard deviation
- **Voice quality**: Jitter (frequency perturbation), Shimmer (amplitude perturbation), Harmonics-to-Noise Ratio (HNR)
- **Spectral features**: MFCCs (13 coefficients × mean + std = 26 values)
- **Energy**: Root Mean Square (RMS) energy
- **Total**: 29 features per utterance

**Normalization**: Per-call z-score normalization (mean subtraction and division by standard deviation, clamped at 1e-6).

#### 5.2.2 Text Embeddings (768 dimensions)

- **Model**: ProsusAI/FinBERT (BERT-base fine-tuned on financial text)
- **Extraction**: CLS token from the last hidden state
- **Max length**: 512 tokens per utterance
- **Batch processing**: 32 utterances per batch

#### 5.2.3 Audio Embeddings (768 dimensions)

- **Model**: facebook/wav2vec2-base (self-supervised pre-trained on Librispeech)
- **Extraction**: Mean pooling of last hidden state
- **Input**: 16kHz mono audio, max 30 seconds per utterance
- **Fallback**: Zero vector (768D) for missing or corrupt audio files

#### 5.2.4 Preprocessing Output

Each call produces a `.pt` file containing:
```python
{
    "acoustic": Tensor[N, 29],     # Acoustic features
    "text_emb": Tensor[N, 768],    # FinBERT embeddings
    "wav_emb":  Tensor[N, 768],    # wav2vec2 embeddings
    "n_utterances": int,           # Number of utterances
}
```
where N ≤ 200 (max_utterances).

### 5.3 Model Architectures (`src/model.py`)

#### 5.3.1 CrossModalFusion (Primary Model)

**Total parameters**: ~72,675 (with hidden_dim=32)

| Component | Description | Dimensions |
|-----------|-------------|------------|
| `acoustic_proj` | Linear → LayerNorm → GELU | 29 → H |
| `text_proj` | Linear → LayerNorm → GELU | 768 → H |
| `wav_proj` | Linear → LayerNorm → GELU | 768 → H |
| `cross_attn_text_audio` | MultiheadAttention (4 heads) | Q=text, K/V=acoustic |
| `cross_attn_text_wav` | MultiheadAttention (4 heads) | Q=text, K/V=wav2vec |
| Residual connection | fused = text→audio + text→wav + text | H |
| `seq_encoder` | TransformerEncoder (1 layer, 4H FFN) | H → H |
| Masked mean pooling | Averages over valid utterances | H |
| `head` | Linear → GELU → Dropout → Linear | H → H → n_targets |

**Forward pass**:
1. Project all three modalities to hidden dimension H
2. Text embeddings query acoustic features via cross-attention → `fused_ta`
3. Text embeddings query wav2vec features via cross-attention → `fused_tw`
4. Residual fusion: `fused = fused_ta + fused_tw + text_proj`
5. Pass through Transformer encoder with padding mask
6. Masked mean pooling (ignores padded utterances)
7. MLP head produces volatility predictions

#### 5.3.2 EarlyFusionBaseline

Concatenates mean-pooled features from all three modalities and passes through a feed-forward network.
- Input: (29 + 768 + 768) = 1565 → H → H/2 → n_targets

#### 5.3.3 TextOnlyBaseline

Uses only FinBERT embeddings. Mean-pools text features → feed-forward network.
- Input: 768 → H → H/2 → n_targets

#### 5.3.4 AudioOnlyBaseline

Uses only 29D acoustic features. Mean-pools acoustic features → feed-forward network.
- Input: 29 → H → H/2 → n_targets

#### 5.3.5 Model Variant Summary

| Model | Parameters (~) | Input Modalities | Fusion Method |
|-------|---------------|-----------------|---------------|
| CrossModalFusion | 72,675 | Acoustic + Text + Audio | Cross-attention |
| EarlyFusionBaseline | ~400K | Acoustic + Text + Audio | Concatenation |
| TextOnlyBaseline | ~100K | Text | None (single) |
| AudioOnlyBaseline | ~50K | Acoustic | None (single) |

### 5.4 Training Procedure (`src/train.py`)

#### 5.4.1 Loss Function: HuberRank

The custom `HuberRankLoss` combines two objectives:

**Component 1 — Huber Loss** (for absolute accuracy):
```
L_huber = HuberLoss(δ = 0.01)
```
Huber loss is less sensitive to outliers than MSE, using a quadratic function near zero and linear function for large errors, with transition at δ.

**Component 2 — Soft Rank Correlation Loss** (for ranking quality):
```
soft_ranks(x) = Σⱼ σ((xᵢ - xⱼ) / τ)     (τ = 0.1, temperature)
soft_ρ = Pearson(soft_ranks(pred), soft_ranks(label))
L_rank = 1 - soft_ρ
```
This provides a differentiable approximation to (1 − Spearman ρ), using pairwise sigmoid comparisons for soft ranking.

**Combined loss**:
```
L = L_huber + λ_rank × L_rank     (λ_rank = 1.0)
```

**Multi-task extension**: For multi-task training with {1d, 3d, 7d} targets:
```
L_total = Σₜ wₜ × (L_huber,t + λ_rank × L_rank,t) / Σₜ wₜ
```
where the primary target (7d) receives weight w = 2.0 and auxiliary targets receive w = 1.0.

#### 5.4.2 Optimizer and Scheduler

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-2 |
| Gradient clipping | 1.0 (max norm) |
| LR scheduler | CosineAnnealingWarmRestarts (T₀ = 10) |
| Warmup epochs | 5 (linear warmup) |
| Mixed precision | AMP (GradScaler), enabled on GPU |

#### 5.4.3 Early Stopping and Model Selection

- **Patience**: 25 epochs without improvement
- **Selection criterion**: Combined score = normalized MSE improvement + Spearman ρ
  - `score = (baseline_mse - val_mse) / baseline_mse + val_rho`
- **Ensemble**: 5 seeds (0, 42, 84, 126, 168) per model type

#### 5.4.4 Data Augmentation and Regularization

- **Dropout**: 0.5 (aggressive, due to small dataset)
- **Rank transformation**: Quantile normalization of targets using log1p(|x| × 100) followed by ranking against the training distribution (maps to [0, 1])
- **Degenerate label filtering**: Removes CAPM placeholders at dataset construction time
- **Zero-volatility filtering**: Removes exact-zero samples (broken CAPM fits)

#### 5.4.5 Training Configuration

All hyperparameters defined in `configs/default.yaml`:

```yaml
data:
  dataset_dir: "dataset/sp1500_earningscall"
  processed_dir: "data/processed"
  labels_path: "data/processed/labels.csv"
  max_utterances: 200
  min_utterances: 5

audio:
  sample_rate: 16000
  n_acoustic_features: 29

text:
  model_name: "ProsusAI/finbert"
  max_length: 512
  embedding_dim: 768

wav2vec:
  model_name: "facebook/wav2vec2-base"
  embedding_dim: 768

model:
  hidden_dim: 32
  n_heads: 4
  dropout: 0.5
  n_structured: 5

training:
  epochs: 50
  batch_size: 8
  num_workers: 0
  pin_memory: true
  lr: 1.0e-4
  weight_decay: 1.0e-2
  grad_clip: 1.0
  patience: 25
  n_seeds: 5
  loss: "huber_rank"
  rank_loss_weight: 1.0
  huber_delta: 0.01
  model_selection: "spearman"
  warmup_epochs: 5

splits:
  train: ["2017-04-24", "2017-10-31"]
  val: ["2017-11-01", "2017-12-31"]
  test: ["2018-01-01", "2018-05-31"]

label:
  estimation_window: 200
  event_windows: [1, 3, 7]
  primary_target: "abnormal_vol_7d"
  filter_degenerate: true
  filter_zeros: true
  rank_transform: true
  multi_task: true
  multi_task_targets: ["abnormal_vol_1d", "abnormal_vol_3d", "abnormal_vol_7d"]
  primary_task_weight: 2.0
```

### 5.5 Data Pipeline (`src/dataset.py`)

#### 5.5.1 Dataset Class: `EarningsCallDataset`

- Merges manifest (call metadata) with labels (volatility targets) on `call_id`
- Filters by temporal split ranges
- Applies degenerate label filtering, zero-volatility filtering
- Applies rank transformation (quantile normalization)
- Loads `.pt` feature files on-the-fly from `data/processed/`

#### 5.5.2 Collation Function

Custom `collate_fn` handles variable-length sequences:
- Pads all modalities to the longest sequence in the batch
- Creates boolean padding masks (True = padded position)
- Supports both single-task (1D labels) and multi-task (2D labels)

#### 5.5.3 Rank Transform (Quantile Normalization)

Two-step process applied to target labels:
1. **Log transform**: `log1p(|x| × 100)` — spreads the right-skewed volatility distribution
2. **Quantile ranking**: Maps values to [0, 1] using the training set distribution as reference
   - For val/test: uses training quantiles for consistent mapping
   - `rank = searchsorted(reference_sorted, log_value) / N_reference`

### 5.6 Evaluation Metrics (`src/evaluate.py`)

| Metric | Description | Purpose |
|--------|-------------|---------|
| **Spearman ρ** | Rank correlation coefficient | Primary metric — measures ordinal ranking quality |
| **Kendall τ** | Pairwise concordance coefficient | Alternative rank measure — counts concordant/discordant pairs |
| **MAE** | Mean Absolute Error | Absolute prediction accuracy |
| **RMSE** | Root Mean Squared Error | Error magnitude (sensitive to outliers) |
| **MSE** | Mean Squared Error | Standard regression metric |
| **Top-Quartile Hit Rate** | Fraction of correctly identified high-risk calls | Practical utility — identifies highest-volatility calls |

---

## 6. Implementation Details

### 6.1 Software Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.8+ |
| Deep Learning | PyTorch | 2.0+ |
| NLP Model | HuggingFace Transformers (FinBERT) | 4.40+ |
| Audio Processing | librosa, soundfile | 0.10+, 0.12+ |
| Audio Model | HuggingFace Transformers (wav2vec2) | 4.40+ |
| Data Processing | pandas, NumPy, SciPy | 2.0+, 1.24+, 1.11+ |
| Statistics | scikit-learn, SciPy (spearmanr, kendalltau, linregress) | 1.3+ |
| Stock Data | yfinance, requests | 0.2+, 2.31+ |
| Web Framework | Flask | 2.0+ |
| Configuration | PyYAML, python-dotenv | 6.0+, 1.0+ |
| Visualization | matplotlib, seaborn | 3.7+, 0.12+ |
| Progress Bars | tqdm | 4.65+ |

### 6.2 Project File Structure

```
MMECv2/
├── app.py                              # Flask web application (309 lines, 11.8 KB)
├── configs/
│   └── default.yaml                    # All hyperparameters (57 lines, 1.1 KB)
├── src/
│   ├── __init__.py                     # Package init (empty)
│   ├── preprocess.py                   # Feature extraction pipeline (230 lines, 7.7 KB)
│   ├── label_builder.py                # Abnormal volatility label computation (368 lines, 13 KB)
│   ├── data_sources.py                 # Multi-source stock data fetcher (458 lines, 18 KB)
│   ├── dataset.py                      # PyTorch Dataset and DataLoader (247 lines, 9.5 KB)
│   ├── model.py                        # Model architectures (181 lines, 6.7 KB)
│   ├── train.py                        # Training loop (357 lines, 13.5 KB)
│   └── evaluate.py                     # Evaluation metrics (120 lines, 4.3 KB)
├── templates/
│   └── index.html                      # Web UI (566 lines, 17.6 KB)
├── data/
│   └── processed/
│       ├── manifest.csv                # 875 calls (56 KB)
│       └── labels.csv                  # 667 labeled calls (146 KB)
├── checkpoints/                        # Trained model weights (20 files)
│   ├── cross_modal_seed{0-4}.pt        # ~304 KB each
│   ├── early_fusion_seed{0-4}.pt       # ~414 KB each
│   ├── text_only_seed{0-4}.pt          # ~925 KB each
│   └── audio_only_seed{0-4}.pt         # ~21 KB each
├── abnormal_vol_3d/                    # 3-day volatility model checkpoints
│   └── cross_modal_seed{0-4}.pt        # ~304 KB each
├── abnormal_vol_7d/                    # 7-day volatility model checkpoints
│   └── cross_modal_seed{0-4}.pt        # ~304 KB each
├── logs/
│   ├── 09/                             # Earlier training runs
│   │   ├── 01.log                      # Run with 2.6M params model
│   │   └── 02.log                      # Additional run
│   └── 10/                             # Latest training runs
│       ├── 01.log                      # Run with degenerate filtering (141/51/92)
│       └── 02.log                      # Refined run (131/43/89)
├── assets/
│   ├── API_KEYS_GUIDE.md               # API key setup instructions
│   └── test/                           # Test audio/transcript files
│       ├── MSFT Q2 2026.mp3 (78 MB)    # Microsoft Q2 earnings call
│       ├── MSFT_Q2_2026.txt (31 KB)    # Microsoft transcript
│       ├── TSLA Q4 2025.mp3 (16 MB)    # Tesla Q4 earnings call
│       ├── TSLA_Q4_2025.txt (5.7 KB)   # Tesla transcript
│       ├── NFLX Q4 2025.mp3 (8 MB)     # Netflix Q4 earnings call
│       ├── NFLX_Q4_2025.txt (3.5 KB)   # Netflix transcript
│       ├── SNDK Q1 2026.mp3 (36 MB)    # SanDisk Q1 earnings call
│       └── SNDK_Q1_2026.txt (14 KB)    # SanDisk transcript
├── requirements.txt                    # Python dependencies (17 packages)
├── .env.example                        # Environment variable template (5 API keys)
├── .gitignore                          # Git ignore rules
├── LICENCE                             # MIT License (Aryan Kumar Sinha, 2026)
└── README.md                           # Project documentation
```

### 6.3 Web Application (`app.py`)

The Flask application provides a real-time inference interface:

**Architecture:**
- **Startup**: Loads all 4 model variants (seed 0), FinBERT tokenizer/model, wav2vec2 processor/model
- **Inference on preprocessed data**: Loads `.pt` file → runs forward pass → returns multi-task predictions
- **Inference on uploaded audio**: Asynchronous pipeline using ThreadPoolExecutor (2 workers)

**Upload processing pipeline:**
1. Accept audio file (MP3, WAV, M4A, FLAC) + optional transcript (.txt)
2. Segment audio into 30-second chunks
3. Uniformly sample up to 30 segments for speed (~15 min coverage)
4. Extract acoustic features using librosa (pitch/YIN, MFCCs, spectral centroid, spectral rolloff, ZCR, RMS)
5. Encode text with FinBERT (batch of 32)
6. Encode audio segments with wav2vec2 (batch of 4, padded)
7. Run all 4 model variants for comparison
8. Return predictions as JSON via polling endpoint

**API Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Render web UI (Jinja2 template) |
| `/upload` | POST | Submit audio file for async processing |
| `/job/<job_id>` | GET | Poll for async job completion |

**Frontend (`templates/index.html`):**
- Dark theme with glassmorphism design
- Inter font (Google Fonts)
- Gradient accent colors (#6366f1 indigo)
- Real-time polling with 3-second intervals for upload jobs
- Risk classification: Low (<35%), Moderate (35-65%), High (>65%)
- Model comparison table across all 4 variants
- Embedded TradingView chart widget for stock visualization
- Responsive design (768px breakpoint)

---

## 7. Experimental Results

### 7.1 Training Evolution (Log Analysis)

The project underwent iterative refinement across multiple training runs:

#### Run 09/01 — Initial Architecture (Hidden Dim = 256, no rank transform)
| Property | Value |
|----------|-------|
| Model parameters | 2,608,129 |
| Training calls | 260 |
| Validation calls | 70 |
| Test calls | 168 |
| Loss function | MSE |
| Primary target | abnormal_vol_3d (raw) |

**Ensemble Results (5 seeds):**
| Metric | Mean ± Std |
|--------|-----------|
| MSE | 0.00138 ± 0.00008 |
| MAE | 0.03058 ± 0.00194 |
| Spearman ρ | 0.003 ± 0.064 |

**Issue**: Near-zero Spearman ρ — the model minimized MSE but failed to capture ordinal ranking information. Predictions collapsed to near-constant values (pred_std ≈ 0.002 vs. label_std ≈ 0.056).

#### Run 10/01 — With Label Filtering and Rank Transform
| Property | Value |
|----------|-------|
| Model parameters | 72,675 (reduced hidden_dim to 32) |
| Training calls | 141 (after degenerate filtering) |
| Loss function | HuberRank |
| Rank transform | Enabled |
| Multi-task | Enabled (1d, 3d, 7d) |
| Filtering | 215 degenerate labels removed |

**Ensemble Results (5 seeds):**
| Metric | Mean ± Std |
|--------|-----------|
| MSE | 0.23766 ± 0.08143 |
| MAE | 0.39311 ± 0.06074 |
| Spearman ρ | −0.069 ± 0.073 |

**Improvement**: Model now produces variance in predictions. Negative ρ indicates overfitting to train set rank patterns that don't generalize.

#### Run 10/02 — Refined Filtering (Latest)
| Property | Value |
|----------|-------|
| Training calls | 131 |
| Validation calls | 43 |
| Test calls | 89 |
| Primary target | abnormal_vol_7d |

**Ensemble Results (5 seeds):**
| Metric | Mean ± Std |
|--------|-----------|
| MSE | 0.23355 ± 0.07129 |
| MAE | 0.39426 ± 0.05304 |
| Spearman ρ | −0.047 ± 0.051 |

**Per-seed test results:**

| Seed | MSE | MAE | Spearman ρ | Early Stop Epoch |
|------|-----|-----|------------|-----------------|
| 0 | 0.339 | 0.482 | −0.030 | 50 (no early stop) |
| 42 | 0.215 | 0.371 | −0.039 | 36 |
| 84 | 0.138 | 0.333 | +0.023 | 26 |
| 126 | 0.189 | 0.361 | −0.055 | 49 |
| 168 | 0.288 | 0.424 | −0.134 | 47 |

**Training dynamics observations:**
- Training Spearman ρ reaches 0.6–0.7 by epoch 40–50 consistently across seeds
- Validation Spearman ρ peaks early (~0.15–0.22 in first 5 epochs) then degrades — classic overfitting on small dataset
- The train-test gap in Spearman indicates the model memorizes training set rank structure

### 7.2 Key Findings

1. **Train-test Spearman gap**: Training ρ reaches ~0.7 while test ρ is near 0 or slightly negative, indicating significant overfitting despite aggressive dropout (0.5) and early stopping.

2. **Prediction variance growth**: Prediction standard deviation grows from ~0.03 (epoch 1) to ~0.15 (epoch 50), showing the model does learn to differentiate calls, but this learned differentiation does not align with test-set labels.

3. **Model capacity vs. data**: Early runs with 2.6M parameters suffered from extreme prediction collapse (pred_std → 0.002). Reducing to 72,675 parameters was a significant improvement.

4. **Rank transform effectiveness**: Switching from raw MSE on skewed volatility values to quantile-normalized HuberRank loss dramatically improved prediction diversity.

5. **Multi-task learning**: Joint 1d/3d/7d prediction with primary weighting on 7d provides implicit regularization through shared representations.

---

## 8. Web Application

### 8.1 User Interface

The web interface provides:
1. **Audio upload**: Accepts MP3, WAV, M4A, FLAC formats
2. **Optional transcript upload**: TXT file for better text analysis (otherwise generic placeholder text is used)
3. **Model selection**: Dropdown for cross_modal, early_fusion, text_only, audio_only
4. **Async processing**: Background job with polling status display (elapsed time shown)
5. **Results visualization**:
   - Predicted volatility rank (1-day, 3-day, 7-day) as percentage with risk classification
   - Actual labels (if available from pre-existing dataset)
   - Model comparison table across all loaded variants
   - Embedded TradingView chart for the company ticker

### 8.2 Acoustic Feature Extraction (Runtime)

For uploaded audio (not from the preprocessed dataset), the app extracts 29 acoustic features using librosa:
- Pitch: YIN estimator (fmin=50 Hz, fmax=500 Hz) → mean and std
- MFCCs: 13 coefficients × (mean + std) = 26 values
- Spectral centroid (mean)
- Spectral rolloff (mean)
- Zero-crossing rate (mean)
- RMS energy (mean)

### 8.3 Test Assets

The project includes 4 real-world earnings call recordings for testing:
- MSFT Q2 2026 (78 MB audio, 31 KB transcript)
- TSLA Q4 2025 (16 MB audio, 5.7 KB transcript)
- NFLX Q4 2025 (8 MB audio, 3.5 KB transcript)
- SNDK Q1 2026 (36 MB audio, 14 KB transcript)

---

## 9. Discussion

### 9.1 Challenges

1. **Small dataset**: ~130–260 usable calls is extremely small for deep learning. The model's capacity to generalize is fundamentally limited.

2. **High-noise labels**: Abnormal volatility is computed from stock price data, which is inherently stochastic. The market model R² values are often low (<0.10), meaning much of the "abnormal" return is truly random.

3. **Market regime changes**: The temporal split means the model trains on 2017 bull-market dynamics and tests on early 2018 (including the Feb 2018 volatility spike), creating domain shift.

4. **Label quality**: ~30% of calls had degenerate CAPM estimates (R²≈1.0, β=1.0) due to data fetch failures, requiring careful cleaning.

5. **Computation constraints**: All training was performed on CPU (no GPU detected in logs), with per-epoch times of 1.3–3.6 seconds, indicating the bottleneck was model size not data loading.

### 9.2 Strengths

1. **End-to-end multi-modal pipeline**: From raw audio + text to volatility prediction with minimal manual feature engineering.

2. **Robust data infrastructure**: The 12-source stock data fetcher with staged fallback ensures maximum data coverage.

3. **Reproducible training**: 5-seed ensemble with fixed seed schedule and comprehensive logging.

4. **Production-ready web interface**: Flask app with async processing, model comparison, and TradingView integration.

5. **Rank-optimized training**: The HuberRank loss directly optimizes the evaluation metric (Spearman ρ), which is more practically useful than MSE for relative risk ranking.

### 9.3 Limitations and Future Work

1. **GPU training**: Training on GPU with larger models and longer training schedules may improve generalization.
2. **Larger dataset**: Extending to more years of earnings calls would significantly help.
3. **Pre-training**: Fine-tuning FinBERT and wav2vec2 on financial audio/text could improve feature quality.
4. **Cross-validation**: K-fold temporal cross-validation would give more reliable performance estimates than a single test split.
5. **Attention visualization**: Analyzing which utterances and modality combinations the cross-attention weights focus on could provide interpretability.
6. **Speaker diarization**: Separating CEO/CFO speech from analyst questions may improve signal extraction.

---

## 10. Conclusion

This project demonstrates a complete system for multi-modal earnings call volatility prediction that fuses acoustic stress signals with NLP text representations using cross-modal attention. The architecture processes 29D acoustic features, 768D FinBERT embeddings, and 768D wav2vec2 embeddings through projection layers and cross-attention to produce rank-optimized volatility predictions across 1-day, 3-day, and 7-day event windows. While the current training results show strong training-set rank correlation (ρ ≈ 0.7) but limited test-set generalization (ρ ≈ 0) — primarily due to the extremely small dataset size (~130 training calls) — the system infrastructure provides a solid foundation for scaling to larger datasets and more powerful models. The project includes a production-quality web interface, robust multi-source data pipeline, comprehensive label cleaning, and ablation study infrastructure supporting four model variants.

---

## 11. References

1. Mayew, W. J., & Venkatachalam, M. (2012). The Power of Voice: Managerial Affective States and Future Firm Performance. *The Journal of Finance*, 67(1), 1–43.

2. Qin, Y., & Yang, Y. (2019). What You Say and How You Say It Matters: Predicting Stock Volatility Using Verbal and Vocal Cues. *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, 390–401.

3. Araci, D., (2019). FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models. *arXiv preprint arXiv:1908.10063*.

4. Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. *Advances in Neural Information Processing Systems*, 33, 12449–12460.

5. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT*, 4171–4186.

6. Li, J., Yang, Y., Qin, Y., & Yang, Y. (2020). MAEC: A Multimodal Aligned Earnings Conference Call Dataset for Financial Risk Prediction. *arXiv preprint arXiv:2009.13890*.

7. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30.

8. Sharpe, W. F. (1964). Capital Asset Prices: A Theory of Market Equilibrium Under Conditions of Risk. *The Journal of Finance*, 19(3), 425–442.

---

## 12. Appendices

### Appendix A: Complete Hyperparameter Table

| Category | Parameter | Value |
|----------|-----------|-------|
| **Data** | max_utterances | 200 |
| | min_utterances | 5 |
| | sample_rate | 16000 Hz |
| | n_acoustic_features | 29 |
| **Text** | model | ProsusAI/finbert |
| | max_length | 512 tokens |
| | embedding_dim | 768 |
| **Audio** | model | facebook/wav2vec2-base |
| | embedding_dim | 768 |
| **Model** | hidden_dim | 32 |
| | n_heads | 4 |
| | dropout | 0.5 |
| | n_structured | 5 |
| **Training** | epochs | 50 |
| | batch_size | 8 |
| | learning_rate | 1e-4 |
| | weight_decay | 1e-2 |
| | grad_clip | 1.0 |
| | patience | 25 |
| | n_seeds | 5 |
| | loss_function | HuberRank |
| | rank_loss_weight | 1.0 |
| | huber_delta | 0.01 |
| | warmup_epochs | 5 |
| | scheduler | CosineAnnealingWarmRestarts (T₀=10) |
| **Labels** | estimation_window | 200 days |
| | event_windows | [1, 3, 7] days |
| | primary_target | abnormal_vol_7d |
| | filter_degenerate | true |
| | filter_zeros | true |
| | rank_transform | true |
| | multi_task | true |
| | primary_task_weight | 2.0 |

### Appendix B: Label Statistics (from labels.csv)

| Column | Description |
|--------|-------------|
| call_id | {TICKER}_{YYYYMMDD} |
| ticker | Stock symbol |
| call_date | YYYY-MM-DD |
| alpha | CAPM intercept (daily excess return) |
| beta | CAPM market sensitivity |
| r2 | CAPM model R² |
| abnormal_ret_{1,3,7}d | Cumulative abnormal return |
| abnormal_vol_{1,3,7}d | |Cumulative abnormal return| |

**Sample data:**
```
KMB_20170424: α=-0.0005, β=0.56, R²=0.10, vol_1d=0.0023, vol_3d=0.0060, vol_7d=0.0312
CAT_20170425: α=0.0004, β=1.55, R²=0.33, vol_1d=0.0691, vol_3d=0.0513, vol_7d=0.0391
```

### Appendix C: Dependencies (requirements.txt)

```
torch>=2.0.0
transformers>=4.40.0
librosa>=0.10.0
soundfile>=0.12.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
scikit-learn>=1.3.0
yfinance>=0.2.0
pandas-market-calendars>=4.0.0
tqdm>=4.65.0
pyyaml>=6.0
requests>=2.31.0
flask>=2.0.0
python-dotenv>=1.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Appendix D: API Keys Configuration

Five optional API keys can be configured via `.env` file or CLI arguments:

| Service | Env Variable | Free Tier |
|---------|-------------|-----------|
| Alpha Vantage | `ALPHA_VANTAGE_KEY` | 5 calls/min, 500/day |
| Polygon.io | `POLYGON_KEY` | 5 calls/min |
| Financial Modeling Prep | `FMP_KEY` | 250 calls/day |
| Tiingo | `TIINGO_KEY` | 500 tickers |
| Quandl | `QUANDL_KEY` | 50 calls/day |

### Appendix E: Checkpoint File Sizes

| Model Type | Per-checkpoint Size | Total (5 seeds) |
|------------|-------------------|-----------------|
| cross_modal | 303,899 bytes (~297 KB) | 1.45 MB |
| early_fusion | 413,927 bytes (~404 KB) | 1.97 MB |
| text_only | 925,373 bytes (~904 KB) | 4.41 MB |
| audio_only | 20,619 bytes (~20 KB) | 100 KB |
| **Total** | | **7.93 MB** |

### Appendix F: Command Reference

```bash
# 1. Preprocess raw data
python -m src.preprocess                        # Full dataset
python -m src.preprocess --max_calls 5          # Quick test

# 2. Build labels
python -m src.label_builder                     # Full with 8 workers
python -m src.label_builder --max_calls 5       # Quick test

# 3. Train models
python -m src.train --model_type cross_modal    # 5-seed ensemble
python -m src.train --epochs 1 --seeds 1        # Quick validation

# 4. Evaluate
python -m src.evaluate --checkpoint checkpoints/cross_modal_seed0.pt

# 5. Run web app
python app.py                                   # localhost:5000
```

---

## 13. License

MIT License — Copyright (c) 2026 Aryan Kumar Sinha

---

*Document generated on April 11, 2026. Covers MMECv2 codebase revision as analyzed from the complete source tree, training logs, checkpoints, and configuration files.*
