# Multi-Modal Earnings Call Volatility Prediction

Predicts post-earnings stock volatility by fusing **acoustic stress signals** (pitch, jitter, shimmer, HNR) with **NLP text embeddings** (FinBERT) and **wav2vec2 audio embeddings** via cross-modal attention.

**Dataset:** MAEC (S&P 1500 earnings calls with pre-segmented utterance audio + Praat features)

## Setup

```bash
pip install -r requirements.txt
```

## Run Pipeline

### 1. Preprocess (GPU: encodes text with FinBERT + audio with wav2vec2)

```bash
# Smoke test (5 calls)
python -m src.preprocess --max_calls 5

# Full run
python -m src.preprocess
```

Creates per-call `.pt` files in `data/processed/` + `manifest.csv`.

### 2. Build Labels (fetches stock returns from yfinance)

```bash
# Smoke test
python -m src.label_builder --max_calls 5

# Full run (uses 8 threads by default)
python -m src.label_builder

```

Creates `data/processed/labels.csv` with abnormal volatility targets.


### 3. Train

```bash
# Dry run (1 epoch, 1 seed)
python -m src.train --epochs 1 --seeds 1

# Full training (50 epochs, 5-seed ensemble)
python -m src.train

# Ablation baselines
python -m src.train --model_type audio_only
python -m src.train --model_type text_only
python -m src.train --model_type early_fusion
```

Saves checkpoints to `checkpoints/`.

### 4. Evaluate

```bash
python -m src.evaluate --checkpoint checkpoints/cross_modal_seed0.pt
```

Reports: Spearman ρ, Kendall τ, MAE, RMSE, top-quartile hit rate + ablation table.

## Architecture

```
Acoustic (29D Praat) ──┐
                       ├── Cross-Modal Attention ──► Transformer ──► Mean Pool ──► MLP ──► σ̂
Text (768D FinBERT) ───┤
                       │
Audio (768D wav2vec2) ─┘
```

- **Cross-attention:** text queries attend to acoustic + wav2vec features
- **~500K params** (appropriate for ~900 effective training samples)
- **5-seed ensemble** for stable predictions

## Config

All hyperparameters in `configs/default.yaml`. Key settings:

| Parameter | Value |
|---|---|
| Learning rate | 2e-4 |
| Batch size | 16 |
| Hidden dim | 256 |
| Attention heads | 4 |
| Patience | 15 epochs |
| Primary metric | Spearman ρ |
| Target | 3-day abnormal volatility |
