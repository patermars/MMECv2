# MMEC v2 Project Progress

This document tracks the evaluation metrics across different rounds of optimization for the multi-modal earnings call volatility prediction model.

## Summary Table

| Metric | Baseline | Round 1 (Regularization) | Round 2 (PCA + MLP) |
| :--- | :--- | :--- | :--- |
| **Spearman ρ** | 0.3292 | 0.3206 | 0.0976 |
| **R²** | -66.6 | **-0.0199** | -0.0261 |
| **Directional Acc** | 46.5% | **60.56%** | 50.70% |
| **MAE** | - | 0.8126 | 0.8398 |
| **RMSE** | - | 1.7552 | 1.7605 |
| **Class. Accuracy** | ~25% | **32%** | 24% |

---

## Detailed Results

### Round 1: Regularization & Hybrid Loss
**Strategy:** Reduced hidden dim (128), added dropout (0.4), switched to SmoothL1Loss (Huber), added feature noise.

*   **Result:** **Major breakthrough in R²**. The model moved from predicting wild outliers to being centered around the mean.
*   **Analysis:** Directional accuracy hit 60%, showing the model started learning the "sign" of volatility correctly. However, it still overfitted heavily after 1 epoch.

### Round 2: Architecture Simplification (PCA + MLP)
**Strategy:** Reduced text dim via PCA (768 → 32), replaced Transformer with simple MLP, added SWA.

*   **Result:** **Significant Regression**. All metrics dropped.
*   **Analysis:** PCA likely destroyed the nuance in FinBERT embeddings, and the simple MLP lost the "Cross-Modal Attention" benefit where text features attend to audio features.

---

## Round 3 Roadmap (Current)

The user has approved a strategy to address the **324 training sample** bottleneck:

1.  **Revert to Round 1 Architecture:** Keep the Transformer + Cross-Attention.
2.  **Multi-Window Dataset:** Use 1-day, 3-day, and 7-day targets from the same calls to effectively **3× the training data** (~938 samples).
3.  **Rank-Based Normalization:** Use Quantile mapping to handle the extreme 4.77 label skewness.

---

## Guidance: Building Your Own Multi-Modal Dataset

Since you are limited by the small size of ACL19, here is how you can expand it ourselves:

### 1. Data Sources
*   **Audio:** Use `yfinance` or `EarningsCast` to find Tickers. Search YouTube/Company IR pages for high-quality audio recordings of recent calls (2020-2024).
*   **Transcripts:** Use `whisper-large-v3` to transcribe raw audio if you can't find official transcripts. Official ones are available on SeekingAlpha/TheMotleyFool.
*   **Labels:** Continue using `yfinance` for abnormal volatility calculation (already automated in your `label_builder.py`).

### 2. Augmentation Strategy
*   **Temporal Slicing:** Instead of one sample per call, treat the **Management Discussion** and **Q&A** sections as separate but related training samples.
*   **Cross-Sample Mixing (Mixup):** Mix audio/text features of two similar calls to create a synthetic third call.
*   **Back-translation:** For text, translate to another language and back to increase variety.

> [!TIP]
> Prioritize **Quantity** over **Quality** at this stage. 5,000 "noisy" calls will likely train a better Transformer than 500 perfect ones.
