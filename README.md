# Predicting Airbnb Listing Prices with Deep Learning

> **Research question:** Can Natural Language Processing on listing descriptions
> improve Airbnb price predictions compared to tabular features alone?

## Authors

- Ethan B. _(add co-authors here)_

## Overview

This project builds and compares five progressively richer models for
predicting the **log-price** of Airbnb listings in London:

| Model                  | Features                                    | Architecture                             |
| ---------------------- | ------------------------------------------- | ---------------------------------------- |
| **Baseline**           | Tabular (numeric + categorical)             | OLS & Random Forest (scikit-learn)       |
| **Text Model**         | Text (description + neighbourhood overview) | Embedding → LSTM → Linear head (PyTorch) |
| **Hybrid**             | Tabular + Text                              | MLP + LSTM → Late fusion (PyTorch)       |
| **Transformer**        | Text (description + neighbourhood overview) | Embedding → Transformer Encoder → Linear |
| **Hybrid Transformer** | Tabular + Text                              | Transformer Encoder + MLP → Late fusion  |

Data is sourced from [Inside Airbnb](http://insideairbnb.com/).

## Project Structure

```
src/
├── config.py              # Centralised constants & hyperparameters
├── pipeline/              # Data collection, cleaning, splitting
├── features/              # Text & tabular feature engineering
├── models/                # PyTorch model architectures
├── training/              # Training scripts (CLI entry points)
└── visualization/         # Plot generation + interpretability
```

## Installation

Requires **Python ≥ 3.11**.

```bash
git clone https://github.com/ethanbch/deep-learning-airbnb.git
cd deep-learning-airbnb
```

**With [uv](https://docs.astral.sh/uv/)** (recommended):

```bash
uv sync
```

**With pip:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the Results

Run the seven steps below **in order**. All hyperparameters are configured
in `src/config.py` — the CLI commands can stay minimal.

> Replace `uv run` with `python` if you use a regular virtualenv.

### 1. Data Collection & Cleaning

Downloads the latest London listings from Inside Airbnb, cleans the data
and creates train / validation / test splits (70 / 15 / 15):

```bash
uv run python main.py
```

### 2. Tabular Baseline

Trains OLS and Random Forest on numeric + categorical features only:

```bash
uv run python src/training/train_baseline.py
```

### 3. Text Model (LSTM)

Trains a text-only LSTM on listing descriptions:

```bash
uv run python src/training/train_text_model.py --use-neighborhood-overview
```

### 4. Hybrid Model (Text + Tabular)

Trains the fusion model combining both modalities and generates
a comparison chart (`data/results/london/comparison_chart.png`):

```bash
uv run python src/training/train_hybrid_model.py --use-neighborhood-overview
```

### 5. Transformer Model (Encoder-only)

Trains a basic Transformer encoder model (attention-based) on text:

```bash
uv run python src/training/train_transformer_model.py --use-neighborhood-overview
```

### 6. Hybrid Transformer Model (Text + Tabular)

Trains the hybrid Transformer + tabular model and saves artifacts under
`data/results/london/hybrid_transformer/`:

```bash
uv run python src/training/train_hybrid_transformer.py --use-neighborhood-overview
```

### 7. Interpretability (Post-training)

Runs residual and lexical analysis without retraining, and exports insights:

```bash
uv run python src/visualization/interpretability.py
```

### 8. Poster Figures (Publication-ready)

Generates three high-resolution (300 DPI) figures for the final poster:

```bash
uv run python src/visualization/poster_plots.py --use-neighborhood-overview
```

## Results

All metrics and artifacts are saved to `data/results/london/`:

| File                                   | Content                         |
| -------------------------------------- | ------------------------------- |
| `baseline_metrics.json`                | OLS & RF test metrics           |
| `text_model/test_metrics.json`         | LSTM test metrics               |
| `hybrid/test_metrics.json`             | Hybrid test metrics             |
| `transformer_model/test_metrics.json`  | Transformer test metrics        |
| `hybrid_transformer/test_metrics.json` | Hybrid Transformer test metrics |
| `comparison_chart.png`                 | R² bar chart across all models  |

Poster figures are saved in `data/results/london/plots/`:

| File                                    | Content                                              |
| --------------------------------------- | ---------------------------------------------------- |
| `figure_1_methodological_benchmark.png` | Benchmark bar chart (all model families)             |
| `figure_2_best_model_fit.png`           | Scatter plot: real vs predicted (Hybrid Transformer) |
| `figure_3_residual_robustness.png`      | Residual histogram + KDE robustness analysis         |

Interpretability outputs are saved in `data/results/london/interpretability/`:

| File                                      | Content                                                                 |
| ----------------------------------------- | ----------------------------------------------------------------------- |
| `test_residuals_all_models.csv`           | Residuals and errors for all available deep models                      |
| `test_residuals.csv`                      | Compatibility alias of the multi-model residual export                  |
| `model_metrics_summary.csv`               | Per-model RMSE/MAE/R² summary on the test set                           |
| `top_best_model_vs_baseline_listings.csv` | Listings where the best deep model has strongest gain proxy vs baseline |
| `top_words_high_price.csv`                | Top lexical markers associated with expensive listings                  |
| `top_words_low_price.csv`                 | Top lexical markers associated with cheaper listings                    |
| `word_price_signals.png`                  | Bar chart of high-price vs low-price lexical signals                    |
| `interpretability_summary.json`           | Global summary (models evaluated, best model, lexical outputs)          |
