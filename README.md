# Predicting Airbnb Listing Prices with Deep Learning

> **Research question:** Can Natural Language Processing on listing descriptions
> improve Airbnb price predictions compared to tabular features alone?

## Authors

- Ethan B. _(add co-authors here)_

## Overview

This project builds and compares three progressively richer models for
predicting the **log-price** of Airbnb listings in London:

| Model          | Features                                    | Architecture                             |
| -------------- | ------------------------------------------- | ---------------------------------------- |
| **Baseline**   | Tabular (numeric + categorical)             | OLS & Random Forest (scikit-learn)       |
| **Text Model** | Text (description + neighbourhood overview) | Embedding → LSTM → Linear head (PyTorch) |
| **Hybrid**     | Tabular + Text                              | MLP + LSTM → Late fusion (PyTorch)       |

Data is sourced from [Inside Airbnb](http://insideairbnb.com/).

## Project Structure

```
src/
├── config.py              # Centralised constants & hyperparameters
├── pipeline/              # Data collection, cleaning, splitting
├── features/              # Text & tabular feature engineering
├── models/                # PyTorch model architectures
├── training/              # Training scripts (CLI entry points)
└── visualization/         # Plot generation
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

Run the four steps below **in order**. All hyperparameters are configured
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

## Results

All metrics and artifacts are saved to `data/results/london/`:

| File                           | Content                        |
| ------------------------------ | ------------------------------ |
| `baseline_metrics.json`        | OLS & RF test metrics          |
| `text_model/test_metrics.json` | LSTM test metrics              |
| `hybrid/test_metrics.json`     | Hybrid test metrics            |
| `comparison_chart.png`         | R² bar chart across all models |

## License

This project is for academic purposes only.
