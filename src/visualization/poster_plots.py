"""Generate publication-quality poster figures for model benchmarking and robustness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import (  # noqa: E402
    DEFAULT_CITY,
    MAX_SEQUENCE_LENGTH,
    RESULTS_DIR,
    TARGET_COLUMN,
)
from features.tabular import (  # noqa: E402
    HybridDataset,
    collate_hybrid_batch,
    preprocess_tabular_features,
)
from features.text import Vocabulary  # noqa: E402
from models.hybrid_transformer_predictor import HybridTransformerPredictor  # noqa: E402
from training.utils import (  # noqa: E402
    combine_text_columns,
    detect_device,
    evaluate_metrics,
    load_splits,
)


def _configure_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )


def _load_vocab(vocab_path: Path) -> Vocabulary:
    with vocab_path.open("r", encoding="utf-8") as input_file:
        return Vocabulary.from_dict(json.load(input_file))


def _load_hybrid_transformer_model(
    city: str, device: torch.device
) -> HybridTransformerPredictor:
    checkpoint_path = RESULTS_DIR / city / "hybrid_transformer" / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}. Run train_hybrid_transformer.py first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = dict(checkpoint["model_config"])
    if "dropout" not in model_config and "transformer_dropout" in model_config:
        model_config["dropout"] = model_config.pop("transformer_dropout")

    model = HybridTransformerPredictor(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _predict_hybrid_transformer(
    model: HybridTransformerPredictor,
    test_texts: list[str],
    x_test: np.ndarray,
    y_test: np.ndarray,
    vocabulary: Vocabulary,
    max_sequence_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = HybridDataset(
        texts=test_texts,
        tabular_features=x_test,
        targets=y_test,
        vocabulary=vocabulary,
        max_sequence_length=max_sequence_length,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_hybrid_batch(
            batch, pad_index=vocabulary.pad_index
        ),
    )

    predictions: list[float] = []
    with torch.no_grad():
        for input_ids, lengths, tabular_features, _ in data_loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            tabular_features = tabular_features.to(device)

            outputs = model(
                input_ids=input_ids,
                lengths=lengths,
                tabular_features=tabular_features,
            )
            predictions.extend(outputs.detach().cpu().tolist())

    return np.asarray(predictions, dtype=np.float32)


def _collect_benchmark_rows(city: str) -> pd.DataFrame:
    rows: list[dict[str, str | float]] = []

    baseline_path = RESULTS_DIR / city / "baseline_metrics.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        models = baseline.get("models", {})
        if "linear_regression_ols" in models:
            rows.append(
                {
                    "model": "OLS",
                    "family": "baseline",
                    "r2": float(models["linear_regression_ols"]["r2"]),
                }
            )
        if "random_forest" in models:
            rows.append(
                {
                    "model": "Random Forest",
                    "family": "baseline",
                    "r2": float(models["random_forest"]["r2"]),
                }
            )

    metric_specs = [
        ("text_model/test_metrics.json", "LSTM", "nlp"),
        ("transformer_model/test_metrics.json", "Transformer", "nlp"),
        ("hybrid/test_metrics.json", "Hybrid", "hybrid"),
        ("hybrid_transformer/test_metrics.json", "Hybrid Transformer", "hybrid"),
    ]
    for rel_path, label, family in metric_specs:
        path = RESULTS_DIR / city / rel_path
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": label,
                "family": family,
                "r2": float(data["metrics"]["r2"]),
            }
        )

    if not rows:
        raise FileNotFoundError(
            "No metrics files found for benchmark figure. Run training scripts first."
        )

    benchmark_df = pd.DataFrame(rows)
    order = [
        "OLS",
        "Random Forest",
        "LSTM",
        "Transformer",
        "Hybrid",
        "Hybrid Transformer",
    ]
    benchmark_df["model"] = pd.Categorical(
        benchmark_df["model"], categories=order, ordered=True
    )
    return benchmark_df.sort_values("model")


def _plot_benchmark(benchmark_df: pd.DataFrame, output_path: Path) -> None:
    color_map = {
        "OLS": "#9CA3AF",
        "Random Forest": "#6B7280",
        "LSTM": "#3B82F6",
        "Transformer": "#1D4ED8",
        "Hybrid": "#F97316",
        "Hybrid Transformer": "#DC2626",
    }

    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = [color_map[str(model)] for model in benchmark_df["model"]]
    bars = ax.bar(benchmark_df["model"].astype(str), benchmark_df["r2"], color=colors)

    ax.set_title("Figure 1 — Methodological Benchmark")
    ax.set_ylabel("R² Score (Test Set)")
    ax.set_xlabel("Model")
    ax.set_ylim(0, max(benchmark_df["r2"]) + 0.08)
    ax.tick_params(axis="x", rotation=15)

    for bar, value in zip(bars, benchmark_df["r2"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(value) + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    metrics = evaluate_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    sns.scatterplot(
        x=y_true,
        y=y_pred,
        alpha=0.5,
        s=28,
        edgecolor=None,
        color="#DC2626",
        ax=ax,
    )

    min_val = float(min(np.min(y_true), np.min(y_pred)))
    max_val = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color="black",
        linewidth=1.6,
    )

    ax.set_title("Figure 2 — Hybrid Transformer Fit (Test Set)")
    ax.set_xlabel("Real Price (Log)")
    ax.set_ylabel("Predicted Price (Log)")

    summary_text = f"RMSE: {metrics['rmse']:.3f}\nR²: {metrics['r2']:.3f}"
    ax.text(
        0.04,
        0.96,
        summary_text,
        transform=ax.transAxes,
        va="top",
        fontsize=13,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_residual_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    residuals = y_true - y_pred
    mean_error = float(np.mean(residuals))

    fig, ax = plt.subplots(figsize=(8.8, 6.5))
    sns.histplot(
        residuals,
        bins=50,
        kde=True,
        color="#2563EB",
        alpha=0.7,
        ax=ax,
    )
    ax.axvline(0.0, color="red", linestyle="-", linewidth=1.8)

    ax.set_title("Figure 3 — Residual Robustness Analysis")
    ax.set_xlabel("Residual (Real - Predicted)")
    ax.set_ylabel("Frequency")

    ax.text(
        0.04,
        0.94,
        f"Mean Error: {mean_error:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=13,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_poster_plots(
    city: str,
    use_neighborhood_overview: bool,
    max_sequence_length: int,
    batch_size: int,
    device_preference: str,
) -> None:
    _configure_style()

    output_dir = RESULTS_DIR / city / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_df = _collect_benchmark_rows(city)
    _plot_benchmark(benchmark_df, output_dir / "figure_1_methodological_benchmark.png")

    train_df, val_df, test_df = load_splits(city)
    _ = train_df, val_df

    x_train, _, x_test, _, _ = preprocess_tabular_features(train_df, val_df, test_df)
    _ = x_train
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    test_texts = combine_text_columns(test_df, use_neighborhood_overview).tolist()

    vocab_path = RESULTS_DIR / city / "hybrid_transformer" / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Missing vocabulary: {vocab_path}. Run train_hybrid_transformer.py first."
        )

    vocabulary = _load_vocab(vocab_path)
    device = detect_device(device_preference)
    model = _load_hybrid_transformer_model(city, device)

    y_pred = _predict_hybrid_transformer(
        model=model,
        test_texts=test_texts,
        x_test=x_test,
        y_test=y_test,
        vocabulary=vocabulary,
        max_sequence_length=max_sequence_length,
        batch_size=batch_size,
        device=device,
    )

    _plot_actual_vs_predicted(
        y_test, y_pred, output_dir / "figure_2_best_model_fit.png"
    )
    _plot_residual_distribution(
        y_test, y_pred, output_dir / "figure_3_residual_robustness.png"
    )

    print("\n=== Poster plots generated ===")
    print(f"Output directory: {output_dir}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality poster figures for Airbnb DL project."
    )
    parser.add_argument("--city", default=DEFAULT_CITY)
    parser.add_argument(
        "--use-neighborhood-overview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use description + neighborhood_overview as text input.",
    )
    parser.add_argument("--max-sequence-length", type=int, default=MAX_SEQUENCE_LENGTH)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    generate_poster_plots(
        city=args.city,
        use_neighborhood_overview=args.use_neighborhood_overview,
        max_sequence_length=args.max_sequence_length,
        batch_size=args.batch_size,
        device_preference=args.device,
    )
