"""Post-training interpretability analysis for lexical-economic insights.

This script does not retrain deep models. It loads saved artifacts to:
1) Compute residuals on test set for all available deep models.
2) Compare model errors (MAE / RMSE) and identify the best model.
3) Identify listings where the best model is strongest vs baseline proxy.
4) Extract high-value vs low-value lexical signals from train-set quartiles.
5) Generate a bar chart of discriminative words.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from torch.utils.data import DataLoader

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import (  # noqa: E402
    DEFAULT_CITY,
    DESCRIPTION_COLUMN,
    MAX_SEQUENCE_LENGTH,
    RESULTS_DIR,
    TARGET_COLUMN,
)
from features.tabular import (  # noqa: E402
    HybridDataset,
    collate_hybrid_batch,
    preprocess_tabular_features,
)
from features.text import (  # noqa: E402
    TextRegressionDataset,
    Vocabulary,
    collate_text_batch,
    simple_tokenize,
)
from models.hybrid_predictor import HybridPredictor  # noqa: E402
from models.hybrid_transformer_predictor import HybridTransformerPredictor  # noqa: E402
from models.text_predictor import TextPricePredictor  # noqa: E402
from models.transformer_predictor import TransformerPricePredictor  # noqa: E402
from training.utils import (  # noqa: E402
    combine_text_columns,
    detect_device,
    evaluate_metrics,
    load_splits,
)

ENGLISH_STOPWORDS: set[str] = set(ENGLISH_STOP_WORDS) | {
    "s",
    "t",
    "ll",
    "re",
    "ve",
    "d",
    "m",
}


def _load_vocab(vocab_path: Path) -> Vocabulary:
    """Load a serialized vocabulary JSON file."""
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing vocabulary file: {vocab_path}")
    with vocab_path.open("r", encoding="utf-8") as input_file:
        return Vocabulary.from_dict(json.load(input_file))


def _load_checkpoint_model(
    checkpoint_path: Path,
    model_class: type[torch.nn.Module],
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate and load a model from a checkpoint path."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = dict(checkpoint["model_config"])

    if model_class is HybridTransformerPredictor:
        if "dropout" not in model_config and "transformer_dropout" in model_config:
            model_config["dropout"] = model_config.pop("transformer_dropout")

    model = model_class(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _predict_text_only(
    model: TextPricePredictor | TransformerPricePredictor,
    test_texts: list[str],
    y_test: np.ndarray,
    vocabulary: Vocabulary,
    max_sequence_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Generate text-only predictions for the full test set."""
    dataset = TextRegressionDataset(
        texts=test_texts,
        targets=y_test.tolist(),
        vocabulary=vocabulary,
        max_sequence_length=max_sequence_length,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_text_batch(
            batch, pad_index=vocabulary.pad_index
        ),
    )

    predictions: list[float] = []
    with torch.no_grad():
        for input_ids, lengths, _ in data_loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            outputs = model(input_ids=input_ids, lengths=lengths)
            predictions.extend(outputs.detach().cpu().tolist())

    return np.asarray(predictions, dtype=np.float32)


def _predict_hybrid_like(
    model: HybridPredictor | HybridTransformerPredictor,
    test_texts: list[str],
    x_test: np.ndarray,
    y_test: np.ndarray,
    vocabulary: Vocabulary,
    max_sequence_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Generate hybrid-like predictions for the full test set."""
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
                input_ids=input_ids, lengths=lengths, tabular_features=tabular_features
            )
            predictions.extend(outputs.detach().cpu().tolist())

    return np.asarray(predictions, dtype=np.float32)


def _token_counter(texts: pd.Series) -> Counter[str]:
    """Count token frequencies while removing stopwords and short tokens."""
    counter: Counter[str] = Counter()
    for text in texts.fillna("").astype(str):
        tokens = simple_tokenize(text)
        filtered = [
            token
            for token in tokens
            if token not in ENGLISH_STOPWORDS and len(token) > 2 and not token.isdigit()
        ]
        counter.update(filtered)
    return counter


def _compute_discriminative_words(
    train_df: pd.DataFrame,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return words associated with expensive vs cheap listings.

    Uses quartiles on log-price and a smoothed log-odds score.
    """
    q1 = float(train_df[TARGET_COLUMN].quantile(0.25))
    q3 = float(train_df[TARGET_COLUMN].quantile(0.75))

    low_group = train_df[train_df[TARGET_COLUMN] <= q1]
    high_group = train_df[train_df[TARGET_COLUMN] >= q3]

    high_counter = _token_counter(
        high_group[DESCRIPTION_COLUMN]
        if DESCRIPTION_COLUMN in high_group.columns
        else pd.Series([], dtype=str)
    )
    low_counter = _token_counter(
        low_group[DESCRIPTION_COLUMN]
        if DESCRIPTION_COLUMN in low_group.columns
        else pd.Series([], dtype=str)
    )

    vocabulary = set(high_counter) | set(low_counter)
    total_high = sum(high_counter.values())
    total_low = sum(low_counter.values())
    vocab_size = max(1, len(vocabulary))

    rows: list[dict[str, float | int | str]] = []
    for token in vocabulary:
        high_count = high_counter[token]
        low_count = low_counter[token]

        p_high = (high_count + 1.0) / (total_high + vocab_size)
        p_low = (low_count + 1.0) / (total_low + vocab_size)
        score = math.log(p_high) - math.log(p_low)

        rows.append(
            {
                "word": token,
                "score": score,
                "high_count": high_count,
                "low_count": low_count,
            }
        )

    score_df = pd.DataFrame(rows).sort_values("score", ascending=False)
    top_high_df = score_df.head(top_n).reset_index(drop=True)
    top_low_df = (
        score_df.tail(top_n).sort_values("score", ascending=True).reset_index(drop=True)
    )
    return top_high_df, top_low_df


def _save_word_chart(
    top_high_df: pd.DataFrame,
    top_low_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create side-by-side bar chart for expensive vs cheap lexical markers."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 8), sharey=False)

    axes[0].barh(top_high_df["word"], top_high_df["score"], color="#1f77b4")
    axes[0].invert_yaxis()
    axes[0].set_title("Top mots associés aux prix élevés")
    axes[0].set_xlabel("Score lexical (log-odds)")

    axes[1].barh(top_low_df["word"], -top_low_df["score"], color="#ff7f0e")
    axes[1].invert_yaxis()
    axes[1].set_title("Top mots associés aux prix bas")
    axes[1].set_xlabel("Score lexical (log-odds inversé)")

    fig.suptitle("Insights lexicaux: annonces chères vs bon marché")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_interpretability(
    city: str,
    use_neighborhood_overview: bool,
    max_sequence_length: int,
    batch_size: int,
    top_n_words: int,
    top_n_residuals: int,
    device_preference: str,
) -> None:
    """Run post-training lexical and residual interpretability analysis."""
    output_dir = RESULTS_DIR / city / "interpretability"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_splits(city)

    x_train, _, x_test, _, _ = preprocess_tabular_features(train_df, val_df, test_df)
    _ = x_train  # explicit: preprocessing fit is required for x_test consistency

    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    test_texts = combine_text_columns(
        test_df, use_neighborhood_overview=use_neighborhood_overview
    ).tolist()

    device = detect_device(device_preference)
    residual_df = test_df.copy()

    model_specs: list[dict[str, object]] = [
        {
            "name": "text_model",
            "checkpoint": RESULTS_DIR / city / "text_model" / "best_model.pth",
            "vocab": RESULTS_DIR / city / "text_model" / "vocab.json",
            "type": "text_only",
            "class": TextPricePredictor,
        },
        {
            "name": "transformer_model",
            "checkpoint": RESULTS_DIR / city / "transformer_model" / "best_model.pth",
            "vocab": RESULTS_DIR / city / "transformer_model" / "vocab.json",
            "type": "text_only",
            "class": TransformerPricePredictor,
        },
        {
            "name": "hybrid",
            "checkpoint": RESULTS_DIR / city / "hybrid" / "best_model.pth",
            "vocab": RESULTS_DIR / city / "hybrid" / "vocab.json",
            "type": "hybrid_like",
            "class": HybridPredictor,
        },
        {
            "name": "hybrid_transformer",
            "checkpoint": RESULTS_DIR / city / "hybrid_transformer" / "best_model.pth",
            "vocab": RESULTS_DIR / city / "hybrid_transformer" / "vocab.json",
            "type": "hybrid_like",
            "class": HybridTransformerPredictor,
        },
    ]

    model_metrics_rows: list[dict[str, float | str]] = []

    for spec in model_specs:
        model_name = str(spec["name"])
        checkpoint_path = spec["checkpoint"]
        vocab_path = spec["vocab"]
        model_type = str(spec["type"])
        model_class = spec["class"]

        if not isinstance(checkpoint_path, Path) or not checkpoint_path.exists():
            continue
        if not isinstance(vocab_path, Path) or not vocab_path.exists():
            continue

        vocabulary = _load_vocab(vocab_path)
        model = _load_checkpoint_model(checkpoint_path, model_class, device)

        if model_type == "text_only":
            predictions = _predict_text_only(
                model=model,
                test_texts=test_texts,
                y_test=y_test,
                vocabulary=vocabulary,
                max_sequence_length=max_sequence_length,
                batch_size=batch_size,
                device=device,
            )
        else:
            predictions = _predict_hybrid_like(
                model=model,
                test_texts=test_texts,
                x_test=x_test,
                y_test=y_test,
                vocabulary=vocabulary,
                max_sequence_length=max_sequence_length,
                batch_size=batch_size,
                device=device,
            )

        prediction_col = f"{model_name}_prediction"
        abs_error_col = f"{model_name}_abs_error"
        residual_df[prediction_col] = predictions
        residual_df[abs_error_col] = np.abs(residual_df[TARGET_COLUMN] - predictions)

        metrics = evaluate_metrics(y_test, predictions)
        model_metrics_rows.append(
            {
                "model": model_name,
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
                "r2": float(metrics["r2"]),
            }
        )

    if not model_metrics_rows:
        raise FileNotFoundError(
            "No model checkpoints found for interpretability. "
            "Train at least one model before running this script."
        )

    model_metrics_df = pd.DataFrame(model_metrics_rows).sort_values("rmse")
    best_model_name = str(model_metrics_df.iloc[0]["model"])
    best_abs_error_col = f"{best_model_name}_abs_error"

    baseline_path = RESULTS_DIR / city / "baseline_metrics.json"
    baseline_rmse = None
    if baseline_path.exists():
        baseline_data = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_rmse = float(baseline_data["models"]["random_forest"]["rmse"])
        residual_df["baseline_rmse_proxy"] = baseline_rmse
        residual_df["best_model_vs_baseline_gain_proxy"] = (
            baseline_rmse - residual_df[best_abs_error_col]
        )
    else:
        residual_df["baseline_rmse_proxy"] = np.nan
        residual_df["best_model_vs_baseline_gain_proxy"] = np.nan

    top_listings_df = residual_df.sort_values(
        "best_model_vs_baseline_gain_proxy", ascending=False
    ).head(top_n_residuals)

    residual_output_columns = [TARGET_COLUMN]
    residual_output_columns.extend(
        [column for column in residual_df.columns if column.endswith("_prediction")]
    )
    residual_output_columns.extend(
        [column for column in residual_df.columns if column.endswith("_abs_error")]
    )
    residual_output_columns.extend(
        ["baseline_rmse_proxy", "best_model_vs_baseline_gain_proxy"]
    )
    top_output_columns = list(residual_output_columns)

    if DESCRIPTION_COLUMN in top_listings_df.columns:
        top_listings_df["description_preview"] = (
            top_listings_df[DESCRIPTION_COLUMN].fillna("").astype(str).str.slice(0, 180)
        )
        top_output_columns.append("description_preview")

    top_listings_df[top_output_columns].to_csv(
        output_dir / "top_best_model_vs_baseline_listings.csv", index=False
    )
    residual_df[residual_output_columns].to_csv(
        output_dir / "test_residuals_all_models.csv", index=False
    )
    residual_df[residual_output_columns].to_csv(
        output_dir / "test_residuals.csv", index=False
    )
    model_metrics_df.to_csv(output_dir / "model_metrics_summary.csv", index=False)

    top_high_df, top_low_df = _compute_discriminative_words(train_df, top_n=top_n_words)
    top_high_df.to_csv(output_dir / "top_words_high_price.csv", index=False)
    top_low_df.to_csv(output_dir / "top_words_low_price.csv", index=False)

    word_chart_path = output_dir / "word_price_signals.png"
    _save_word_chart(top_high_df, top_low_df, word_chart_path)

    summary = {
        "city": city,
        "n_test_rows": int(len(residual_df)),
        "models_evaluated": model_metrics_df["model"].tolist(),
        "best_model": best_model_name,
        "best_model_rmse": float(model_metrics_df.iloc[0]["rmse"]),
        "best_model_mae": float(model_metrics_df.iloc[0]["mae"]),
        "baseline_rmse_proxy": baseline_rmse,
        "top_n_words": top_n_words,
        "top_n_residuals": top_n_residuals,
        "word_chart": str(word_chart_path),
    }
    (output_dir / "interpretability_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== Interpretability analysis complete ===")
    print(f"City: {city}")
    print(f"Rows analysed (test): {len(residual_df):,}")
    print(f"Models evaluated: {', '.join(model_metrics_df['model'].tolist())}")
    print(
        f"Best model (RMSE): {best_model_name} "
        f"({float(model_metrics_df.iloc[0]['rmse']):.4f})"
    )
    if baseline_rmse is not None:
        print(f"Baseline RMSE proxy (RF): {baseline_rmse:.4f}")
    else:
        print("Baseline metrics not found: gain proxy unavailable.")
    print(f"Outputs saved to: {output_dir}")


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for interpretability analysis script."""
    parser = argparse.ArgumentParser(
        description="Post-training interpretability analysis for lexical-economic insights."
    )
    parser.add_argument("--city", default=DEFAULT_CITY)
    parser.add_argument(
        "--use-neighborhood-overview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include neighborhood_overview in text used by the hybrid model.",
    )
    parser.add_argument("--max-sequence-length", type=int, default=MAX_SEQUENCE_LENGTH)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-n-words", type=int, default=20)
    parser.add_argument("--top-n-residuals", type=int, default=50)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device (default: auto).",
    )
    return parser


if __name__ == "__main__":
    arguments = _build_parser().parse_args()
    run_interpretability(
        city=arguments.city,
        use_neighborhood_overview=arguments.use_neighborhood_overview,
        max_sequence_length=arguments.max_sequence_length,
        batch_size=arguments.batch_size,
        top_n_words=arguments.top_n_words,
        top_n_residuals=arguments.top_n_residuals,
        device_preference=arguments.device,
    )
