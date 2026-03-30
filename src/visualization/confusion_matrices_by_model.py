"""Generate confusion matrices for all available models on binned price classes.

Since this project is regression-based, predictions are converted to ordered
price classes (quantile bins) before computing confusion matrices.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import (  # noqa: E402
    BATCH_SIZE,
    CATEGORICAL_COLUMNS,
    DEFAULT_CITY,
    MAX_SEQUENCE_LENGTH,
    NUMERIC_COLUMNS,
    RANDOM_SEED,
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
)
from models.hybrid_predictor import HybridPredictor  # noqa: E402
from models.hybrid_transformer_predictor import HybridTransformerPredictor  # noqa: E402
from models.text_predictor import TextPricePredictor  # noqa: E402
from models.transformer_predictor import TransformerPricePredictor  # noqa: E402
from training.utils import (  # noqa: E402
    combine_text_columns,
    detect_device,
    load_splits,
)


def _load_vocab(path: Path) -> Vocabulary:
    if not path.exists():
        raise FileNotFoundError(f"Missing vocabulary file: {path}")
    return Vocabulary.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    columns = [*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS]
    available = [column for column in columns if column in df.columns]
    return df[available].copy()


def _build_preprocessor(features_df: pd.DataFrame) -> ColumnTransformer:
    numeric = [column for column in NUMERIC_COLUMNS if column in features_df.columns]
    categorical = [
        column for column in CATEGORICAL_COLUMNS if column in features_df.columns
    ]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def _predict_baselines(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}

    x_train = _select_features(train_df)
    x_test = _select_features(test_df)
    y_train = train_df[TARGET_COLUMN]

    preprocessor = _build_preprocessor(x_train)

    # OLS baseline (always reconstructed)
    ols_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", LinearRegression())]
    )
    ols_pipeline.fit(x_train, y_train)
    predictions["Baseline OLS"] = np.asarray(
        ols_pipeline.predict(x_test), dtype=np.float32
    )

    # RF: use saved joblib if available, else reconstruct
    rf_joblib = RESULTS_DIR / DEFAULT_CITY / "baseline" / "random_forest.joblib"
    if rf_joblib.exists():
        rf_model = joblib.load(rf_joblib)
        predictions["Random Forest"] = np.asarray(
            rf_model.predict(x_test), dtype=np.float32
        )
    else:
        rf_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=2,
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        rf_pipeline.fit(x_train, y_train)
        predictions["Random Forest"] = np.asarray(
            rf_pipeline.predict(x_test), dtype=np.float32
        )

    return predictions


def _load_checkpoint_model(
    checkpoint_path: Path,
    model_class: type[torch.nn.Module],
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = dict(checkpoint["model_config"])

    if model_class is HybridTransformerPredictor:
        if "dropout" not in model_config and "transformer_dropout" in model_config:
            model_config["dropout"] = model_config.pop("transformer_dropout")

    model = model_class(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _predict_text_model(
    model: torch.nn.Module,
    texts: list[str],
    targets: np.ndarray,
    vocabulary: Vocabulary,
    max_sequence_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = TextRegressionDataset(
        texts=texts,
        targets=targets.tolist(),
        vocabulary=vocabulary,
        max_sequence_length=max_sequence_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_text_batch(
            batch, pad_index=vocabulary.pad_index
        ),
    )

    predictions: list[float] = []
    with torch.no_grad():
        for input_ids, lengths, _ in loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            outputs = model(input_ids=input_ids, lengths=lengths)
            predictions.extend(outputs.detach().cpu().tolist())

    return np.asarray(predictions, dtype=np.float32)


def _predict_hybrid_model(
    model: torch.nn.Module,
    texts: list[str],
    tabular_features: np.ndarray,
    targets: np.ndarray,
    vocabulary: Vocabulary,
    max_sequence_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = HybridDataset(
        texts=texts,
        tabular_features=tabular_features,
        targets=targets,
        vocabulary=vocabulary,
        max_sequence_length=max_sequence_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_hybrid_batch(
            batch, pad_index=vocabulary.pad_index
        ),
    )

    predictions: list[float] = []
    with torch.no_grad():
        for input_ids, lengths, tabular, _ in loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            tabular = tabular.to(device)
            outputs = model(
                input_ids=input_ids, lengths=lengths, tabular_features=tabular
            )
            predictions.extend(outputs.detach().cpu().tolist())

    return np.asarray(predictions, dtype=np.float32)


def _price_class_labels(n_classes: int) -> list[str]:
    defaults = ["Budget", "Economy", "Mid-range", "Premium", "Luxury"]
    if n_classes <= len(defaults):
        return defaults[:n_classes]
    return [f"Class {index + 1}" for index in range(n_classes)]


def _plot_confusion_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
    normalize: bool,
) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7.8, 6.6))

    if normalize:
        display_matrix = np.where(
            matrix.sum(axis=1, keepdims=True) > 0,
            matrix / matrix.sum(axis=1, keepdims=True),
            0.0,
        )
        fmt = ".2f"
        cbar_label = "Row-normalized proportion"
    else:
        display_matrix = matrix
        fmt = "d"
        cbar_label = "Count"

    sns.heatmap(
        display_matrix,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar_kws={"label": cbar_label},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted price class")
    ax.set_ylabel("True price class")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_confusion_matrices(
    city: str,
    use_neighborhood_overview: bool,
    max_sequence_length: int,
    batch_size: int,
    n_classes: int,
    device_preference: str,
) -> Path:
    train_df, val_df, test_df = load_splits(city)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    test_texts = combine_text_columns(test_df, use_neighborhood_overview).tolist()
    x_train, _, x_test, _, _ = preprocess_tabular_features(train_df, val_df, test_df)
    _ = x_train

    device = detect_device(device_preference)
    predictions: dict[str, np.ndarray] = {}

    predictions.update(_predict_baselines(train_df, test_df))

    deep_specs = [
        (
            "LSTM",
            RESULTS_DIR / city / "text_model" / "best_model.pth",
            RESULTS_DIR / city / "text_model" / "vocab.json",
            TextPricePredictor,
            "text",
        ),
        (
            "Transformer",
            RESULTS_DIR / city / "transformer_model" / "best_model.pth",
            RESULTS_DIR / city / "transformer_model" / "vocab.json",
            TransformerPricePredictor,
            "text",
        ),
        (
            "Hybrid",
            RESULTS_DIR / city / "hybrid" / "best_model.pth",
            RESULTS_DIR / city / "text_model" / "vocab.json",
            HybridPredictor,
            "hybrid",
        ),
        (
            "Hybrid Transformer",
            RESULTS_DIR / city / "hybrid_transformer" / "best_model.pth",
            RESULTS_DIR / city / "hybrid_transformer" / "vocab.json",
            HybridTransformerPredictor,
            "hybrid",
        ),
    ]

    for label, checkpoint_path, vocab_path, model_class, model_type in deep_specs:
        try:
            if not checkpoint_path.exists() or not vocab_path.exists():
                print(f"[skip] {label} artifacts missing.")
                continue

            model = _load_checkpoint_model(checkpoint_path, model_class, device)
            vocabulary = _load_vocab(vocab_path)

            if model_type == "text":
                predictions[label] = _predict_text_model(
                    model=model,
                    texts=test_texts,
                    targets=y_test,
                    vocabulary=vocabulary,
                    max_sequence_length=max_sequence_length,
                    batch_size=batch_size,
                    device=device,
                )
            else:
                predictions[label] = _predict_hybrid_model(
                    model=model,
                    texts=test_texts,
                    tabular_features=x_test,
                    targets=y_test,
                    vocabulary=vocabulary,
                    max_sequence_length=max_sequence_length,
                    batch_size=batch_size,
                    device=device,
                )
        except Exception as error:
            print(f"[skip] {label} prediction failed: {error}")

    if not predictions:
        raise RuntimeError(
            "No model predictions available to build confusion matrices."
        )

    _, bin_edges = pd.qcut(y_test, q=n_classes, retbins=True, duplicates="drop")
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    labels = _price_class_labels(len(bin_edges) - 1)
    y_true_cls = pd.cut(y_test, bins=bin_edges, labels=False, include_lowest=True)

    output_dir = RESULTS_DIR / city / "plots" / "confusion_matrices"
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, y_pred in predictions.items():
        y_pred_cls = pd.cut(y_pred, bins=bin_edges, labels=False, include_lowest=True)
        matrix = confusion_matrix(
            y_true_cls, y_pred_cls, labels=list(range(len(labels)))
        )

        slug = model_name.lower().replace(" ", "_")
        _plot_confusion_heatmap(
            matrix=matrix,
            labels=labels,
            title=f"{model_name} — Confusion Matrix (counts)",
            output_path=output_dir / f"{slug}_confusion_counts.png",
            normalize=False,
        )
        _plot_confusion_heatmap(
            matrix=matrix,
            labels=labels,
            title=f"{model_name} — Confusion Matrix (normalized)",
            output_path=output_dir / f"{slug}_confusion_normalized.png",
            normalize=True,
        )

    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate confusion matrices for each available model."
    )
    parser.add_argument("--city", default=DEFAULT_CITY)
    parser.add_argument(
        "--use-neighborhood-overview",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-sequence-length", type=int, default=MAX_SEQUENCE_LENGTH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--n-classes",
        type=int,
        default=4,
        help="Number of quantile-based price classes.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
    )
    return parser


if __name__ == "__main__":
    arguments = _build_parser().parse_args()
    output = generate_confusion_matrices(
        city=arguments.city,
        use_neighborhood_overview=arguments.use_neighborhood_overview,
        max_sequence_length=arguments.max_sequence_length,
        batch_size=arguments.batch_size,
        n_classes=arguments.n_classes,
        device_preference=arguments.device,
    )
    print(f"Confusion matrices saved to: {output}")
