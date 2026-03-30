"""Compare model error across price ranges for poster analysis.

Creates a line chart of MAE by price decile on the test split, with one
curve per available model.
"""

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
    BATCH_SIZE,
    CATEGORICAL_COLUMNS,
    DEFAULT_CITY,
    MAX_SEQUENCE_LENGTH,
    NEIGHBORHOOD_OVERVIEW_COLUMN,
    NUMERIC_COLUMNS,
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
from models.hybrid_transformer_predictor import HybridTransformerPredictor  # noqa: E402
from models.text_predictor import TextPricePredictor  # noqa: E402
from models.transformer_predictor import TransformerPricePredictor  # noqa: E402
from training.utils import combine_text_columns, detect_device  # noqa: E402


def _load_splits(city: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed train/val/test splits with compatibility fallback."""
    try:
        from pipeline.splitting import (
            load_splits as pipeline_load_splits,  # type: ignore[attr-defined]
        )

        return pipeline_load_splits(city)
    except Exception:
        from training.utils import load_splits as training_load_splits

        return training_load_splits(city)


def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    columns = [*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS]
    available = [column for column in columns if column in df.columns]
    return df[available].copy()


def _load_vocab(path: Path) -> Vocabulary:
    if not path.exists():
        raise FileNotFoundError(f"Missing vocabulary file: {path}")
    return Vocabulary.from_dict(json.loads(path.read_text(encoding="utf-8")))


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


def _predict_hybrid_transformer(
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


def _compute_binned_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    n_bins: int,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    frame["abs_error"] = np.abs(frame["y_true"] - frame["y_pred"])
    frame["price_bin"] = pd.qcut(frame["y_true"], q=n_bins, duplicates="drop")

    grouped = (
        frame.groupby("price_bin", observed=True)
        .agg(
            mae=("abs_error", "mean"),
            mean_price_log=("y_true", "mean"),
        )
        .reset_index(drop=True)
    )
    grouped["mean_price_real"] = np.exp(grouped["mean_price_log"])
    grouped["model"] = model_name
    return grouped[["model", "mean_price_real", "mean_price_log", "mae"]]


def compare_models_by_price(
    city: str,
    use_neighborhood_overview: bool,
    max_sequence_length: int,
    batch_size: int,
    n_bins: int,
    device_preference: str,
) -> Path:
    train_df, val_df, test_df = _load_splits(city)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    test_texts = combine_text_columns(
        test_df,
        use_neighborhood_overview=use_neighborhood_overview,
    ).tolist()
    x_train, _, x_test, _, _ = preprocess_tabular_features(train_df, val_df, test_df)
    _ = x_train

    device = detect_device(device_preference)
    predictions: dict[str, np.ndarray] = {}

    # Random Forest (optional, as requested)
    try:
        import joblib

        rf_path = RESULTS_DIR / city / "baseline" / "random_forest.joblib"
        if rf_path.exists():
            rf_model = joblib.load(rf_path)
            x_test_raw = _select_features(test_df)
            predictions["Random Forest"] = np.asarray(
                rf_model.predict(x_test_raw), dtype=np.float32
            )
        else:
            print(f"[skip] Random Forest not found: {rf_path}")
    except Exception as error:
        print(f"[skip] Random Forest loading failed: {error}")

    # LSTM text model
    try:
        lstm_checkpoint = RESULTS_DIR / city / "text_model" / "best_model.pth"
        lstm_vocab = RESULTS_DIR / city / "text_model" / "vocab.json"
        if lstm_checkpoint.exists() and lstm_vocab.exists():
            lstm_model = _load_checkpoint_model(
                lstm_checkpoint, TextPricePredictor, device
            )
            lstm_vocabulary = _load_vocab(lstm_vocab)
            predictions["LSTM"] = _predict_text_model(
                model=lstm_model,
                texts=test_texts,
                targets=y_test,
                vocabulary=lstm_vocabulary,
                max_sequence_length=max_sequence_length,
                batch_size=batch_size,
                device=device,
            )
        else:
            print("[skip] LSTM artifacts missing.")
    except Exception as error:
        print(f"[skip] LSTM loading/prediction failed: {error}")

    # Transformer text model
    try:
        transformer_checkpoint = (
            RESULTS_DIR / city / "transformer_model" / "best_model.pth"
        )
        transformer_vocab = RESULTS_DIR / city / "transformer_model" / "vocab.json"
        if transformer_checkpoint.exists() and transformer_vocab.exists():
            transformer_model = _load_checkpoint_model(
                transformer_checkpoint,
                TransformerPricePredictor,
                device,
            )
            transformer_vocabulary = _load_vocab(transformer_vocab)
            predictions["Transformer"] = _predict_text_model(
                model=transformer_model,
                texts=test_texts,
                targets=y_test,
                vocabulary=transformer_vocabulary,
                max_sequence_length=max_sequence_length,
                batch_size=batch_size,
                device=device,
            )
        else:
            print("[skip] Transformer artifacts missing.")
    except Exception as error:
        print(f"[skip] Transformer loading/prediction failed: {error}")

    # Hybrid Transformer
    try:
        hybrid_t_checkpoint = (
            RESULTS_DIR / city / "hybrid_transformer" / "best_model.pth"
        )
        hybrid_t_vocab = RESULTS_DIR / city / "hybrid_transformer" / "vocab.json"
        if hybrid_t_checkpoint.exists() and hybrid_t_vocab.exists():
            hybrid_t_model = _load_checkpoint_model(
                hybrid_t_checkpoint,
                HybridTransformerPredictor,
                device,
            )
            hybrid_t_vocabulary = _load_vocab(hybrid_t_vocab)
            predictions["Hybrid Transformer"] = _predict_hybrid_transformer(
                model=hybrid_t_model,
                texts=test_texts,
                tabular_features=x_test,
                targets=y_test,
                vocabulary=hybrid_t_vocabulary,
                max_sequence_length=max_sequence_length,
                batch_size=batch_size,
                device=device,
            )
        else:
            print("[skip] Hybrid Transformer artifacts missing.")
    except Exception as error:
        print(f"[skip] Hybrid Transformer loading/prediction failed: {error}")

    if not predictions:
        raise RuntimeError(
            "No model could be loaded. Check artifacts under data/results/<city>/."
        )

    binned_rows: list[pd.DataFrame] = []
    for model_name, y_pred in predictions.items():
        binned_rows.append(
            _compute_binned_mae(y_test, y_pred, model_name=model_name, n_bins=n_bins)
        )

    chart_df = pd.concat(binned_rows, ignore_index=True)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10.5, 6.5))
    ax = sns.lineplot(
        data=chart_df,
        x="mean_price_real",
        y="mae",
        hue="model",
        marker="o",
        linewidth=2.2,
    )
    ax.set_title("Performance par Gamme de Prix")
    ax.set_xlabel("Prix moyen de la tranche (échelle réelle)")
    ax.set_ylabel("MAE (Erreur Moyenne)")

    output_dir = RESULTS_DIR / city / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "error_by_price_bin.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare model error by price range (test deciles)."
    )
    parser.add_argument("--city", default=DEFAULT_CITY)
    parser.add_argument(
        "--use-neighborhood-overview",
        action=argparse.BooleanOptionalAction,
        default=(NEIGHBORHOOD_OVERVIEW_COLUMN is not None),
    )
    parser.add_argument("--max-sequence-length", type=int, default=MAX_SEQUENCE_LENGTH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    output = compare_models_by_price(
        city=args.city,
        use_neighborhood_overview=args.use_neighborhood_overview,
        max_sequence_length=args.max_sequence_length,
        batch_size=args.batch_size,
        n_bins=args.n_bins,
        device_preference=args.device,
    )
    print(f"Error-by-price chart saved to: {output}")
