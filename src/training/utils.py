"""Shared training utilities: seeding, device detection, evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import DESCRIPTION_COLUMN, NEIGHBORHOOD_OVERVIEW_COLUMN, PROJECT_ROOT


@dataclass
class TrainingHistory:
    """Accumulator for per-epoch train and validation losses."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all backends.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device(preference: str = "auto") -> torch.device:
    """Select the best available compute device.

    Priority: CUDA -> MPS (Apple Silicon) -> CPU.

    Args:
        preference: Explicit device (``"cuda"``, ``"mps"``, ``"cpu"``)
                    or ``"auto"`` for automatic detection.

    Returns:
        A :class:`torch.device` instance.
    """
    pref = preference.lower()
    if pref in {"cpu", "cuda", "mps"}:
        if pref == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but unavailable – falling back to auto.")
        elif pref == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            print("MPS requested but unavailable – falling back to auto.")
        else:
            return torch.device(pref)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_splits(
    city: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the train / validation / test CSVs for a city.

    Args:
        city: City identifier (subdirectory name under ``data/processed/``).

    Returns:
        Tuple of ``(train_df, validation_df, test_df)``.

    Raises:
        FileNotFoundError: If any expected CSV file is missing.
    """
    base_dir = PROJECT_ROOT / "data" / "processed" / city
    train_path = base_dir / "train.csv"
    val_path = base_dir / "validation.csv"
    test_path = base_dir / "test.csv"

    for path in (train_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")

    return (
        pd.read_csv(train_path),
        pd.read_csv(val_path),
        pd.read_csv(test_path),
    )


def combine_text_columns(
    listings_df: pd.DataFrame,
    use_neighborhood_overview: bool,
) -> pd.Series:
    """Merge description and (optionally) neighborhood overview into one column.

    Args:
        listings_df: DataFrame with text columns.
        use_neighborhood_overview: Whether to append the overview text.

    Returns:
        A Series of combined text strings.
    """
    description = listings_df.get(
        DESCRIPTION_COLUMN, pd.Series("", index=listings_df.index)
    ).fillna("")

    if (
        use_neighborhood_overview
        and NEIGHBORHOOD_OVERVIEW_COLUMN in listings_df.columns
    ):
        overview = listings_df[NEIGHBORHOOD_OVERVIEW_COLUMN].fillna("")
        return (description.astype(str) + " " + overview.astype(str)).str.strip()

    return description.astype(str)


def evaluate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute RMSE, MAE and R² for a set of predictions.

    Args:
        y_true: Ground-truth values.
        y_pred: Model predictions.

    Returns:
        Dict with ``rmse``, ``mae`` and ``r2`` keys.
    """
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "rmse": mse**0.5,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
