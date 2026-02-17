"""Shared training utilities: seeding, device detection, evaluation."""

from __future__ import annotations

import gc
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


@dataclass
class CheckpointState:
    """State values returned by checkpoint resume helper."""

    resumed: bool
    start_epoch: int
    best_val_loss: float
    patience_counter: int


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


def clear_device_cache(device: torch.device, aggressive: bool = False) -> None:
    """Clear backend caches to reduce memory pressure during long training runs.

    Args:
        device: Active torch device.
        aggressive: When True, also triggers Python garbage collection.
    """
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    if (
        device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "empty_cache")
    ):
        torch.mps.empty_cache()

    if aggressive:
        gc.collect()


def maybe_resume_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    device: torch.device,
    resume_from_checkpoint: bool,
    initial_best_val_loss: float,
) -> CheckpointState:
    """Resume model/optimizer state from checkpoint if available.

    Args:
        model: Model instance to restore.
        optimizer: Optimizer instance to restore.
        checkpoint_path: Checkpoint file path.
        device: Device used for map_location.
        resume_from_checkpoint: Whether resume is enabled.
        initial_best_val_loss: Fallback best validation loss.

    Returns:
        :class:`CheckpointState` with resume metadata.
    """
    if not resume_from_checkpoint or not checkpoint_path.exists():
        return CheckpointState(
            resumed=False,
            start_epoch=1,
            best_val_loss=initial_best_val_loss,
            patience_counter=0,
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    best_val_loss = float(checkpoint.get("best_val_loss", initial_best_val_loss))
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    patience_counter = int(checkpoint.get("patience_counter", 0))

    print(
        f"Resuming from checkpoint: {checkpoint_path} "
        f"(start_epoch={start_epoch}, best_val_loss={best_val_loss:.6f})"
    )

    return CheckpointState(
        resumed=True,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        patience_counter=patience_counter,
    )


def build_checkpoint_payload(
    model: torch.nn.Module,
    model_config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    best_val_loss: float,
    epoch: int,
    patience_counter: int,
) -> dict[str, Any]:
    """Build a standardized checkpoint dictionary for all training scripts."""
    return {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "epoch": epoch,
        "patience_counter": patience_counter,
    }


def initialize_epoch_metrics_log(
    metrics_log_path: Path,
    resume_from_existing_log: bool,
) -> None:
    """Prepare the per-epoch metrics log file.

    Args:
        metrics_log_path: Target JSONL file path.
        resume_from_existing_log: Keep and append to existing log when True.
    """
    metrics_log_path.parent.mkdir(parents=True, exist_ok=True)

    if not resume_from_existing_log:
        metrics_log_path.write_text("", encoding="utf-8")


def append_epoch_metrics(
    metrics_log_path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    patience_counter: int,
) -> None:
    """Append one epoch record to JSONL metrics log."""
    payload = {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "best_val_loss": float(best_val_loss),
        "patience_counter": int(patience_counter),
    }

    with metrics_log_path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(payload) + "\n")
