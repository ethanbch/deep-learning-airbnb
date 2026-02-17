"""Reproducible train / validation / test split."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_SEED, TEST_RATIO, TRAIN_RATIO, VALIDATION_RATIO


@dataclass
class DatasetSplits:
    """Container for the three split DataFrames."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_dataset(
    listings_df: pd.DataFrame,
    random_seed: int = RANDOM_SEED,
    train_ratio: float = TRAIN_RATIO,
    validation_ratio: float = VALIDATION_RATIO,
    test_ratio: float = TEST_RATIO,
) -> DatasetSplits:
    """Split a DataFrame into train, validation and test sets.

    Uses random sampling (sklearn) with a fixed seed to guarantee
    reproducibility.

    Args:
        listings_df: Cleaned listings DataFrame.
        random_seed: Random state for reproducibility.
        train_ratio: Proportion of data for training.
        validation_ratio: Proportion of data for validation.
        test_ratio: Proportion of data for testing.

    Returns:
        A :class:`DatasetSplits` instance.

    Raises:
        ValueError: If the three ratios do not sum to 1.0.
    """
    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Ratios must sum to 1.0, got {total:.4f}.")

    train_df, temp_df = train_test_split(
        listings_df,
        test_size=1.0 - train_ratio,
        random_state=random_seed,
        shuffle=True,
    )

    validation_share = validation_ratio / (validation_ratio + test_ratio)
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=1.0 - validation_share,
        random_state=random_seed,
        shuffle=True,
    )

    return DatasetSplits(
        train=train_df.reset_index(drop=True),
        validation=validation_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )
