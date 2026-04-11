"""Reproducible stratified train / validation / test split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_SEED, TARGET_COLUMN, TEST_RATIO, TRAIN_RATIO, VALIDATION_RATIO


@dataclass
class DatasetSplits:
    """Container for the three split DataFrames."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _build_stratification_bins(
    listings_df: pd.DataFrame,
    n_price_bins: int = 10,
) -> pd.Series:
    """Create stratification labels from price quantiles and room type.

    Combines binned log-price with room type to produce a composite
    stratification key.  This ensures each split has a similar distribution
    of prices **and** room types.

    Args:
        listings_df: Cleaned listings DataFrame.
        n_price_bins: Number of quantile bins for the target variable.

    Returns:
        A Series of string labels suitable for ``stratify=`` parameter.
    """
    price_bins = pd.qcut(
        listings_df[TARGET_COLUMN],
        q=n_price_bins,
        labels=False,
        duplicates="drop",
    ).astype(str)

    if "room_type" in listings_df.columns:
        room = listings_df["room_type"].astype(str)
        return price_bins + "_" + room
    return price_bins


def split_dataset(
    listings_df: pd.DataFrame,
    random_seed: int = RANDOM_SEED,
    train_ratio: float = TRAIN_RATIO,
    validation_ratio: float = VALIDATION_RATIO,
    test_ratio: float = TEST_RATIO,
) -> DatasetSplits:
    """Split a DataFrame into train, validation and test sets.

    Uses **stratified** random sampling to guarantee that each split has
    a similar distribution of prices and room types.

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

    strat_labels = _build_stratification_bins(listings_df)

    # Some composite bins may have very few members.  Fall back to
    # un-stratified split for bins with fewer than 2 members by
    # merging them into a catch-all group.
    counts = strat_labels.value_counts()
    rare_labels = counts[counts < 2].index
    strat_labels = strat_labels.replace(rare_labels, "__rare__")

    train_df, temp_df, strat_train, strat_temp = train_test_split(
        listings_df,
        strat_labels,
        test_size=1.0 - train_ratio,
        random_state=random_seed,
        shuffle=True,
        stratify=strat_labels,
    )

    validation_share = validation_ratio / (validation_ratio + test_ratio)

    # Second split also stratified
    temp_strat_counts = strat_temp.value_counts()
    temp_rare = temp_strat_counts[temp_strat_counts < 2].index
    strat_temp_safe = strat_temp.replace(temp_rare, "__rare__")

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=1.0 - validation_share,
        random_state=random_seed,
        shuffle=True,
        stratify=strat_temp_safe,
    )

    return DatasetSplits(
        train=train_df.reset_index(drop=True),
        validation=validation_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )
