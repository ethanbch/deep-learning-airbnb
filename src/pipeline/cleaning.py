"""Data cleaning: price parsing, column selection and outlier removal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    CATEGORICAL_COLUMNS,
    MAX_PRICE,
    MIN_PRICE,
    NUMERIC_COLUMNS,
    RAW_PRICE_COLUMN,
    SPATIAL_COLUMNS,
    TARGET_COLUMN,
    TEXT_COLUMNS,
)


def parse_price(price_series: pd.Series) -> pd.Series:
    """Convert a price column (e.g. ``"$1,200.00"``) to numeric float.

    Args:
        price_series: Raw price strings.

    Returns:
        Numeric Series with unparseable values set to ``NaN``.
    """
    cleaned = (
        price_series.astype(str)
        .str.replace(r"[^\d,.-]", "", regex=True)
        .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def select_columns(
    listings_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Keep only the columns needed for modelling.

    Creates the ``log_price`` target column from the raw price.

    Args:
        listings_df: Raw listings DataFrame.

    Returns:
        A tuple of (filtered DataFrame, list of missing column names).
    """
    selected = [
        RAW_PRICE_COLUMN,
        *NUMERIC_COLUMNS,
        *SPATIAL_COLUMNS,
        *CATEGORICAL_COLUMNS,
        *TEXT_COLUMNS,
    ]
    missing_columns = [col for col in selected if col not in listings_df.columns]
    available_columns = [col for col in selected if col in listings_df.columns]

    filtered_df = listings_df[available_columns].copy()

    if RAW_PRICE_COLUMN in filtered_df.columns:
        filtered_df[RAW_PRICE_COLUMN] = parse_price(filtered_df[RAW_PRICE_COLUMN])
        filtered_df[TARGET_COLUMN] = np.log(
            filtered_df[RAW_PRICE_COLUMN].where(filtered_df[RAW_PRICE_COLUMN] > 0)
        )

    return filtered_df, missing_columns


def count_missing_text(listings_df: pd.DataFrame) -> pd.Series:
    """Count ``NaN`` values in each text column.

    Args:
        listings_df: DataFrame with (some) text columns.

    Returns:
        Series mapping column name to missing count.
    """
    available = [col for col in TEXT_COLUMNS if col in listings_df.columns]
    if not available:
        return pd.Series(dtype="int64")
    return listings_df[available].isna().sum()


def _coerce_numeric_columns(listings_df: pd.DataFrame) -> pd.DataFrame:
    """Cast numeric and spatial columns to proper dtype, setting errors to ``NaN``."""
    all_numeric = [
        col
        for col in [*NUMERIC_COLUMNS, *SPATIAL_COLUMNS]
        if col in listings_df.columns
    ]
    for col in all_numeric:
        listings_df[col] = pd.to_numeric(listings_df[col], errors="coerce")
    return listings_df


def clean_dataset(
    listings_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove outliers, invalid rows and normalise text fields.

    Args:
        listings_df: DataFrame output of :func:`select_columns`.

    Returns:
        A tuple of (cleaned DataFrame, summary dict with row counts).
    """
    cleaned_df = listings_df.copy()
    initial_rows = len(cleaned_df)

    cleaned_df = _coerce_numeric_columns(cleaned_df)

    if RAW_PRICE_COLUMN in cleaned_df.columns:
        cleaned_df = cleaned_df[
            cleaned_df[RAW_PRICE_COLUMN].between(MIN_PRICE, MAX_PRICE)
        ]

    cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)

    required_columns = [
        TARGET_COLUMN,
        *NUMERIC_COLUMNS,
        *SPATIAL_COLUMNS,
        *CATEGORICAL_COLUMNS,
    ]
    available_required = [col for col in required_columns if col in cleaned_df.columns]
    cleaned_df = cleaned_df.dropna(subset=available_required)

    if "room_type" in cleaned_df.columns:
        cleaned_df["room_type"] = cleaned_df["room_type"].astype(str).str.strip()
        cleaned_df = cleaned_df[cleaned_df["room_type"] != ""]

    for col in TEXT_COLUMNS:
        if col in cleaned_df.columns:
            cleaned_df[col] = (
                cleaned_df[col]
                .fillna("")
                .astype(str)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )

    final_rows = len(cleaned_df)
    summary = {
        "rows_initial": initial_rows,
        "rows_final": final_rows,
        "rows_removed": initial_rows - final_rows,
    }
    return cleaned_df, summary
