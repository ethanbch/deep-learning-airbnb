"""Data pipeline: download, clean and split Airbnb listing data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DEFAULT_CITY, PROCESSED_DATA_DIR

from .cleaning import clean_dataset, count_missing_text, select_columns
from .download import download_file, find_latest_listings_url
from .splitting import DatasetSplits, split_dataset


def run_download(city: str = DEFAULT_CITY) -> pd.DataFrame:
    """Download raw listings and select relevant columns.

    Args:
        city: City identifier for Inside Airbnb (e.g. ``"london"``).

    Returns:
        DataFrame with selected and price-transformed columns.
    """
    print(f"Searching for the latest listing URL for {city.title()}...")
    listings_url = find_latest_listings_url(city=city)
    print(f"URL found: {listings_url}")

    output_path = Path("data") / "raw" / city / "listings.csv.gz"
    print(f"Downloading to: {output_path}")
    download_file(listings_url, output_path)

    print("Loading CSV into a pandas DataFrame...")
    listings_df = pd.read_csv(output_path, low_memory=False)
    print(f"Raw shape: {listings_df.shape}")

    filtered_df, missing_columns = select_columns(listings_df)
    if missing_columns:
        print(
            "Columns absent from the downloaded snapshot (ignored): "
            + ", ".join(missing_columns)
        )

    print("\nFirst rows (selected columns):")
    print(filtered_df.head())

    missing_text = count_missing_text(filtered_df)
    if missing_text.empty:
        print("\nNo requested text columns available in this file.")
    else:
        print("\nMissing values per text column:")
        print(missing_text)

    return filtered_df


def _save_splits(city: str, splits: DatasetSplits) -> None:
    """Save train / validation / test DataFrames to disk.

    Args:
        city: City identifier used as subdirectory name.
        splits: The three DataFrames to persist.
    """
    city_dir = Path(PROCESSED_DATA_DIR) / city
    city_dir.mkdir(parents=True, exist_ok=True)

    train_path = city_dir / "train.csv"
    validation_path = city_dir / "validation.csv"
    test_path = city_dir / "test.csv"

    splits.train.to_csv(train_path, index=False)
    splits.validation.to_csv(validation_path, index=False)
    splits.test.to_csv(test_path, index=False)

    print("\nSplit datasets saved:")
    print(f"  Train:      {train_path} ({len(splits.train):,} rows)")
    print(f"  Validation: {validation_path} ({len(splits.validation):,} rows)")
    print(f"  Test:       {test_path} ({len(splits.test):,} rows)")


def run_data_pipeline(city: str = DEFAULT_CITY) -> DatasetSplits:
    """Execute the full data pipeline: download -> clean -> split.

    Args:
        city: City identifier (default from config).

    Returns:
        A ``DatasetSplits`` instance with train, validation and test DataFrames.
    """
    raw_df = run_download(city=city)

    print("\nCleaning dataset...")
    cleaned_df, cleaning_summary = clean_dataset(raw_df)
    print(
        f"Cleaning done: {cleaning_summary['rows_initial']:,} -> "
        f"{cleaning_summary['rows_final']:,} rows "
        f"({cleaning_summary['rows_removed']:,} removed)."
    )

    print("Splitting into train / validation / test (no leakage)...")
    splits = split_dataset(cleaned_df)
    _save_splits(city=city, splits=splits)
    return splits
