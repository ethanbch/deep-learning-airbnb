"""Tabular feature preprocessing and hybrid dataset."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from config import (
    CATEGORICAL_COLUMNS,
    LONDON_CENTRE_LAT,
    LONDON_CENTRE_LON,
    NUMERIC_COLUMNS,
    SPATIAL_COLUMNS,
)
from features.text import Vocabulary, collate_text_batch, encode_text


def _haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: float,
    lon2: float,
) -> np.ndarray:
    """Compute Haversine distance in km between arrays of coords and a point."""
    r = 6371.0  # Earth radius in km
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


def _engineer_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered spatial features from raw lat/lon.

    Produces:
    - ``dist_to_centre_km``: Haversine distance to city centre.
    - ``lat_sin``, ``lat_cos``, ``lon_sin``, ``lon_cos``: Cyclical encoding
      of coordinates (scaled to reasonable range for London).

    Falls back gracefully if lat/lon columns are missing.
    """
    result = pd.DataFrame(index=df.index)

    if "latitude" not in df.columns or "longitude" not in df.columns:
        return result

    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")

    result["dist_to_centre_km"] = _haversine_km(
        lat.to_numpy(), lon.to_numpy(), LONDON_CENTRE_LAT, LONDON_CENTRE_LON
    )
    # Normalised lat/lon relative to city centre (captures direction)
    result["lat_offset"] = lat - LONDON_CENTRE_LAT
    result["lon_offset"] = lon - LONDON_CENTRE_LON

    return result


def _encode_categorical_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """One-hot encode categorical columns (fit on train only).

    Returns:
        Tuple of (train_encoded, val_encoded, test_encoded, column_names).
    """
    available_cats = [c for c in CATEGORICAL_COLUMNS if c in train_df.columns]
    if not available_cats:
        empty = np.empty((0, 0), dtype=np.float32)
        return empty, empty, empty, []

    # Build category mapping from training set only
    cat_maps: dict[str, list[str]] = {}
    for col in available_cats:
        categories = sorted(train_df[col].dropna().astype(str).unique())
        cat_maps[col] = categories

    def _encode(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        encoded_parts: list[np.ndarray] = []
        names: list[str] = []
        for col in available_cats:
            categories = cat_maps[col]
            values = df[col].fillna("").astype(str)
            for cat in categories:
                encoded_parts.append((values == cat).astype(np.float32).to_numpy().reshape(-1, 1))
                names.append(f"{col}_{cat}")
        if encoded_parts:
            return np.hstack(encoded_parts), names
        return np.empty((len(df), 0), dtype=np.float32), names

    train_enc, col_names = _encode(train_df)
    val_enc, _ = _encode(val_df)
    test_enc, _ = _encode(test_df)

    return train_enc, val_enc, test_enc, col_names


def preprocess_tabular_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, list[str]]:
    """Preprocess all tabular features: numeric + spatial + categorical.

    Numeric and spatial features are standardised (fit on train only).
    Categorical features are one-hot encoded (fit on train only).
    Missing values are imputed with the training-set median before scaling.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.

    Returns:
        Tuple of ``(X_train, X_val, X_test, fitted_scaler, column_names)``.

    Raises:
        ValueError: If no feature column is available.
    """
    # --- Numeric features ---------------------------------------------------
    available_numeric = [c for c in NUMERIC_COLUMNS if c in train_df.columns]

    train_num = train_df[available_numeric].apply(pd.to_numeric, errors="coerce") if available_numeric else pd.DataFrame(index=train_df.index)
    val_num = val_df[available_numeric].apply(pd.to_numeric, errors="coerce") if available_numeric else pd.DataFrame(index=val_df.index)
    test_num = test_df[available_numeric].apply(pd.to_numeric, errors="coerce") if available_numeric else pd.DataFrame(index=test_df.index)

    # --- Spatial features ---------------------------------------------------
    train_spatial = _engineer_spatial_features(train_df)
    val_spatial = _engineer_spatial_features(val_df)
    test_spatial = _engineer_spatial_features(test_df)

    # Combine numeric + spatial for joint scaling
    train_continuous = pd.concat([train_num, train_spatial], axis=1)
    val_continuous = pd.concat([val_num, val_spatial], axis=1)
    test_continuous = pd.concat([test_num, test_spatial], axis=1)

    continuous_columns = list(train_continuous.columns)

    if not continuous_columns:
        raise ValueError("No numeric or spatial tabular columns available.")

    # Impute and scale
    medians = train_continuous.median()
    train_continuous = train_continuous.fillna(medians)
    val_continuous = val_continuous.fillna(medians)
    test_continuous = test_continuous.fillna(medians)

    scaler = StandardScaler()
    x_train_cont = scaler.fit_transform(train_continuous).astype(np.float32)
    x_val_cont = scaler.transform(val_continuous).astype(np.float32)
    x_test_cont = scaler.transform(test_continuous).astype(np.float32)

    # --- Categorical features -----------------------------------------------
    train_cat, val_cat, test_cat, cat_columns = _encode_categorical_features(
        train_df, val_df, test_df
    )

    # --- Combine all features -----------------------------------------------
    all_columns = continuous_columns + cat_columns

    if train_cat.size > 0:
        x_train = np.hstack([x_train_cont, train_cat])
        x_val = np.hstack([x_val_cont, val_cat])
        x_test = np.hstack([x_test_cont, test_cat])
    else:
        x_train = x_train_cont
        x_val = x_val_cont
        x_test = x_test_cont

    return x_train, x_val, x_test, scaler, all_columns


class HybridDataset(Dataset):
    """PyTorch Dataset combining encoded text with tabular features.

    Args:
        texts: Raw text strings.
        tabular_features: 2-D array of preprocessed numeric features.
        targets: Regression target values.
        vocabulary: Fitted :class:`~features.text.Vocabulary`.
        max_sequence_length: Maximum number of tokens per text sample.
    """

    def __init__(
        self,
        texts: Sequence[str],
        tabular_features: np.ndarray,
        targets: np.ndarray,
        vocabulary: Vocabulary,
        max_sequence_length: int,
    ) -> None:
        self.texts = [str(t) for t in texts]
        self.tabular_features = tabular_features
        self.targets = targets.astype(np.float32)
        self.vocabulary = vocabulary
        self.max_sequence_length = max_sequence_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[list[int], np.ndarray, float]:
        token_ids = encode_text(
            self.texts[index],
            vocabulary=self.vocabulary,
            max_sequence_length=self.max_sequence_length,
        )
        return token_ids, self.tabular_features[index], float(self.targets[index])


def collate_hybrid_batch(
    batch: list[tuple[list[int], np.ndarray, float]],
    pad_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad text sequences and stack with tabular features into tensors.

    Args:
        batch: List of ``(token_ids, tabular_row, target)`` tuples.
        pad_index: Index used for right-padding shorter sequences.

    Returns:
        Tuple of ``(input_ids, lengths, tabular_features, targets)``.
    """
    text_batch = [(token_ids, target) for token_ids, _, target in batch]
    input_ids, lengths, targets = collate_text_batch(text_batch, pad_index=pad_index)
    tabular_array = np.asarray([row for _, row, _ in batch], dtype=np.float32)
    tabular_tensor = torch.from_numpy(tabular_array)
    return input_ids, lengths, tabular_tensor, targets
