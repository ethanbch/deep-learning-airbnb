"""Tabular feature preprocessing and hybrid dataset."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from config import NUMERIC_COLUMNS
from features.text import Vocabulary, collate_text_batch, encode_text


def preprocess_tabular_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, list[str]]:
    """Standardise numeric features (fit on train only).

    Missing values are imputed with the training-set median before
    applying ``StandardScaler``.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.

    Returns:
        Tuple of ``(X_train, X_val, X_test, fitted_scaler, column_names)``.

    Raises:
        ValueError: If no numeric column is available.
    """
    available_columns = [c for c in NUMERIC_COLUMNS if c in train_df.columns]
    if not available_columns:
        raise ValueError("No numeric tabular columns available.")

    train_num = train_df[available_columns].apply(pd.to_numeric, errors="coerce")
    val_num = val_df[available_columns].apply(pd.to_numeric, errors="coerce")
    test_num = test_df[available_columns].apply(pd.to_numeric, errors="coerce")

    medians = train_num.median()
    train_num = train_num.fillna(medians)
    val_num = val_num.fillna(medians)
    test_num = test_num.fillna(medians)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_num).astype(np.float32)
    x_val = scaler.transform(val_num).astype(np.float32)
    x_test = scaler.transform(test_num).astype(np.float32)

    return x_train, x_val, x_test, scaler, available_columns


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
    tabular_tensor = torch.tensor([row for _, row, _ in batch], dtype=torch.float32)
    return input_ids, lengths, tabular_tensor, targets
