"""Shared neural network layers used across multiple models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for token embeddings.

    Args:
        embedding_dim: Dimensionality of the token embeddings.
        dropout: Dropout probability applied after adding positional signal.
        max_len: Maximum sequence length supported.
    """

    positional: torch.Tensor

    def __init__(self, embedding_dim: int, dropout: float, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )

        positional = torch.zeros(1, max_len, embedding_dim)
        positional[0, :, 0::2] = torch.sin(position * div_term)
        positional[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional", positional)

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            input_embeddings: ``(batch, seq_len, embedding_dim)`` tensor.

        Returns:
            Positionally-encoded embeddings with dropout applied.
        """
        sequence_length = input_embeddings.size(1)
        encoded = input_embeddings + self.positional[:, :sequence_length, :]
        return self.dropout(encoded)
