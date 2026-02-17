"""Two-layer MLP for tabular (numeric) features."""

from __future__ import annotations

import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    """Simple feed-forward network for tabular inputs.

    Architecture: ``Linear -> ReLU -> Dropout -> Linear -> ReLU``.

    Args:
        input_dim: Number of input features.
        hidden_dim: Width of the hidden layers.
        dropout: Dropout probability after the first hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features: ``(batch, input_dim)`` numeric feature tensor.

        Returns:
            ``(batch, hidden_dim)`` learned representation.
        """
        return self.network(features)
