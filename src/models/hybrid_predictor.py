"""Hybrid predictor combining text and tabular features for regression."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.tabular_mlp import TabularMLP
from models.text_encoder import TextEncoder


class HybridPredictor(nn.Module):
    """Multimodal model fusing text and tabular representations.

    Architecture::

        text  -> TextEncoder  -> text_repr  -+
                                              +-> concat -> FC -> ReLU -> Dropout -> FC -> scalar
        table -> TabularMLP   -> tab_repr   -+

    Args:
        vocab_size: Vocabulary size for the text encoder.
        pad_index: Padding token index.
        tabular_input_dim: Number of tabular numeric features.
        embedding_dim: Token embedding dimension.
        text_hidden_dim: RNN hidden dimension.
        tabular_hidden_dim: MLP hidden dimension.
        fusion_hidden_dim: Dimension of the fusion layer.
        num_layers: Stacked RNN layers.
        dropout: Dropout probability.
        rnn_type: ``"lstm"`` or ``"gru"``.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_index: int,
        tabular_input_dim: int,
        embedding_dim: int = 100,
        text_hidden_dim: int = 64,
        tabular_hidden_dim: int = 32,
        fusion_hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        rnn_type: str = "lstm",
    ) -> None:
        super().__init__()
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=text_hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_index=pad_index,
            rnn_type=rnn_type,
        )
        self.tabular_encoder = TabularMLP(
            input_dim=tabular_input_dim,
            hidden_dim=tabular_hidden_dim,
            dropout=dropout,
        )

        fusion_input_dim = text_hidden_dim + tabular_hidden_dim
        self.regressor = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a scalar from text + tabular inputs.

        Args:
            input_ids: ``(batch, max_seq_len)`` padded token ids.
            lengths: ``(batch,)`` actual sequence lengths.
            tabular_features: ``(batch, tabular_input_dim)`` numeric features.

        Returns:
            ``(batch,)`` predicted values.
        """
        text_repr = self.text_encoder(input_ids=input_ids, lengths=lengths)
        tabular_repr = self.tabular_encoder(tabular_features)
        fused = torch.cat([text_repr, tabular_repr], dim=1)
        return self.regressor(fused).squeeze(1)
