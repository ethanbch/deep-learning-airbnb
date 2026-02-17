"""Text-only price predictor (Embedding + LSTM/GRU -> regression)."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.text_encoder import TextEncoder


class TextPricePredictor(nn.Module):
    """End-to-end text-only regression model.

    Wraps a :class:`TextEncoder` and adds a dropout + linear head
    producing a single scalar prediction.

    Args:
        vocab_size: Vocabulary size.
        embedding_dim: Token embedding dimension.
        hidden_dim: RNN hidden dimension.
        num_layers: Number of stacked RNN layers.
        dropout: Dropout probability.
        pad_index: Padding token index.
        rnn_type: ``"lstm"`` or ``"gru"``.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_index: int,
        rnn_type: str = "lstm",
    ) -> None:
        super().__init__()
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_index=pad_index,
            rnn_type=rnn_type,
        )
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Predict a scalar from text.

        Args:
            input_ids: ``(batch, max_seq_len)`` padded token ids.
            lengths: ``(batch,)`` actual lengths.

        Returns:
            ``(batch,)`` predicted values.
        """
        hidden = self.text_encoder(input_ids, lengths)
        return self.regressor(self.dropout(hidden)).squeeze(1)
