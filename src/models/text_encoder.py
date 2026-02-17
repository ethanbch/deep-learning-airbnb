"""Recurrent text encoder (LSTM / GRU)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """Embed token ids and encode them with a recurrent network.

    The last hidden state of the top RNN layer is returned as a
    fixed-size text representation.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        embedding_dim: Dimensionality of the token embeddings.
        hidden_dim: Number of hidden units in the RNN.
        num_layers: Number of stacked RNN layers.
        dropout: Dropout probability (applied between layers when
                 ``num_layers > 1``).
        pad_index: Vocabulary index used for padding.
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
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index,
        )
        self.rnn_type = rnn_type.lower()
        effective_dropout = dropout if num_layers > 1 else 0.0

        rnn_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode a padded batch of token-id sequences.

        Args:
            input_ids: ``(batch, max_seq_len)`` padded token ids.
            lengths: ``(batch,)`` actual sequence lengths.

        Returns:
            ``(batch, hidden_dim)`` last hidden state.
        """
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        if self.rnn_type == "gru":
            _, hidden = self.rnn(packed)
        else:
            _, (hidden, _) = self.rnn(packed)

        return hidden[-1]
