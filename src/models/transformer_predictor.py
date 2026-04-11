"""Basic Transformer encoder model for Airbnb price regression."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.layers import PositionalEncoding


class TransformerPricePredictor(nn.Module):
    """Embedding -> PositionalEncoding -> TransformerEncoder -> MeanPool -> Linear."""

    def __init__(
        self,
        vocab_size: int,
        pad_index: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        max_sequence_length: int,
    ) -> None:
        super().__init__()
        self.pad_index = pad_index

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index,
        )
        self.positional_encoding = PositionalEncoding(
            embedding_dim=embedding_dim,
            dropout=dropout,
            max_len=max_sequence_length,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.regressor = nn.Linear(embedding_dim, 1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        del lengths
        padding_mask = input_ids.eq(self.pad_index)

        embeddings = self.embedding(input_ids)
        embeddings = self.positional_encoding(embeddings)

        encoded = self.encoder(
            embeddings,
            src_key_padding_mask=padding_mask,
        )

        valid_tokens = (~padding_mask).unsqueeze(-1)
        summed = (encoded * valid_tokens).sum(dim=1)
        counts = valid_tokens.sum(dim=1).clamp(min=1)
        pooled = summed / counts

        return self.regressor(pooled).squeeze(1)
