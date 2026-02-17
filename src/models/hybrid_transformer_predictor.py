"""Hybrid predictor combining Transformer text encoder and tabular MLP."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.tabular_mlp import TabularMLP


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for token embeddings."""

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
        sequence_length = input_embeddings.size(1)
        encoded = input_embeddings + self.positional[:, :sequence_length, :]
        return self.dropout(encoded)


class HybridTransformerPredictor(nn.Module):
    """Text Transformer + tabular MLP fusion model for price regression.

    Architecture::

        text_input -> Embedding -> PositionalEncoding -> TransformerEncoder
                  -> Masked MeanPooling -> text_vector

        tabular_input -> TabularMLP -> tabular_vector

        concat(text_vector, tabular_vector) -> Regressor -> scalar
    """

    def __init__(
        self,
        vocab_size: int,
        pad_index: int,
        tabular_input_dim: int,
        embedding_dim: int,
        transformer_hidden_dim: int,
        transformer_heads: int,
        transformer_layers: int,
        tabular_hidden_dim: int,
        fusion_hidden_dim: int,
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
            nhead=transformer_heads,
            dim_feedforward=transformer_hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )

        self.tabular_encoder = TabularMLP(
            input_dim=tabular_input_dim,
            hidden_dim=tabular_hidden_dim,
            dropout=dropout,
        )

        fusion_input_dim = embedding_dim + tabular_hidden_dim
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
        del lengths
        padding_mask = input_ids.eq(self.pad_index)

        text_embeddings = self.embedding(input_ids)
        text_embeddings = self.positional_encoding(text_embeddings)

        encoded = self.text_encoder(
            text_embeddings,
            src_key_padding_mask=padding_mask,
        )

        valid_tokens = (~padding_mask).unsqueeze(-1)
        summed = (encoded * valid_tokens).sum(dim=1)
        counts = valid_tokens.sum(dim=1).clamp(min=1)
        text_vector = summed / counts

        tabular_vector = self.tabular_encoder(tabular_features)
        fused = torch.cat([text_vector, tabular_vector], dim=1)

        return self.regressor(fused).squeeze(1)
