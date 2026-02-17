"""NLP preprocessing: tokenisation, vocabulary and text datasets."""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

import torch
from torch.utils.data import Dataset

from config import PAD_INDEX, PAD_TOKEN, UNK_INDEX, UNK_TOKEN

TOKEN_PATTERN: re.Pattern[str] = re.compile(r"[a-z0-9']+")


def simple_tokenize(text: str) -> list[str]:
    """Lower-case and split text into alphanumeric tokens.

    Args:
        text: Raw input string.

    Returns:
        List of lower-cased tokens.
    """
    return TOKEN_PATTERN.findall(str(text).lower())


class Vocabulary:
    """Bidirectional mapping between tokens and integer indices.

    Attributes:
        token_to_index: Dict mapping token strings to integer ids.
        index_to_token: List where position *i* is the token for id *i*.
    """

    def __init__(
        self,
        token_to_index: dict[str, int],
        index_to_token: list[str],
    ) -> None:
        self.token_to_index = token_to_index
        self.index_to_token = index_to_token

    def __len__(self) -> int:
        """Return the total number of tokens (including PAD and UNK)."""
        return len(self.index_to_token)

    @property
    def pad_index(self) -> int:
        """Index reserved for the padding token."""
        return PAD_INDEX

    @property
    def unk_index(self) -> int:
        """Index reserved for the unknown-word token."""
        return UNK_INDEX

    def encode_tokens(self, tokens: Sequence[str]) -> list[int]:
        """Convert a list of tokens to their integer ids.

        Args:
            tokens: Token strings to encode.

        Returns:
            List of integer ids (unknown tokens map to :pyattr:`unk_index`).
        """
        return [self.token_to_index.get(token, UNK_INDEX) for token in tokens]

    def to_dict(self) -> dict[str, object]:
        """Serialise the vocabulary to a JSON-friendly dict."""
        return {
            "token_to_index": self.token_to_index,
            "index_to_token": self.index_to_token,
            "pad_token": PAD_TOKEN,
            "unk_token": UNK_TOKEN,
            "pad_index": PAD_INDEX,
            "unk_index": UNK_INDEX,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Vocabulary:
        """Reconstruct a vocabulary from a dict (e.g. loaded from JSON).

        Args:
            payload: Dict with ``token_to_index`` and ``index_to_token`` keys.

        Returns:
            A new :class:`Vocabulary` instance.

        Raises:
            ValueError: If the payload format is invalid.
        """
        token_to_index = payload.get("token_to_index", {})
        index_to_token = payload.get("index_to_token", [])
        if not isinstance(token_to_index, dict) or not isinstance(index_to_token, list):
            raise ValueError("Invalid vocabulary format.")
        return cls(token_to_index=token_to_index, index_to_token=index_to_token)


def build_vocabulary(
    texts: Sequence[str],
    max_vocab_size: int,
) -> Vocabulary:
    """Build a :class:`Vocabulary` from a corpus of texts.

    Args:
        texts: Iterable of raw text strings (typically the training set).
        max_vocab_size: Maximum number of unique tokens to retain
                        (excluding PAD and UNK).

    Returns:
        A :class:`Vocabulary` fitted on *texts*.
    """
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))

    most_common = [token for token, _ in counter.most_common(max_vocab_size)]
    index_to_token = [PAD_TOKEN, UNK_TOKEN, *most_common]
    token_to_index = {token: idx for idx, token in enumerate(index_to_token)}
    return Vocabulary(token_to_index=token_to_index, index_to_token=index_to_token)


def encode_text(
    text: str,
    vocabulary: Vocabulary,
    max_sequence_length: int,
) -> list[int]:
    """Tokenise and encode a single text string.

    Args:
        text: Raw input text.
        vocabulary: Fitted vocabulary for id lookup.
        max_sequence_length: Truncation limit.

    Returns:
        List of integer token ids (length <= *max_sequence_length*).
    """
    token_ids = vocabulary.encode_tokens(simple_tokenize(text))[:max_sequence_length]
    if not token_ids:
        token_ids = [vocabulary.unk_index]
    return token_ids


class TextRegressionDataset(Dataset):
    """PyTorch Dataset mapping (text, target) pairs to encoded sequences.

    Args:
        texts: Raw text strings.
        targets: Numeric regression targets.
        vocabulary: Fitted :class:`Vocabulary`.
        max_sequence_length: Maximum number of tokens per sample.
    """

    def __init__(
        self,
        texts: Sequence[str],
        targets: Sequence[float],
        vocabulary: Vocabulary,
        max_sequence_length: int,
    ) -> None:
        self.texts = [str(t) for t in texts]
        self.targets = [float(t) for t in targets]
        self.vocabulary = vocabulary
        self.max_sequence_length = max_sequence_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[list[int], float]:
        token_ids = encode_text(
            self.texts[index],
            vocabulary=self.vocabulary,
            max_sequence_length=self.max_sequence_length,
        )
        return token_ids, self.targets[index]


def collate_text_batch(
    batch: list[tuple[list[int], float]],
    pad_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of variable-length sequences and stack into tensors.

    Args:
        batch: List of ``(token_ids, target)`` pairs.
        pad_index: Index used for right-padding shorter sequences.

    Returns:
        A tuple of ``(input_ids, lengths, targets)`` tensors.
    """
    sequences, targets = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)

    padded = [seq + [pad_index] * (max_len - len(seq)) for seq in sequences]

    input_ids = torch.tensor(padded, dtype=torch.long)
    length_tensor = torch.tensor(lengths, dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    return input_ids, length_tensor, target_tensor
