"""Generate wordclouds for high-price and low-price listing language."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import DEFAULT_CITY, DESCRIPTION_COLUMN, RESULTS_DIR, TARGET_COLUMN
from features.text import simple_tokenize
from training.utils import load_splits

STOPWORDS: set[str] = set(ENGLISH_STOP_WORDS) | {
    "s",
    "t",
    "ll",
    "re",
    "ve",
    "d",
    "m",
}


def _token_counter(texts: pd.Series) -> Counter[str]:
    counter: Counter[str] = Counter()
    for text in texts.fillna("").astype(str):
        tokens = simple_tokenize(text)
        filtered = [
            token
            for token in tokens
            if token not in STOPWORDS and len(token) > 2 and not token.isdigit()
        ]
        counter.update(filtered)
    return counter


def _compute_discriminative_frequencies(
    train_df: pd.DataFrame,
    max_words: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute high-vs-low word scores with same log-odds method as interpretability.

    Returns:
        Tuple of ``(high_word_weights, low_word_weights)`` where values are
        positive weights suitable for WordCloud.
    """
    q1 = float(train_df[TARGET_COLUMN].quantile(0.25))
    q3 = float(train_df[TARGET_COLUMN].quantile(0.75))

    low_group = train_df[train_df[TARGET_COLUMN] <= q1]
    high_group = train_df[train_df[TARGET_COLUMN] >= q3]

    high_counter = _token_counter(
        low_group[DESCRIPTION_COLUMN]
        if DESCRIPTION_COLUMN in low_group.columns
        else pd.Series([], dtype=str)
    )
    low_counter = _token_counter(
        high_group[DESCRIPTION_COLUMN]
        if DESCRIPTION_COLUMN in high_group.columns
        else pd.Series([], dtype=str)
    )

    # Keep exact mapping used in interpretability.py
    high_counter, low_counter = low_counter, high_counter

    vocabulary = set(high_counter) | set(low_counter)
    total_high = sum(high_counter.values())
    total_low = sum(low_counter.values())
    vocab_size = max(1, len(vocabulary))

    scores: list[tuple[str, float]] = []
    for token in vocabulary:
        p_high = (high_counter[token] + 1.0) / (total_high + vocab_size)
        p_low = (low_counter[token] + 1.0) / (total_low + vocab_size)
        score = math.log(p_high) - math.log(p_low)
        scores.append((token, score))

    high_tokens = sorted(
        (item for item in scores if item[1] > 0), key=lambda x: x[1], reverse=True
    )[:max_words]
    low_tokens = sorted((item for item in scores if item[1] < 0), key=lambda x: x[1])[
        :max_words
    ]

    high_weights = {token: float(score) for token, score in high_tokens}
    low_weights = {token: float(-score) for token, score in low_tokens}
    return high_weights, low_weights


def _make_wordcloud(
    frequencies: dict[str, float], output_path: Path, colormap: str
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not frequencies:
        raise ValueError("No tokens available to generate wordcloud.")

    cloud = WordCloud(
        width=1800,
        height=1000,
        background_color="white",
        colormap=colormap,
        max_words=250,
        random_state=42,
    ).generate_from_frequencies(frequencies)

    plt.figure(figsize=(12, 7))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_wordclouds(city: str) -> tuple[Path, Path, Path]:
    train_df, _, _ = load_splits(city)

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    if DESCRIPTION_COLUMN not in train_df.columns:
        raise ValueError(f"Missing description column: {DESCRIPTION_COLUMN}")

    high_frequencies, low_frequencies = _compute_discriminative_frequencies(
        train_df,
        max_words=220,
    )

    plots_dir = RESULTS_DIR / city / "plots"
    high_path = plots_dir / "wordcloud_high_price.png"
    low_path = plots_dir / "wordcloud_low_price.png"
    combined_path = plots_dir / "wordcloud_high_low_price.png"

    _make_wordcloud(high_frequencies, high_path, colormap="Reds")
    _make_wordcloud(low_frequencies, low_path, colormap="Blues")

    high_cloud = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        colormap="Reds",
        max_words=220,
        random_state=42,
    ).generate_from_frequencies(high_frequencies)
    low_cloud = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        colormap="Blues",
        max_words=220,
        random_state=42,
    ).generate_from_frequencies(low_frequencies)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    axes[0].imshow(high_cloud, interpolation="bilinear")
    axes[0].axis("off")
    axes[0].set_title("Mots associés aux prix élevés")

    axes[1].imshow(low_cloud, interpolation="bilinear")
    axes[1].axis("off")
    axes[1].set_title("Mots associés aux prix bas")

    fig.tight_layout()
    fig.savefig(combined_path, dpi=300)
    plt.close(fig)

    return high_path, low_path, combined_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate high-price vs low-price wordclouds from training data."
    )
    parser.add_argument("--city", default=DEFAULT_CITY)
    args = parser.parse_args()

    high, low, combined = generate_wordclouds(city=args.city)
    print(f"High-price wordcloud: {high}")
    print(f"Low-price wordcloud: {low}")
    print(f"Combined wordcloud: {combined}")
