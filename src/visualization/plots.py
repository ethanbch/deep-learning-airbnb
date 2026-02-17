"""Plot generation for training curves and model comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR


def save_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    output_path: Path,
    title: str = "Training Curves",
) -> None:
    """Plot and save train / validation loss curves.

    Args:
        train_losses: Per-epoch training losses.
        val_losses: Per-epoch validation losses.
        output_path: Destination PNG path.
        title: Chart title.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_comparison_chart(city: str, hybrid_r2: float | None = None) -> None:
    """Generate a bar chart comparing R² across available models.

    Reads previously saved metrics from disk and includes optional models
    when artifacts are available.

    Args:
        city: City identifier (used to locate result files).
        hybrid_r2: Optional in-memory R² score for the hybrid model.
    """
    baseline_path = RESULTS_DIR / city / "baseline_metrics.json"
    text_model_path = RESULTS_DIR / city / "text_model" / "test_metrics.json"
    hybrid_path = RESULTS_DIR / city / "hybrid" / "test_metrics.json"
    transformer_path = RESULTS_DIR / city / "transformer_model" / "test_metrics.json"

    if not baseline_path.exists() or not text_model_path.exists():
        print("Comparison files missing – chart not generated.")
        return

    baseline_data = json.loads(baseline_path.read_text(encoding="utf-8"))
    text_data = json.loads(text_model_path.read_text(encoding="utf-8"))

    labels = ["Baseline RF", "Text Model"]
    values = [
        baseline_data["models"]["random_forest"]["r2"],
        text_data["metrics"]["r2"],
    ]

    if hybrid_r2 is None and hybrid_path.exists():
        hybrid_data = json.loads(hybrid_path.read_text(encoding="utf-8"))
        hybrid_r2 = hybrid_data["metrics"]["r2"]

    if hybrid_r2 is not None:
        labels.append("Hybrid")
        values.append(hybrid_r2)

    if transformer_path.exists():
        transformer_data = json.loads(transformer_path.read_text(encoding="utf-8"))
        labels.append("Transformer")
        values.append(transformer_data["metrics"]["r2"])

    output_path = RESULTS_DIR / city / "comparison_chart.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, values)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.ylabel("R² (Test)")
    plt.title("Model Comparison")
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
