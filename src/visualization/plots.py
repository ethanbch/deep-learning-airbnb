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


def save_comparison_chart(city: str, hybrid_r2: float) -> None:
    """Generate a bar chart comparing R² across all models.

    Reads previously saved baseline and text-model metrics from disk.

    Args:
        city: City identifier (used to locate result files).
        hybrid_r2: R² score of the hybrid model on the test set.
    """
    baseline_path = RESULTS_DIR / city / "baseline_metrics.json"
    text_model_path = RESULTS_DIR / city / "text_model" / "test_metrics.json"

    if not baseline_path.exists() or not text_model_path.exists():
        print("Comparison files missing – chart not generated.")
        return

    baseline_data = json.loads(baseline_path.read_text(encoding="utf-8"))
    text_data = json.loads(text_model_path.read_text(encoding="utf-8"))

    baseline_r2 = baseline_data["models"]["random_forest"]["r2"]
    text_r2 = text_data["metrics"]["r2"]

    labels = ["Baseline RF", "Text Model", "Hybrid"]
    values = [baseline_r2, text_r2, hybrid_r2]

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
