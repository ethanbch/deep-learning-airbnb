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
    best_val_index = int(np.argmin(val_losses))
    best_epoch = int(epochs[best_val_index])
    best_val_loss = float(val_losses[best_val_index])

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2.0)
    plt.plot(epochs, val_losses, label="Validation Loss", linewidth=2.0)

    plt.axvline(
        x=best_epoch,
        color="red",
        linestyle="--",
        linewidth=1.6,
        alpha=0.9,
        label=f"Best val epoch ({best_epoch})",
    )
    plt.scatter(
        [best_epoch],
        [best_val_loss],
        color="red",
        s=60,
        zorder=5,
    )
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)

    summary = (
        f"Best val: {best_val_loss:.4f} (epoch {best_epoch})\n"
        f"Final train: {train_losses[-1]:.4f} | Final val: {val_losses[-1]:.4f}"
    )
    plt.text(
        0.02,
        0.98,
        summary,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.75},
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_comparison_chart(
    city: str,
    hybrid_r2: float | None = None,
    hybrid_transformer_r2: float | None = None,
) -> None:
    """Generate a bar chart comparing R² across available models.

    Reads previously saved metrics from disk and includes optional models
    when artifacts are available.

    Args:
        city: City identifier (used to locate result files).
        hybrid_r2: Optional in-memory R² score for the hybrid model.
        hybrid_transformer_r2: Optional in-memory R² for hybrid-transformer model.
    """
    baseline_path = RESULTS_DIR / city / "baseline_metrics.json"
    text_model_path = RESULTS_DIR / city / "text_model" / "test_metrics.json"
    hybrid_path = RESULTS_DIR / city / "hybrid" / "test_metrics.json"
    transformer_path = RESULTS_DIR / city / "transformer_model" / "test_metrics.json"
    hybrid_transformer_path = (
        RESULTS_DIR / city / "hybrid_transformer" / "test_metrics.json"
    )

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

    if hybrid_transformer_r2 is None and hybrid_transformer_path.exists():
        hybrid_transformer_data = json.loads(
            hybrid_transformer_path.read_text(encoding="utf-8")
        )
        hybrid_transformer_r2 = hybrid_transformer_data["metrics"]["r2"]

    if hybrid_transformer_r2 is not None:
        labels.append("Hybrid Transformer")
        values.append(hybrid_transformer_r2)

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
