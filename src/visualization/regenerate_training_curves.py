"""Regenerate training curves from persisted epoch metrics (JSONL)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import DEFAULT_CITY, RESULTS_DIR  # noqa: E402
from visualization.plots import save_training_curves  # noqa: E402


def _read_epoch_losses(epoch_metrics_path: Path) -> tuple[list[float], list[float]]:
    """Read all train/validation losses from an epoch_metrics JSONL file."""
    if not epoch_metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {epoch_metrics_path}")

    train_losses: list[float] = []
    val_losses: list[float] = []

    with epoch_metrics_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    if not train_losses or not val_losses:
        raise ValueError(f"No usable epoch rows found in: {epoch_metrics_path}")

    return train_losses, val_losses


def regenerate_training_curves(city: str, model: str) -> Path:
    """Regenerate the training curves PNG for one model folder."""
    model_folders = {
        "text": "text_model",
        "hybrid": "hybrid",
        "transformer": "transformer_model",
        "hybrid-transformer": "hybrid_transformer",
    }
    titles = {
        "text": "Text Model Training Curves",
        "hybrid": "Hybrid Model Training Curves",
        "transformer": "Transformer Model Training Curves",
        "hybrid-transformer": "Hybrid Transformer Training Curves",
    }

    if model not in model_folders:
        available = ", ".join(model_folders)
        raise ValueError(f"Unknown model '{model}'. Available: {available}")

    model_dir = RESULTS_DIR / city / model_folders[model]
    epoch_metrics_path = model_dir / "epoch_metrics.jsonl"
    curve_path = model_dir / "training_curves.png"

    train_losses, val_losses = _read_epoch_losses(epoch_metrics_path)
    save_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        output_path=curve_path,
        title=titles[model],
    )

    return curve_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate training curves from epoch_metrics.jsonl."
    )
    parser.add_argument("--city", default=DEFAULT_CITY)
    parser.add_argument(
        "--model",
        choices=["text", "hybrid", "transformer", "hybrid-transformer"],
        default="text",
        help="Model family to regenerate curve for.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    output = regenerate_training_curves(city=args.city, model=args.model)
    print(f"Training curve regenerated: {output}")
