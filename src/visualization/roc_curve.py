"""ROC curve for regression models by binarizing target price.

This creates a binary task:
    high_price = log_price >= quantile_threshold
Then plots one ROC curve per available prediction column in
interpretability residual files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import DEFAULT_CITY, RESULTS_DIR, TARGET_COLUMN


def _find_residuals_file(city: str) -> Path:
    base = RESULTS_DIR / city / "interpretability"
    preferred = base / "test_residuals_all_models.csv"
    fallback = base / "test_residuals.csv"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Residuals file not found in {base}. Run interpretability.py first."
    )


def generate_roc_curve(
    city: str,
    positive_quantile: float,
) -> Path:
    residuals_path = _find_residuals_file(city)
    df = pd.read_csv(residuals_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in {residuals_path}"
        )

    prediction_columns = [c for c in df.columns if c.endswith("_prediction")]
    if not prediction_columns:
        raise ValueError("No '*_prediction' columns found in residuals file.")

    threshold = float(df[TARGET_COLUMN].quantile(positive_quantile))
    y_true = (df[TARGET_COLUMN] >= threshold).astype(int)

    plt.figure(figsize=(8.5, 6.2))
    for column in prediction_columns:
        fpr, tpr, _ = roc_curve(y_true, df[column])
        roc_auc = auc(fpr, tpr)
        label = column.replace("_prediction", "")
        plt.plot(fpr, tpr, linewidth=2.0, label=f"{label} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC — High price vs rest (q={positive_quantile:.2f})")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_dir = RESULTS_DIR / city / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "roc_curve.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Residuals source: {residuals_path}")
    print(f"Binary threshold (log_price q={positive_quantile:.2f}): {threshold:.4f}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ROC curve from saved residuals."
    )
    parser.add_argument("--city", default=DEFAULT_CITY)
    parser.add_argument(
        "--positive-quantile",
        type=float,
        default=0.75,
        help="Quantile defining positive class (high-price). Default: 0.75",
    )
    args = parser.parse_args()

    out = generate_roc_curve(city=args.city, positive_quantile=args.positive_quantile)
    print(f"ROC figure saved to: {out}")
