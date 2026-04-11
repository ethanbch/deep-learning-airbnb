"""Train tabular baseline models (OLS + Random Forest + TF-IDF Ridge)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# -- path setup ---------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    DEFAULT_CITY,
    NUMERIC_COLUMNS,
    RANDOM_SEED,
    RESULTS_DIR,
    SPATIAL_COLUMNS,
    TARGET_COLUMN,
)
from training.utils import combine_text_columns, evaluate_metrics, load_splits  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_features(listings_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only numeric + spatial + categorical feature columns.

    Args:
        listings_df: Raw or processed listings DataFrame.

    Returns:
        DataFrame with baseline feature columns only.
    """
    candidates = [*NUMERIC_COLUMNS, *SPATIAL_COLUMNS, *CATEGORICAL_COLUMNS]
    available = [c for c in candidates if c in listings_df.columns]
    return listings_df[available].copy()


def _build_preprocessor(
    features_df: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build an sklearn preprocessor (impute + one-hot encode).

    Args:
        features_df: Feature DataFrame used to detect available columns.

    Returns:
        Tuple of ``(preprocessor, numeric_cols, categorical_cols)``.
    """
    numeric = [
        c for c in [*NUMERIC_COLUMNS, *SPATIAL_COLUMNS] if c in features_df.columns
    ]
    categorical = [c for c in CATEGORICAL_COLUMNS if c in features_df.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return preprocessor, numeric, categorical


def _save_results(
    city: str,
    metrics: dict,
    rf_importance_df: pd.DataFrame,
    ols_coef_df: pd.DataFrame,
) -> None:
    """Persist baseline metrics and feature importances to disk.

    Args:
        city: City identifier.
        metrics: Combined metrics dict for both models.
        rf_importance_df: Random-forest feature importances.
        ols_coef_df: OLS regression coefficients.
    """
    results_dir = RESULTS_DIR / city
    results_dir.mkdir(parents=True, exist_ok=True)

    with (results_dir / "baseline_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    rf_importance_df.to_csv(results_dir / "rf_feature_importance.csv", index=False)
    ols_coef_df.to_csv(results_dir / "linear_coefficients.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_baseline(city: str = DEFAULT_CITY) -> None:
    """Train OLS, Random Forest and TF-IDF+Ridge baselines and save results.

    Args:
        city: City identifier whose processed splits will be loaded.
    """
    train_df, _, test_df = load_splits(city)

    x_train = _select_features(train_df)
    x_test = _select_features(test_df)

    if TARGET_COLUMN not in train_df.columns or TARGET_COLUMN not in test_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from splits.")

    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    preprocessor, _, _ = _build_preprocessor(x_train)

    # ---- OLS ----------------------------------------------------------------
    linear_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", LinearRegression())]
    )
    linear_pipeline.fit(x_train, y_train)
    y_pred_linear = np.asarray(linear_pipeline.predict(x_test))
    linear_metrics = evaluate_metrics(y_test.to_numpy(), y_pred_linear)

    # ---- Random Forest ------------------------------------------------------
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_pipeline.fit(x_train, y_train)
    y_pred_rf = np.asarray(rf_pipeline.predict(x_test))
    rf_metrics = evaluate_metrics(y_test.to_numpy(), y_pred_rf)

    # ---- TF-IDF + Ridge (text-only baseline) --------------------------------
    train_texts = combine_text_columns(train_df, use_neighborhood_overview=True)
    test_texts = combine_text_columns(test_df, use_neighborhood_overview=True)

    tfidf_pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20_000,
                    ngram_range=(1, 2),
                    min_df=3,
                    sublinear_tf=True,
                ),
            ),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
        ]
    )
    tfidf_pipeline.fit(train_texts, y_train)
    y_pred_tfidf = np.asarray(tfidf_pipeline.predict(test_texts))
    tfidf_metrics = evaluate_metrics(y_test.to_numpy(), y_pred_tfidf)

    # ---- Feature analysis ---------------------------------------------------
    feature_names = linear_pipeline.named_steps["preprocessor"].get_feature_names_out()

    ols_coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": linear_pipeline.named_steps["model"].coef_,
        }
    ).sort_values("coefficient", key=lambda s: s.abs(), ascending=False)

    rf_importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": rf_pipeline.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    all_metrics = {
        "city": city,
        "target": TARGET_COLUMN,
        "models": {
            "linear_regression_ols": linear_metrics,
            "random_forest": rf_metrics,
            "tfidf_ridge": tfidf_metrics,
        },
    }

    # ---- Console output -----------------------------------------------------
    print("\n=== Tabular Baseline (no text) ===")
    print("\n[Linear Regression / OLS]")
    print(f"  RMSE: {linear_metrics['rmse']:.4f}")
    print(f"  MAE:  {linear_metrics['mae']:.4f}")
    print(f"  R2:   {linear_metrics['r2']:.4f}")

    print("\n[Random Forest]")
    print(f"  RMSE: {rf_metrics['rmse']:.4f}")
    print(f"  MAE:  {rf_metrics['mae']:.4f}")
    print(f"  R2:   {rf_metrics['r2']:.4f}")

    print("\n=== Text Baseline ===")
    print("\n[TF-IDF + Ridge]")
    print(f"  RMSE: {tfidf_metrics['rmse']:.4f}")
    print(f"  MAE:  {tfidf_metrics['mae']:.4f}")
    print(f"  R2:   {tfidf_metrics['r2']:.4f}")

    print("\nTop 15 Feature Importances (Random Forest):")
    print(rf_importance_df.head(15).to_string(index=False))

    _save_results(city, all_metrics, rf_importance_df, ols_coef_df)
    print(f"\nResults saved to {RESULTS_DIR / city}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tabular baseline models.")
    parser.add_argument("--city", default=DEFAULT_CITY, help="City name")
    args = parser.parse_args()
    run_baseline(city=args.city)
