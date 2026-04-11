"""Centralized configuration for the Airbnb price prediction project.

All constants, hyperparameter defaults, column definitions and file paths
are gathered here so every module imports from a single source of truth.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
RESULTS_DIR: Path = DATA_DIR / "results"

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
INSIDE_AIRBNB_URL: str = "https://insideairbnb.com/get-the-data/"
DEFAULT_CITY: str = "london"

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------
RAW_PRICE_COLUMN: str = "price"
TARGET_COLUMN: str = "log_price"
DESCRIPTION_COLUMN: str = "description"
NEIGHBORHOOD_OVERVIEW_COLUMN: str = "neighborhood_overview"

NUMERIC_COLUMNS: list[str] = [
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
]

SPATIAL_COLUMNS: list[str] = ["latitude", "longitude"]

CATEGORICAL_COLUMNS: list[str] = ["room_type"]

TEXT_COLUMNS: list[str] = [DESCRIPTION_COLUMN, NEIGHBORHOOD_OVERVIEW_COLUMN]

# ---------------------------------------------------------------------------
# Spatial feature engineering
# ---------------------------------------------------------------------------
# Approximate centre of London (Trafalgar Square)
LONDON_CENTRE_LAT: float = 51.5074
LONDON_CENTRE_LON: float = -0.1278

# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------
MIN_PRICE: float = 10.0
MAX_PRICE: float = 2000.0

# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
TRAIN_RATIO: float = 0.70
VALIDATION_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# ---------------------------------------------------------------------------
# NLP / Vocabulary
# ---------------------------------------------------------------------------
PAD_TOKEN: str = "<PAD>"
UNK_TOKEN: str = "<UNK>"
PAD_INDEX: int = 0
UNK_INDEX: int = 1
MAX_VOCAB_SIZE: int = 20_000
MAX_SEQUENCE_LENGTH: int = 120

# ---------------------------------------------------------------------------
# Training hyperparameters (shared defaults)
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
DROPOUT: float = 0.2
EMBEDDING_DIM: int = 100
NUM_RNN_LAYERS: int = 1
RNN_TYPE: str = "lstm"
GRAD_CLIP_MAX_NORM: float = 1.0
LR_SCHEDULER_FACTOR: float = 0.5
LR_SCHEDULER_PATIENCE: int = 2

# ---------------------------------------------------------------------------
# Text model defaults
# ---------------------------------------------------------------------------
TEXT_MODEL_HIDDEN_DIM: int = 128
TEXT_MODEL_MAX_EPOCHS: int = 15
TEXT_MODEL_PATIENCE: int = 3

# ---------------------------------------------------------------------------
# Hybrid model defaults
# ---------------------------------------------------------------------------
HYBRID_TEXT_HIDDEN_DIM: int = 64
HYBRID_TABULAR_HIDDEN_DIM: int = 32
HYBRID_FUSION_HIDDEN_DIM: int = 64
HYBRID_MAX_EPOCHS: int = 15
HYBRID_PATIENCE: int = 3

# ---------------------------------------------------------------------------
# Transformer model defaults
# ---------------------------------------------------------------------------
TRANSFORMER_HEADS: int = 4
TRANSFORMER_LAYERS: int = 2
TRANSFORMER_DROPOUT: float = 0.1
TRANSFORMER_HIDDEN_DIM: int = 64
