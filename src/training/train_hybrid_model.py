"""Train the hybrid (text + tabular) model for Airbnb price prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# -- path setup ---------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import (  # noqa: E402
    BATCH_SIZE,
    DEFAULT_CITY,
    DROPOUT,
    EMBEDDING_DIM,
    HYBRID_FUSION_HIDDEN_DIM,
    HYBRID_MAX_EPOCHS,
    HYBRID_PATIENCE,
    HYBRID_TABULAR_HIDDEN_DIM,
    HYBRID_TEXT_HIDDEN_DIM,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    NUM_RNN_LAYERS,
    RANDOM_SEED,
    RESULTS_DIR,
    RNN_TYPE,
    TARGET_COLUMN,
)
from features.tabular import (  # noqa: E402
    HybridDataset,
    collate_hybrid_batch,
    preprocess_tabular_features,
)
from features.text import Vocabulary  # noqa: E402
from models.hybrid_predictor import HybridPredictor  # noqa: E402
from training.utils import (  # noqa: E402
    TrainingHistory,
    combine_text_columns,
    detect_device,
    evaluate_metrics,
    load_splits,
    set_seed,
)
from visualization.plots import (  # noqa: E402
    save_comparison_chart,
    save_training_curves,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_vocabulary(city: str) -> Vocabulary:
    """Load the vocabulary saved by the text model training.

    Args:
        city: City identifier.

    Returns:
        A :class:`~features.text.Vocabulary` instance.

    Raises:
        FileNotFoundError: If the vocabulary file does not exist.
    """
    vocab_path = RESULTS_DIR / city / "text_model" / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocabulary not found: {vocab_path}. " "Run train_text_model.py first."
        )
    with vocab_path.open("r", encoding="utf-8") as f:
        return Vocabulary.from_dict(json.load(f))


def _make_hybrid_loader(
    texts: list[str],
    tabular_features: np.ndarray,
    targets: np.ndarray,
    vocabulary: Vocabulary,
    max_sequence_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a DataLoader for the hybrid model.

    Args:
        texts: Raw text strings.
        tabular_features: Pre-processed numeric feature matrix.
        targets: Regression target values.
        vocabulary: Fitted vocabulary.
        max_sequence_length: Token truncation limit.
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle each epoch.

    Returns:
        A configured :class:`DataLoader`.
    """
    dataset = HybridDataset(
        texts=texts,
        tabular_features=tabular_features,
        targets=targets,
        vocabulary=vocabulary,
        max_sequence_length=max_sequence_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_hybrid_batch(
            batch, pad_index=vocabulary.pad_index
        ),
    )


# ---------------------------------------------------------------------------
# Epoch runner & prediction
# ---------------------------------------------------------------------------


def _run_epoch(
    model: HybridPredictor,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    label: str,
) -> float:
    """Run one training or validation epoch.

    Args:
        model: The hybrid prediction model.
        data_loader: Batched data source.
        loss_fn: Loss function (e.g. MSELoss).
        optimizer: Optimiser (``None`` for evaluation mode).
        device: Target compute device.
        label: Progress-bar description.

    Returns:
        Mean loss over all batches.
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    num_batches = 0

    for input_ids, lengths, tabular, targets in tqdm(
        data_loader, desc=label, leave=False
    ):
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(is_training):
            predictions = model(
                input_ids=input_ids, lengths=lengths, tabular_features=tabular
            )
            loss = loss_fn(predictions, targets)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(1, num_batches)


def _predict(
    model: HybridPredictor,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions for a full DataLoader.

    Args:
        model: Trained hybrid model.
        data_loader: Evaluation data source.
        device: Compute device.

    Returns:
        Tuple of ``(y_true, y_pred)`` numpy arrays.
    """
    model.eval()
    all_targets: list[float] = []
    all_predictions: list[float] = []

    with torch.no_grad():
        for input_ids, lengths, tabular, targets in data_loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            tabular = tabular.to(device)
            outputs = model(
                input_ids=input_ids, lengths=lengths, tabular_features=tabular
            )
            all_predictions.extend(outputs.detach().cpu().tolist())
            all_targets.extend(targets.tolist())

    return np.asarray(all_targets), np.asarray(all_predictions)


# ---------------------------------------------------------------------------
# Main training procedure
# ---------------------------------------------------------------------------


def train_hybrid_model(
    city: str,
    use_neighborhood_overview: bool,
    max_sequence_length: int,
    embedding_dim: int,
    text_hidden_dim: int,
    tabular_hidden_dim: int,
    fusion_hidden_dim: int,
    num_layers: int,
    dropout: float,
    rnn_type: str,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
    device_preference: str,
) -> None:
    """Train, evaluate and save the hybrid model.

    Args:
        city: City identifier.
        use_neighborhood_overview: Concatenate overview text to description.
        max_sequence_length: Token truncation limit.
        embedding_dim: Embedding vector size.
        text_hidden_dim: RNN hidden units for the text encoder.
        tabular_hidden_dim: MLP hidden units for the tabular encoder.
        fusion_hidden_dim: Hidden units in the fusion layer.
        num_layers: Stacked RNN layers.
        dropout: Dropout probability.
        rnn_type: ``"lstm"`` or ``"gru"``.
        learning_rate: Adam learning rate.
        batch_size: Mini-batch size.
        max_epochs: Maximum training epochs.
        patience: Early-stopping patience (epochs).
        seed: Random seed.
        device_preference: ``"auto"``, ``"cuda"``, ``"mps"`` or ``"cpu"``.
    """
    set_seed(seed)

    # -- data -----------------------------------------------------------------
    train_df, val_df, test_df = load_splits(city)
    vocabulary = _load_vocabulary(city)

    train_texts = combine_text_columns(train_df, use_neighborhood_overview).tolist()
    val_texts = combine_text_columns(val_df, use_neighborhood_overview).tolist()
    test_texts = combine_text_columns(test_df, use_neighborhood_overview).tolist()

    x_train, x_val, x_test, scaler, tabular_columns = preprocess_tabular_features(
        train_df, val_df, test_df
    )

    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    y_val = val_df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    train_loader = _make_hybrid_loader(
        train_texts,
        x_train,
        y_train,
        vocabulary,
        max_sequence_length,
        batch_size,
        shuffle=True,
    )
    val_loader = _make_hybrid_loader(
        val_texts,
        x_val,
        y_val,
        vocabulary,
        max_sequence_length,
        batch_size,
        shuffle=False,
    )
    test_loader = _make_hybrid_loader(
        test_texts,
        x_test,
        y_test,
        vocabulary,
        max_sequence_length,
        batch_size,
        shuffle=False,
    )

    # -- model ----------------------------------------------------------------
    device = detect_device(device_preference)
    model = HybridPredictor(
        vocab_size=len(vocabulary),
        pad_index=vocabulary.pad_index,
        tabular_input_dim=x_train.shape[1],
        embedding_dim=embedding_dim,
        text_hidden_dim=text_hidden_dim,
        tabular_hidden_dim=tabular_hidden_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        rnn_type=rnn_type,
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # -- output paths ---------------------------------------------------------
    results_dir = RESULTS_DIR / city / "hybrid"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = results_dir / "best_model.pth"
    metrics_path = results_dir / "test_metrics.json"
    curve_path = results_dir / "training_curves.png"
    scaler_path = results_dir / "tabular_scaler_stats.json"

    # -- training loop --------------------------------------------------------
    best_val_loss = float("inf")
    patience_counter = 0
    history = TrainingHistory()

    print(f"Device: {device}")
    print(f"Tabular features: {tabular_columns}")

    for epoch in range(1, max_epochs + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            label=f"Epoch {epoch}/{max_epochs} [train]",
        )
        val_loss = _run_epoch(
            model,
            val_loader,
            loss_fn,
            None,
            device,
            label=f"Epoch {epoch}/{max_epochs} [val]",
        )

        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)

        print(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} "
            f"| Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "vocab_size": len(vocabulary),
                        "pad_index": vocabulary.pad_index,
                        "tabular_input_dim": x_train.shape[1],
                        "embedding_dim": embedding_dim,
                        "text_hidden_dim": text_hidden_dim,
                        "tabular_hidden_dim": tabular_hidden_dim,
                        "fusion_hidden_dim": fusion_hidden_dim,
                        "num_layers": num_layers,
                        "dropout": dropout,
                        "rnn_type": rnn_type,
                    },
                },
                model_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered (patience={patience}).")
                break

    # -- save curves & scaler -------------------------------------------------
    save_training_curves(
        history.train_losses,
        history.val_losses,
        curve_path,
        title="Hybrid Model Training Curves",
    )

    scaler_mean = np.asarray(scaler.mean_ if scaler.mean_ is not None else [])
    scaler_scale = np.asarray(scaler.scale_ if scaler.scale_ is not None else [])
    scaler_payload = {
        "columns": tabular_columns,
        "mean": scaler_mean.tolist(),
        "scale": scaler_scale.tolist(),
    }
    scaler_path.write_text(json.dumps(scaler_payload, indent=2), encoding="utf-8")

    # -- test evaluation ------------------------------------------------------
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred = _predict(model, test_loader, device)
    metrics = evaluate_metrics(y_true, y_pred)

    payload = {
        "city": city,
        "target": TARGET_COLUMN,
        "text_source": (
            "description + neighborhood_overview"
            if use_neighborhood_overview
            else "description"
        ),
        "model": "hybrid",
        "best_val_loss": best_val_loss,
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # -- comparison -----------------------------------------------------------
    save_comparison_chart(city=city, hybrid_r2=metrics["r2"])

    baseline_path = RESULTS_DIR / city / "baseline_metrics.json"
    text_model_path = RESULTS_DIR / city / "text_model" / "test_metrics.json"

    baseline_r2 = None
    text_r2 = None
    if baseline_path.exists():
        baseline_r2 = json.loads(baseline_path.read_text(encoding="utf-8"))["models"][
            "random_forest"
        ]["r2"]
    if text_model_path.exists():
        text_r2 = json.loads(text_model_path.read_text(encoding="utf-8"))["metrics"][
            "r2"
        ]

    print("\n=== Hybrid Model (test set) ===")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")

    if baseline_r2 is not None and text_r2 is not None:
        print("\nR² comparison (test):")
        print(f"  Baseline RF:  {baseline_r2:.4f}")
        print(f"  Text Model:   {text_r2:.4f}")
        print(f"  Hybrid:       {metrics['r2']:.4f}")

    print(f"\nArtifacts saved to {results_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the hybrid model training script."""
    parser = argparse.ArgumentParser(
        description="Train hybrid text+tabular model for Airbnb price prediction."
    )
    parser.add_argument("--city", default=DEFAULT_CITY)
    parser.add_argument("--use-neighborhood-overview", action="store_true")
    parser.add_argument("--max-sequence-length", type=int, default=MAX_SEQUENCE_LENGTH)
    parser.add_argument("--embedding-dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--text-hidden-dim", type=int, default=HYBRID_TEXT_HIDDEN_DIM)
    parser.add_argument(
        "--tabular-hidden-dim", type=int, default=HYBRID_TABULAR_HIDDEN_DIM
    )
    parser.add_argument(
        "--fusion-hidden-dim", type=int, default=HYBRID_FUSION_HIDDEN_DIM
    )
    parser.add_argument("--num-layers", type=int, default=NUM_RNN_LAYERS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--rnn-type", choices=["lstm", "gru"], default=RNN_TYPE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=HYBRID_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=HYBRID_PATIENCE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="auto: CUDA > MPS > CPU",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train_hybrid_model(
        city=args.city,
        use_neighborhood_overview=args.use_neighborhood_overview,
        max_sequence_length=args.max_sequence_length,
        embedding_dim=args.embedding_dim,
        text_hidden_dim=args.text_hidden_dim,
        tabular_hidden_dim=args.tabular_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        rnn_type=args.rnn_type,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        device_preference=args.device,
    )
