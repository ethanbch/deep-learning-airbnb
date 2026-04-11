"""Train the hybrid (text + tabular) model for Airbnb price prediction."""

from __future__ import annotations

import argparse
from functools import partial
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
    GRAD_CLIP_MAX_NORM,
    HYBRID_FUSION_HIDDEN_DIM,
    HYBRID_MAX_EPOCHS,
    HYBRID_PATIENCE,
    HYBRID_TABULAR_HIDDEN_DIM,
    HYBRID_TEXT_HIDDEN_DIM,
    LEARNING_RATE,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    MAX_SEQUENCE_LENGTH,
    MAX_VOCAB_SIZE,
    NUM_RNN_LAYERS,
    RANDOM_SEED,
    RESULTS_DIR,
    RNN_TYPE,
    TARGET_COLUMN,
    WEIGHT_DECAY,
)
from features.tabular import (  # noqa: E402
    HybridDataset,
    collate_hybrid_batch,
    preprocess_tabular_features,
)
from features.text import Vocabulary, build_vocabulary  # noqa: E402
from models.hybrid_predictor import HybridPredictor  # noqa: E402
from training.utils import (  # noqa: E402
    TrainingHistory,
    append_epoch_metrics,
    build_checkpoint_payload,
    clear_device_cache,
    combine_text_columns,
    detect_device,
    evaluate_metrics,
    initialize_epoch_metrics_log,
    load_splits,
    maybe_resume_checkpoint,
    set_seed,
)
from visualization.plots import (  # noqa: E402
    save_comparison_chart,
    save_training_curves,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_DATALOADER_WORKERS: int = 4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hybrid_loader(
    texts: list[str],
    tabular_features: np.ndarray,
    targets: np.ndarray,
    vocabulary: Vocabulary,
    max_sequence_length: int,
    batch_size: int,
    shuffle: bool,
    use_cuda: bool = False,
) -> DataLoader:
    """Create a DataLoader for the hybrid model."""
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
        num_workers=NUM_DATALOADER_WORKERS,
        pin_memory=use_cuda,
        collate_fn=partial(collate_hybrid_batch, pad_index=vocabulary.pad_index),
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
    grad_clip_max_norm: float | None = None,
) -> float:
    """Run one training or validation epoch.

    Returns:
        Mean loss over all **samples** (weighted by batch size).
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_samples = 0

    for input_ids, lengths, tabular, targets in tqdm(
        data_loader, desc=label, leave=False
    ):
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device)
        batch_size = targets.size(0)

        with torch.set_grad_enabled(is_training):
            predictions = model(
                input_ids=input_ids, lengths=lengths, tabular_features=tabular
            )
            loss = loss_fn(predictions, targets)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip_max_norm
                    )
                optimizer.step()

        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(1, total_samples)


def _predict(
    model: HybridPredictor,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions for a full DataLoader."""
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
    max_vocab_size: int,
    max_sequence_length: int,
    embedding_dim: int,
    text_hidden_dim: int,
    tabular_hidden_dim: int,
    fusion_hidden_dim: int,
    num_layers: int,
    dropout: float,
    rnn_type: str,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    grad_clip_max_norm: float,
    lr_scheduler_factor: float,
    lr_scheduler_patience: int,
    seed: int,
    device_preference: str,
    resume_from_checkpoint: bool,
) -> None:
    """Train, evaluate and save the hybrid model."""
    set_seed(seed)

    # -- data -----------------------------------------------------------------
    train_df, val_df, test_df = load_splits(city)

    train_texts = combine_text_columns(train_df, use_neighborhood_overview).tolist()
    val_texts = combine_text_columns(val_df, use_neighborhood_overview).tolist()
    test_texts = combine_text_columns(test_df, use_neighborhood_overview).tolist()

    # Build vocabulary on training texts (independent of text-only model)
    vocabulary = build_vocabulary(texts=train_texts, max_vocab_size=max_vocab_size)

    x_train, x_val, x_test, scaler, tabular_columns = preprocess_tabular_features(
        train_df, val_df, test_df
    )

    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    y_val = val_df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    # -- model ----------------------------------------------------------------
    device = detect_device(device_preference)
    use_cuda = device.type == "cuda"

    train_loader = _make_hybrid_loader(
        train_texts, x_train, y_train, vocabulary, max_sequence_length,
        batch_size, shuffle=True, use_cuda=use_cuda,
    )
    val_loader = _make_hybrid_loader(
        val_texts, x_val, y_val, vocabulary, max_sequence_length,
        batch_size, shuffle=False, use_cuda=use_cuda,
    )
    test_loader = _make_hybrid_loader(
        test_texts, x_test, y_test, vocabulary, max_sequence_length,
        batch_size, shuffle=False, use_cuda=use_cuda,
    )

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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_scheduler_factor,
        patience=lr_scheduler_patience, verbose=True,
    )

    # -- output paths ---------------------------------------------------------
    results_dir = RESULTS_DIR / city / "hybrid"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = results_dir / "best_model.pth"
    vocab_path = results_dir / "vocab.json"
    metrics_path = results_dir / "test_metrics.json"
    epoch_metrics_path = results_dir / "epoch_metrics.jsonl"
    curve_path = results_dir / "training_curves.png"
    scaler_path = results_dir / "tabular_scaler_stats.json"

    # -- training loop --------------------------------------------------------
    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 1
    history = TrainingHistory()

    print(f"Device: {device}")
    print(f"Vocabulary size: {len(vocabulary):,}")
    print(f"Tabular features ({x_train.shape[1]}): {tabular_columns}")

    checkpoint_state = maybe_resume_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_path=model_path,
        device=device,
        resume_from_checkpoint=resume_from_checkpoint,
        initial_best_val_loss=best_val_loss,
    )
    best_val_loss = checkpoint_state.best_val_loss
    patience_counter = checkpoint_state.patience_counter
    start_epoch = checkpoint_state.start_epoch
    initialize_epoch_metrics_log(
        epoch_metrics_path,
        resume_from_existing_log=checkpoint_state.resumed,
    )

    model_config = {
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
    }

    for epoch in range(start_epoch, max_epochs + 1):
        should_stop = False
        train_loss = _run_epoch(
            model, train_loader, loss_fn, optimizer, device,
            label=f"Epoch {epoch}/{max_epochs} [train]",
            grad_clip_max_norm=grad_clip_max_norm,
        )
        val_loss = _run_epoch(
            model, val_loader, loss_fn, None, device,
            label=f"Epoch {epoch}/{max_epochs} [val]",
        )

        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)
        clear_device_cache(device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} "
            f"| Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                build_checkpoint_payload(
                    model=model,
                    model_config=model_config,
                    optimizer=optimizer,
                    best_val_loss=best_val_loss,
                    epoch=epoch,
                    patience_counter=patience_counter,
                ),
                model_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                should_stop = True

        append_epoch_metrics(
            metrics_log_path=epoch_metrics_path,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            patience_counter=patience_counter,
        )
        save_training_curves(
            history.train_losses,
            history.val_losses,
            curve_path,
            title="Hybrid Model Training Curves",
        )

        if should_stop:
            print(f"Early stopping triggered (patience={patience}).")
            break

    # -- save vocabulary ------------------------------------------------------
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocabulary.to_dict(), f, ensure_ascii=False)

    # -- save scaler ----------------------------------------------------------
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
    clear_device_cache(device, aggressive=True)
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
    print(f"  R2:   {metrics['r2']:.4f}")

    if baseline_r2 is not None and text_r2 is not None:
        print("\nR2 comparison (test):")
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
    parser.add_argument("--max-vocab-size", type=int, default=MAX_VOCAB_SIZE)
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
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=HYBRID_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=HYBRID_PATIENCE)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP_MAX_NORM)
    parser.add_argument("--lr-scheduler-factor", type=float, default=LR_SCHEDULER_FACTOR)
    parser.add_argument("--lr-scheduler-patience", type=int, default=LR_SCHEDULER_PATIENCE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="auto: CUDA > MPS > CPU",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing checkpoint when available.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train_hybrid_model(
        city=args.city,
        use_neighborhood_overview=args.use_neighborhood_overview,
        max_vocab_size=args.max_vocab_size,
        max_sequence_length=args.max_sequence_length,
        embedding_dim=args.embedding_dim,
        text_hidden_dim=args.text_hidden_dim,
        tabular_hidden_dim=args.tabular_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        rnn_type=args.rnn_type,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        grad_clip_max_norm=args.grad_clip,
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_patience=args.lr_scheduler_patience,
        seed=args.seed,
        device_preference=args.device,
        resume_from_checkpoint=args.resume,
    )
