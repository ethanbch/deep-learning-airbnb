"""Train a basic Transformer encoder model for Airbnb price prediction."""

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

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import (  # noqa: E402
    BATCH_SIZE,
    DEFAULT_CITY,
    EMBEDDING_DIM,
    LEARNING_RATE,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    MAX_SEQUENCE_LENGTH,
    MAX_VOCAB_SIZE,
    RANDOM_SEED,
    RESULTS_DIR,
    TARGET_COLUMN,
    TEXT_MODEL_MAX_EPOCHS,
    TEXT_MODEL_PATIENCE,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_HEADS,
    TRANSFORMER_HIDDEN_DIM,
    TRANSFORMER_LAYERS,
    WEIGHT_DECAY,
)
from features.text import (  # noqa: E402
    TextRegressionDataset,
    Vocabulary,
    build_vocabulary,
    collate_text_batch,
)
from models.transformer_predictor import TransformerPricePredictor  # noqa: E402
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
NUM_DATALOADER_WORKERS: int = 4


def _make_loader(
    texts: list[str],
    targets: np.ndarray,
    vocabulary: Vocabulary,
    max_sequence_length: int,
    batch_size: int,
    shuffle: bool,
    use_cuda: bool = False,
) -> DataLoader:
    dataset = TextRegressionDataset(
        texts=texts,
        targets=targets.tolist(),
        vocabulary=vocabulary,
        max_sequence_length=max_sequence_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_DATALOADER_WORKERS,
        pin_memory=use_cuda,
        collate_fn=partial(collate_text_batch, pad_index=vocabulary.pad_index),
    )


def _run_epoch(
    model: TransformerPricePredictor,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    label: str,
) -> float:
    """Run one epoch. Returns mean loss over all **samples**."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_samples = 0

    for input_ids, lengths, targets in tqdm(data_loader, desc=label, leave=False):
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)
        batch_size = targets.size(0)

        with torch.set_grad_enabled(is_training):
            predictions = model(input_ids, lengths)
            loss = loss_fn(predictions, targets)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(1, total_samples)


def _predict(
    model: TransformerPricePredictor,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[float] = []
    y_pred: list[float] = []

    with torch.no_grad():
        for input_ids, lengths, targets in data_loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            outputs = model(input_ids, lengths)
            y_pred.extend(outputs.detach().cpu().tolist())
            y_true.extend(targets.tolist())

    return np.asarray(y_true), np.asarray(y_pred)


def train_transformer_model(
    city: str,
    use_neighborhood_overview: bool,
    max_vocab_size: int,
    max_sequence_length: int,
    embedding_dim: int,
    transformer_hidden_dim: int,
    transformer_heads: int,
    transformer_layers: int,
    transformer_dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    lr_scheduler_factor: float,
    lr_scheduler_patience: int,
    seed: int,
    device_preference: str,
    resume_from_checkpoint: bool,
) -> None:
    set_seed(seed)

    train_df, val_df, test_df = load_splits(city)
    train_texts = combine_text_columns(train_df, use_neighborhood_overview).tolist()
    val_texts = combine_text_columns(val_df, use_neighborhood_overview).tolist()
    test_texts = combine_text_columns(test_df, use_neighborhood_overview).tolist()

    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    y_val = val_df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    vocabulary = build_vocabulary(train_texts, max_vocab_size=max_vocab_size)

    device = detect_device(device_preference)
    use_cuda = device.type == "cuda"

    train_loader = _make_loader(
        train_texts, y_train, vocabulary, max_sequence_length, batch_size,
        shuffle=True, use_cuda=use_cuda,
    )
    val_loader = _make_loader(
        val_texts, y_val, vocabulary, max_sequence_length, batch_size,
        shuffle=False, use_cuda=use_cuda,
    )
    test_loader = _make_loader(
        test_texts, y_test, vocabulary, max_sequence_length, batch_size,
        shuffle=False, use_cuda=use_cuda,
    )

    model = TransformerPricePredictor(
        vocab_size=len(vocabulary),
        pad_index=vocabulary.pad_index,
        embedding_dim=embedding_dim,
        hidden_dim=transformer_hidden_dim,
        num_heads=transformer_heads,
        num_layers=transformer_layers,
        dropout=transformer_dropout,
        max_sequence_length=max_sequence_length,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_scheduler_factor,
        patience=lr_scheduler_patience, verbose=True,
    )

    results_dir = RESULTS_DIR / city / "transformer_model"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = results_dir / "best_model.pth"
    vocab_path = results_dir / "vocab.json"
    metrics_path = results_dir / "test_metrics.json"
    epoch_metrics_path = results_dir / "epoch_metrics.jsonl"
    curve_path = results_dir / "training_curves.png"

    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 1
    history = TrainingHistory()

    print(f"Device: {device}")
    print(f"Vocabulary size: {len(vocabulary):,}")

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
        "embedding_dim": embedding_dim,
        "hidden_dim": transformer_hidden_dim,
        "num_heads": transformer_heads,
        "num_layers": transformer_layers,
        "dropout": transformer_dropout,
        "max_sequence_length": max_sequence_length,
    }

    for epoch in range(start_epoch, max_epochs + 1):
        should_stop = False
        train_loss = _run_epoch(
            model, train_loader, loss_fn, optimizer, device,
            label=f"Epoch {epoch}/{max_epochs} [train]",
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
            title="Transformer Model Training Curves",
        )

        if should_stop:
            print(f"Early stopping triggered (patience={patience}).")
            break

    with vocab_path.open("w", encoding="utf-8") as output_file:
        json.dump(vocabulary.to_dict(), output_file, ensure_ascii=False)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred = _predict(model, test_loader, device)
    clear_device_cache(device, aggressive=True)
    test_metrics = evaluate_metrics(y_true, y_pred)

    report = {
        "city": city,
        "target": TARGET_COLUMN,
        "text_source": (
            "description + neighborhood_overview"
            if use_neighborhood_overview
            else "description"
        ),
        "model": "transformer",
        "best_val_loss": best_val_loss,
        "metrics": test_metrics,
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    save_comparison_chart(city=city)

    print("\n=== Transformer Model (test set) ===")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  R2:   {test_metrics['r2']:.4f}")
    print(f"\nArtifacts saved to {results_dir}/")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train basic Transformer encoder for Airbnb price prediction."
    )
    parser.add_argument("--city", default=DEFAULT_CITY)
    parser.add_argument("--use-neighborhood-overview", action="store_true")
    parser.add_argument("--max-vocab-size", type=int, default=MAX_VOCAB_SIZE)
    parser.add_argument("--max-sequence-length", type=int, default=MAX_SEQUENCE_LENGTH)
    parser.add_argument("--embedding-dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument(
        "--transformer-hidden-dim", type=int, default=TRANSFORMER_HIDDEN_DIM
    )
    parser.add_argument("--transformer-heads", type=int, default=TRANSFORMER_HEADS)
    parser.add_argument("--transformer-layers", type=int, default=TRANSFORMER_LAYERS)
    parser.add_argument(
        "--transformer-dropout", type=float, default=TRANSFORMER_DROPOUT
    )
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=TEXT_MODEL_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=TEXT_MODEL_PATIENCE)
    parser.add_argument("--lr-scheduler-factor", type=float, default=LR_SCHEDULER_FACTOR)
    parser.add_argument("--lr-scheduler-patience", type=int, default=LR_SCHEDULER_PATIENCE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
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
    train_transformer_model(
        city=args.city,
        use_neighborhood_overview=args.use_neighborhood_overview,
        max_vocab_size=args.max_vocab_size,
        max_sequence_length=args.max_sequence_length,
        embedding_dim=args.embedding_dim,
        transformer_hidden_dim=args.transformer_hidden_dim,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        transformer_dropout=args.transformer_dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_patience=args.lr_scheduler_patience,
        seed=args.seed,
        device_preference=args.device,
        resume_from_checkpoint=args.resume,
    )
