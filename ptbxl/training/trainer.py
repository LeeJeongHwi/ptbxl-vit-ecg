"""Training utilities: seeding, dataloaders and the training loop shared by all experiments."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ptbxl.config import PipelineConfig
from ptbxl.model.registry import build_model, count_parameters
from ptbxl.model.vit import AsymmetricLoss


def set_seed(seed: int) -> None:
    """Fix all RNG seeds for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataloader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
) -> DataLoader:
    """Build a DataLoader from numpy arrays, converting to float32 tensors."""
    return DataLoader(
        TensorDataset(torch.FloatTensor(x), torch.FloatTensor(y)),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )


class TrainingResult(NamedTuple):
    best_loss: float
    best_epoch: int
    train_losses: list[float]
    val_losses: list[float]
    checkpoint: Path


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    checkpoint: Path,
    desc: str = "",
) -> TrainingResult:
    """Train a model with AdamW, cosine annealing and the asymmetric loss.

    The state dict with the lowest validation loss is written to `checkpoint`
    (overwritten whenever the validation loss improves), and training stops once
    the validation loss has not improved for `patience` consecutive epochs.
    The weights left in `model` are those of the last epoch run, not the best ones.
    """
    criterion = AsymmetricLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_epoch = -1
    no_improve = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    tag = f"[{desc}] " if desc else ""

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        bar = tqdm(train_loader, desc=f"{tag}Train Epoch[{epoch}/{epochs}]")
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            bar.set_postfix(loss=f"{loss.item():.3f}")
        train_loss = train_loss_sum / len(train_loader)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss_sum += criterion(model(x), y).item()
        val_loss = val_loss_sum / len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  Epoch {epoch}: train={train_loss:.4f}  val={val_loss:.4f}", end="")
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), checkpoint)
            print("  -> best updated")
        else:
            no_improve += 1
            print()
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        scheduler.step()

    return TrainingResult(best_loss, best_epoch, train_losses, val_losses, checkpoint)


def train_model(
    config: PipelineConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[nn.Module, list[float], list[float]]:
    """Build the configured model and train it with `train_loop`.

    Writes best_<model>.pth (lowest validation loss) and final_<model>.pth (last epoch)
    to config.model_dir, together with a small training_time.txt.
    """
    model = build_model(config.model_name).to(device)
    n_params = count_parameters(model)

    sep = "=" * 95
    print(sep)
    print(f"  Model: {config.model_name}  |  Trainable params: {n_params:,}")
    print(f"  Batch {config.batch_size} | LR {config.learning_rate} | Epochs {config.epochs} | Patience {config.early_stopping_patience}")
    print(sep)

    training_start = time.time()
    result = train_loop(
        model, train_loader, val_loader, device,
        epochs=config.epochs,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        patience=config.early_stopping_patience,
        checkpoint=config.model_dir / f"best_{config.model_name}.pth",
        desc=config.model_name,
    )
    elapsed = time.time() - training_start

    torch.save(model.state_dict(), config.model_dir / f"final_{config.model_name}.pth")
    print(f"\nBest validation loss {result.best_loss:.4f} at epoch {result.best_epoch}")
    print(f"Total training time: {elapsed/60:.2f} min ({elapsed:.1f} s)")
    (config.model_dir / "training_time.txt").write_text(
        f"model: {config.model_name}\n"
        f"epochs_completed: {len(result.train_losses)}\n"
        f"best_epoch: {result.best_epoch}\n"
        f"total_seconds: {elapsed:.2f}\n"
        f"minutes: {elapsed/60:.2f}\n",
        encoding="utf-8",
    )

    return model, result.train_losses, result.val_losses
