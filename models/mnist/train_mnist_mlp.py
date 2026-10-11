"""Train the MNIST MLP on the existing 54,000/6,000 split.

The existing loader still materializes the official test arrays.
This script does not read those fields and does not score them.

A checkpoint path and a history path are required. The selected weights at
models/mnist_mlp.pt are not a default destination. An existing checkpoint or
history file is left in place unless --overwrite is present. The five-epoch
baseline lives at models/experiments/mnist_mlp_5epochs.pt.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import torchvision

import data
import model

SEED = 0
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DEVICE = torch.device("cpu")


def seed_training(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def epoch_metrics(
    network: model.MnistMlp,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    if train:
        network.train()
    else:
        network.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            if train:
                assert optimizer is not None
                optimizer.zero_grad()
            logits = network(images)
            loss = criterion(logits, labels)
            if train:
                assert optimizer is not None
                loss.backward()
                optimizer.step()
            batch = int(labels.shape[0])
            total_loss += float(loss.item()) * batch
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += batch
    if total == 0:
        raise RuntimeError("a data loader produced no samples")
    return total_loss / total, correct / total


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a new MNIST MLP checkpoint. "
            "Pass an explicit checkpoint path. Existing files are kept unless --overwrite is set."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Destination for the new state_dict. There is no default path.",
    )
    parser.add_argument(
        "--history",
        required=True,
        type=Path,
        help="Destination for the new training history JSON. There is no default path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a checkpoint or history file that already exists at the given path.",
    )
    return parser.parse_args(argv)


def require_writable(path: Path, *, overwrite: bool, label: str) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(
            f"{label} already exists at {path}. Pass --overwrite to replace it. "
            "models/mnist_mlp.pt is the selected checkpoint and "
            "models/experiments/mnist_mlp_5epochs.pt is the preserved five-epoch baseline."
        )


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint
    history_path = args.history
    require_writable(checkpoint_path, overwrite=args.overwrite, label="checkpoint")
    require_writable(history_path, overwrite=args.overwrite, label="history")

    started = time.perf_counter()
    seed_training(SEED)
    prepared = data.load_mnist(download=False)
    network = model.MnistMlp().to(DEVICE)
    counted = model.parameter_count(network)
    if counted != model.PARAMETER_COUNT:
        raise RuntimeError(f"parameter count is {counted}, expected {model.PARAMETER_COUNT}")

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(prepared.train_images),
            torch.from_numpy(prepared.train_labels),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )
    validation_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(prepared.validation_images),
            torch.from_numpy(prepared.validation_labels),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    loop_started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        training_loss, _training_accuracy = epoch_metrics(
            network, train_loader, criterion, train=True, optimizer=optimizer
        )
        network.eval()
        validation_loss, validation_accuracy = epoch_metrics(
            network, validation_loader, criterion, train=False
        )
        if not all(np.isfinite(value) for value in (training_loss, validation_loss, validation_accuracy)):
            raise RuntimeError(f"epoch {epoch} produced a non-finite metric")
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in network.state_dict().items()}
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "best_checkpoint_epoch": best_epoch,
            }
        )
        print(
            f"epoch {epoch} training_loss {training_loss:.8f} "
            f"validation_loss {validation_loss:.8f} "
            f"validation_accuracy {validation_accuracy:.8f} "
            f"best_checkpoint_epoch {best_epoch}",
            flush=True,
        )
    loop_seconds = time.perf_counter() - loop_started
    if best_state is None:
        raise RuntimeError("no checkpoint was selected")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_path)
    record = {
        "seed": SEED,
        "device": str(DEVICE),
        "epochs": EPOCHS,
        "final_epoch": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "CrossEntropyLoss",
        "data_loader_workers": 0,
        "architecture": "Linear(784, 128), ReLU, Linear(128, 64), ReLU, Linear(64, 10)",
        "parameter_count": counted,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "training_seconds": loop_seconds,
        "preprocessing": "uint8 pixels divided by 255, row-major float32[784], no mean or std",
        "split": (
            "NumPy PCG64 seed 0 permutation of the official 60000 training images; "
            "first 54000 train and last 6000 validate. The official test set is not scored."
        ),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "checkpoint": checkpoint_path.as_posix(),
        "epochs_log": history,
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - started
    print(f"best_epoch {best_epoch} best_validation_accuracy {best_accuracy:.8f}", flush=True)
    print(f"training_seconds {loop_seconds:.3f} total_seconds {elapsed:.3f}", flush=True)
    print(f"checkpoint {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
