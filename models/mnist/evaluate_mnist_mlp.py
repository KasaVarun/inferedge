"""Score the promoted MNIST checkpoint once on the official test set.

The checkpoint is loaded as a state_dict. This script does not train,
does not change weights, and does not export ONNX.

Example rows use a fixed rule. In official test-index order, keep the first
image of each true label 0 through 9. Then keep up to three later images
whose prediction disagrees with the true label and whose index was not
already kept. An index is recorded once.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

import data
import model

DEVICE = torch.device("cpu")
CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "mnist_mlp.pt"
HISTORY_PATH = Path(__file__).resolve().parent / "training_history.json"
REPORT_PATH = Path(__file__).resolve().parent / "evaluation_report.json"
SELECTED_SHA256 = "b53a41676adfa020e3808617539f7966776fdd20bd7a247f34656e230459f22a"
INCORRECT_EXAMPLE_LIMIT = 3


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path: Path) -> model.MnistMlp:
    """Load state_dict tensors and return an eval-mode MLP on CPU."""
    network = model.MnistMlp().to(DEVICE)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    network.load_state_dict(state)
    network.eval()
    counted = model.parameter_count(network)
    if counted != model.PARAMETER_COUNT:
        raise RuntimeError(f"parameter count is {counted}, expected {model.PARAMETER_COUNT}")
    return network


def predict_logits(network: model.MnistMlp, images: torch.Tensor) -> torch.Tensor:
    """Return float32 logits shaped [N, 10]."""
    if images.ndim != 2 or images.shape[1] != model.INPUT_SIZE:
        raise ValueError(f"images must be [N, {model.INPUT_SIZE}], got {tuple(images.shape)}")
    network.eval()
    with torch.inference_mode():
        logits = network(images.to(device=DEVICE, dtype=torch.float32))
    if tuple(logits.shape) != (images.shape[0], model.CLASS_COUNT) or logits.dtype != torch.float32:
        raise RuntimeError(f"unexpected logits {tuple(logits.shape)} {logits.dtype}")
    if not torch.isfinite(logits).all():
        raise RuntimeError("logits are not finite")
    return logits


def sample_weighted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean of per-example cross-entropy. Each example has equal weight."""
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0] or labels.shape[0] == 0:
        raise ValueError("labels must be rank-1 and match the logits")
    count = int(labels.shape[0])
    with torch.inference_mode():
        total = nn.CrossEntropyLoss(reduction="sum")(logits, labels.to(device=DEVICE, dtype=torch.int64))
    value = float(total.item()) / count
    if not np.isfinite(value):
        raise RuntimeError("test loss is not finite")
    return value


def classification_summary(true_labels: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    """Build counts from a confusion matrix whose rows are true labels."""
    true_labels = np.asarray(true_labels, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    if true_labels.ndim != 1 or true_labels.shape != predicted.shape or true_labels.size == 0:
        raise ValueError("labels and predictions must be the same non-empty rank-1 shape")
    if (
        int(true_labels.min()) < 0
        or int(true_labels.max()) >= model.CLASS_COUNT
        or int(predicted.min()) < 0
        or int(predicted.max()) >= model.CLASS_COUNT
    ):
        raise ValueError(f"labels and predictions must be in 0..{model.CLASS_COUNT - 1}")

    matrix = np.zeros((model.CLASS_COUNT, model.CLASS_COUNT), dtype=np.int64)
    np.add.at(matrix, (true_labels, predicted), 1)
    counts = matrix.sum(axis=1)
    correct_per_class = np.diag(matrix)
    total = int(matrix.sum())
    correct = int(correct_per_class.sum())
    incorrect = total - correct
    if total != int(true_labels.size) or correct + incorrect != total or int(counts.sum()) != total:
        raise RuntimeError("confusion matrix does not match the sample count")

    per_class_accuracy: list[float | None] = []
    for count, hits in zip(counts.tolist(), correct_per_class.tolist(), strict=True):
        if count == 0:
            per_class_accuracy.append(None)
        else:
            per_class_accuracy.append(hits / count)
    return {
        "sample_count": total,
        "correct_predictions": correct,
        "incorrect_predictions": incorrect,
        "accuracy": correct / total,
        "per_class_sample_counts": [int(count) for count in counts.tolist()],
        "per_class_correct": [int(hits) for hits in correct_per_class.tolist()],
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": [[int(value) for value in row] for row in matrix.tolist()],
    }


def select_prediction_examples(
    true_labels: np.ndarray,
    predicted: np.ndarray,
    *,
    incorrect_limit: int = INCORRECT_EXAMPLE_LIMIT,
) -> list[dict[str, int | bool]]:
    """Pick the first index of each true digit, then the first unseen mistakes."""
    true_labels = np.asarray(true_labels, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    if true_labels.shape != predicted.shape:
        raise ValueError("labels and predictions must have the same shape")

    selected: list[int] = []
    seen: set[int] = set()
    for digit in range(model.CLASS_COUNT):
        matches = np.flatnonzero(true_labels == digit)
        if matches.size == 0:
            raise ValueError(f"no example with true label {digit}")
        index = int(matches[0])
        selected.append(index)
        seen.add(index)

    kept = 0
    if incorrect_limit > 0:
        for index in np.flatnonzero(true_labels != predicted).tolist():
            if index in seen:
                continue
            selected.append(int(index))
            seen.add(int(index))
            kept += 1
            if kept == incorrect_limit:
                break

    examples: list[dict[str, int | bool]] = []
    for index in selected:
        true_label = int(true_labels[index])
        predicted_label = int(predicted[index])
        examples.append(
            {
                "test_index": index,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": true_label == predicted_label,
            }
        )
    return examples


def main() -> None:
    digest = file_sha256(CHECKPOINT_PATH)
    if digest != SELECTED_SHA256:
        raise RuntimeError(f"checkpoint sha256 is {digest}, expected {SELECTED_SHA256}")

    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    if history.get("best_epoch") != 10 or history.get("epochs") != 10:
        raise RuntimeError("training history is not the selected ten-epoch run")
    if history.get("checkpoint_sha256") != SELECTED_SHA256:
        raise RuntimeError("training history does not name the promoted checkpoint")

    prepared = data.load_mnist(download=False)
    images_np = np.ascontiguousarray(prepared.test_images)
    labels_np = np.ascontiguousarray(prepared.test_labels)
    if images_np.shape != (data.OFFICIAL_TEST_SIZE, model.INPUT_SIZE) or labels_np.shape != (
        data.OFFICIAL_TEST_SIZE,
    ):
        raise RuntimeError(
            f"official test arrays have shapes {images_np.shape} and {labels_np.shape}"
        )

    network = load_checkpoint(CHECKPOINT_PATH)
    images = torch.from_numpy(images_np)
    labels = torch.from_numpy(labels_np)
    logits = predict_logits(network, images)
    loss = sample_weighted_cross_entropy(logits, labels)
    predicted = logits.argmax(dim=1).cpu().numpy().astype(np.int64, copy=False)
    summary = classification_summary(labels_np, predicted)
    if any(count == 0 for count in summary["per_class_sample_counts"]):
        raise RuntimeError("a test class has no samples")
    if any(value is None for value in summary["per_class_accuracy"]):
        raise RuntimeError("per-class accuracy is undefined")

    matrix = summary["confusion_matrix"]
    diagonal = sum(matrix[index][index] for index in range(model.CLASS_COUNT))
    if (
        sum(sum(row) for row in matrix) != data.OFFICIAL_TEST_SIZE
        or diagonal != summary["correct_predictions"]
        or summary["correct_predictions"] + summary["incorrect_predictions"] != data.OFFICIAL_TEST_SIZE
    ):
        raise RuntimeError("evaluation totals are inconsistent")

    examples = select_prediction_examples(labels_np, predicted)
    report = {
        "architecture": "Linear(784, 128), ReLU, Linear(128, 64), ReLU, Linear(64, 10)",
        "parameter_count": model.parameter_count(network),
        "selected_checkpoint_epoch": history["best_epoch"],
        "checkpoint": "models/mnist_mlp.pt",
        "checkpoint_sha256": digest,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "dataset_split": (
            "Official MNIST test set, 10000 images in IDX order. "
            "These images were not used for training or checkpoint selection."
        ),
        "preprocessing": "uint8 pixels divided by 255, row-major float32[784], no mean or std",
        "logits": "10 float32 values. The predicted class is argmax. Softmax is not applied.",
        "test_sample_count": summary["sample_count"],
        "correct_predictions": summary["correct_predictions"],
        "incorrect_predictions": summary["incorrect_predictions"],
        "test_accuracy": summary["accuracy"],
        "test_loss": loss,
        "per_class_accuracy": summary["per_class_accuracy"],
        "per_class_sample_counts": summary["per_class_sample_counts"],
        "confusion_matrix_orientation": "rows are true labels, columns are predicted labels",
        "confusion_matrix": matrix,
        "evaluation_device": str(DEVICE),
        "example_selection": (
            "Official test index order. First image of each true label 0 through 9, "
            "then up to 3 incorrect predictions whose indices were not already selected."
        ),
        "examples": examples,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"test_sample_count {summary['sample_count']}", flush=True)
    print(f"correct_predictions {summary['correct_predictions']}", flush=True)
    print(f"incorrect_predictions {summary['incorrect_predictions']}", flush=True)
    print(f"test_accuracy {summary['accuracy']:.8f}", flush=True)
    print(f"test_loss {loss:.8f}", flush=True)
    print(f"checkpoint_sha256 {digest}", flush=True)
    print(f"report {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
