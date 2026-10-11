"""Load official MNIST and make the fixed train, validation, and test arrays.

Pixels stay in the IDX order. Each image is uint8 grayscale, divided by 255,
and flattened in row-major order to float32[784]. The official test set is
never mixed into training or validation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OFFICIAL_TRAIN_SIZE = 60_000
OFFICIAL_TEST_SIZE = 10_000
TRAIN_SIZE = 54_000
VALIDATION_SIZE = 6_000
SPLIT_SEED = 0
IMAGE_EDGE = 28
FLAT_SIZE = IMAGE_EDGE * IMAGE_EDGE


def cache_dir() -> Path:
    return Path(__file__).resolve().parent / ".cache"


def preprocess_image(image) -> np.ndarray:
    """Return one float32 vector in [0, 1] with length 784."""
    array = _hw_uint_array(image)
    values = array.astype(np.float32) / np.float32(255.0)
    return np.ascontiguousarray(values.reshape(FLAT_SIZE))


def preprocess_images(images: np.ndarray) -> np.ndarray:
    """Return float32[N, 784] from uint8 images shaped [N, 28, 28]."""
    if images.ndim != 3 or images.shape[1:] != (IMAGE_EDGE, IMAGE_EDGE):
        raise ValueError(f"MNIST batch must be [N, 28, 28], got shape {images.shape}")
    if not np.issubdtype(images.dtype, np.integer):
        raise ValueError(f"MNIST pixels must be integers, got {images.dtype}")
    values = images.astype(np.float32) / np.float32(255.0)
    return np.ascontiguousarray(values.reshape(images.shape[0], FLAT_SIZE))


def train_validation_indices(
    count: int = OFFICIAL_TRAIN_SIZE,
    train_size: int = TRAIN_SIZE,
    seed: int = SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the official training indices. The test set is not an argument."""
    if train_size < 0 or train_size > count:
        raise ValueError(f"train_size {train_size} is outside 0..{count}")
    generator = np.random.Generator(np.random.PCG64(seed))
    permutation = generator.permutation(count)
    return permutation[:train_size], permutation[train_size:]


@dataclass(frozen=True)
class PreparedMnist:
    train_images: np.ndarray
    train_labels: np.ndarray
    validation_images: np.ndarray
    validation_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    cache_dir: Path


def load_mnist(root: Path | None = None, *, download: bool = True) -> PreparedMnist:
    """Download official MNIST if needed and apply the seed-0 split."""
    dataset_root = cache_dir() if root is None else Path(root)
    train_images, train_labels, test_images, test_labels = _official_arrays(
        dataset_root, download=download
    )
    if train_images.shape[0] != OFFICIAL_TRAIN_SIZE:
        raise ValueError(f"official training set has {train_images.shape[0]} images")
    if test_images.shape[0] != OFFICIAL_TEST_SIZE:
        raise ValueError(f"official test set has {test_images.shape[0]} images")

    flat_train = preprocess_images(train_images)
    flat_test = preprocess_images(test_images)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    train_index, validation_index = train_validation_indices()
    return PreparedMnist(
        train_images=flat_train[train_index],
        train_labels=train_labels[train_index],
        validation_images=flat_train[validation_index],
        validation_labels=train_labels[validation_index],
        test_images=flat_test,
        test_labels=test_labels,
        train_indices=train_index,
        validation_indices=validation_index,
        cache_dir=dataset_root,
    )


def archive_md5(root: Path | None = None) -> list[tuple[str, str, str]]:
    """Return filename, torchvision MD5, and the MD5 of the cached gzip."""
    from torchvision.datasets import MNIST

    dataset_root = cache_dir() if root is None else Path(root)
    raw_dir = dataset_root / "MNIST" / "raw"
    checked: list[tuple[str, str, str]] = []
    for filename, expected in MNIST.resources:
        path = raw_dir / filename
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        checked.append((filename, expected, digest))
    return checked


def _hw_uint_array(image) -> np.ndarray:
    if hasattr(image, "detach"):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (IMAGE_EDGE, IMAGE_EDGE):
        raise ValueError(f"MNIST image must be 28x28, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"MNIST pixels must be integers, got {array.dtype}")
    return array


def _official_arrays(root: Path, *, download: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from torchvision.datasets import MNIST

    root.mkdir(parents=True, exist_ok=True)
    training = MNIST(root=str(root), train=True, download=download)
    testing = MNIST(root=str(root), train=False, download=download)
    return (
        training.data.detach().cpu().numpy(),
        training.targets.detach().cpu().numpy(),
        testing.data.detach().cpu().numpy(),
        testing.targets.detach().cpu().numpy(),
    )
