"""Shape and checkpoint checks for the MNIST MLP. These do not train it."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

import model


def seeded_model() -> model.MnistMlp:
    torch.manual_seed(0)
    return model.MnistMlp()


class ModelShapeTest(unittest.TestCase):
    def test_forward_shape_dtype_and_finite_logits(self) -> None:
        network = seeded_model()
        images = torch.zeros(3, model.INPUT_SIZE)
        logits = network(images)
        self.assertEqual(tuple(images.shape), (3, 784))
        self.assertEqual(tuple(logits.shape), (3, 10))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_parameter_count(self) -> None:
        self.assertEqual(model.parameter_count(model.MnistMlp()), 109_386)
        self.assertEqual(model.parameter_count(model.MnistMlp()), model.PARAMETER_COUNT)

    def test_cross_entropy_accepts_the_logits(self) -> None:
        network = seeded_model()
        images = torch.rand(4, 784)
        labels = torch.tensor([0, 1, 2, 9], dtype=torch.int64)
        loss = nn.CrossEntropyLoss()(network(images), labels)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_seed_zero_initialization_matches(self) -> None:
        first = seeded_model()
        second = seeded_model()
        for left, right in zip(first.state_dict().values(), second.state_dict().values(), strict=True):
            self.assertTrue(torch.equal(left, right))

    def test_state_dict_round_trip(self) -> None:
        original = seeded_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mnist_mlp.pt"
            torch.save(original.state_dict(), path)
            restored = model.MnistMlp()
            restored.load_state_dict(torch.load(path, weights_only=True))
        images = torch.rand(2, 784)
        self.assertTrue(torch.equal(original(images), restored(images)))


if __name__ == "__main__":
    unittest.main()
