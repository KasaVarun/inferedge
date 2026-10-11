"""Checks for checkpoint loading and evaluation arithmetic.

The official 10,000-image test set is not scored here.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

import evaluate_mnist_mlp as evaluation
import model


class CheckpointLoadTest(unittest.TestCase):
    def test_checkpoint_loads_with_the_expected_parameter_count(self) -> None:
        network = evaluation.load_checkpoint(evaluation.CHECKPOINT_PATH)
        self.assertEqual(model.parameter_count(network), model.PARAMETER_COUNT)
        self.assertFalse(network.training)

    def test_synthetic_forward_returns_ten_finite_logits(self) -> None:
        network = evaluation.load_checkpoint(evaluation.CHECKPOINT_PATH)
        logits = evaluation.predict_logits(network, torch.zeros(4, model.INPUT_SIZE))
        self.assertEqual(tuple(logits.shape), (4, 10))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(logits).all().item())
        predicted = logits.argmax(dim=1)
        self.assertTrue(torch.all((predicted >= 0) & (predicted <= 9)).item())

    def test_fixed_checkpoint_matches_on_a_second_forward(self) -> None:
        network = evaluation.load_checkpoint(evaluation.CHECKPOINT_PATH)
        images = torch.arange(2 * model.INPUT_SIZE, dtype=torch.float32).reshape(2, model.INPUT_SIZE)
        images = images / float(model.INPUT_SIZE)
        first = evaluation.predict_logits(network, images)
        second = evaluation.predict_logits(network, images)
        self.assertTrue(torch.equal(first, second))


class SummaryTest(unittest.TestCase):
    def test_confusion_matrix_totals_and_per_class_accuracy(self) -> None:
        true_labels = np.arange(10, dtype=np.int64)
        predicted = true_labels.copy()
        predicted[3] = 4
        summary = evaluation.classification_summary(true_labels, predicted)
        matrix = summary["confusion_matrix"]
        self.assertEqual(len(matrix), 10)
        self.assertTrue(all(len(row) == 10 for row in matrix))
        self.assertEqual(matrix[3][4], 1)
        self.assertEqual(matrix[3][3], 0)
        self.assertEqual(matrix[4][4], 1)
        self.assertEqual(sum(sum(row) for row in matrix), 10)
        self.assertEqual(sum(matrix[index][index] for index in range(10)), 9)
        self.assertEqual(summary["sample_count"], 10)
        self.assertEqual(summary["correct_predictions"], 9)
        self.assertEqual(summary["incorrect_predictions"], 1)
        self.assertEqual(summary["correct_predictions"] + summary["incorrect_predictions"], 10)
        self.assertEqual(summary["accuracy"], 9 / 10)
        self.assertEqual(summary["per_class_sample_counts"], [1] * 10)
        self.assertEqual(summary["per_class_accuracy"][3], 0.0)
        self.assertEqual(summary["per_class_accuracy"][4], 1.0)
        for index, count in enumerate(summary["per_class_sample_counts"]):
            self.assertEqual(sum(matrix[index]), count)
            self.assertEqual(summary["per_class_accuracy"][index], matrix[index][index] / count)

    def test_example_selection_is_stable_and_keeps_a_mistake(self) -> None:
        true_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=np.int64)
        predicted = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1], dtype=np.int64)
        first = evaluation.select_prediction_examples(true_labels, predicted, incorrect_limit=3)
        second = evaluation.select_prediction_examples(true_labels, predicted, incorrect_limit=3)
        self.assertEqual(first, second)
        self.assertEqual([row["test_index"] for row in first], list(range(11)))
        self.assertTrue(all(row["correct"] for row in first[:10]))
        self.assertEqual(first[10]["true_label"], 0)
        self.assertEqual(first[10]["predicted_label"], 1)
        self.assertFalse(first[10]["correct"])


if __name__ == "__main__":
    unittest.main()
