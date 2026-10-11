"""Checks for MNIST preprocessing and the fixed train/validation split."""

from __future__ import annotations

import unittest

import numpy as np

import data


class PreprocessTest(unittest.TestCase):
    def test_zero_image_is_a_zero_vector(self) -> None:
        image = np.zeros((28, 28), dtype=np.uint8)
        flat = data.preprocess_image(image)
        self.assertEqual(flat.shape, (784,))
        self.assertEqual(flat.dtype, np.float32)
        self.assertTrue(np.all(flat == 0.0))

    def test_byte_255_scales_to_one_at_its_index(self) -> None:
        image = np.zeros((28, 28), dtype=np.uint8)
        image[2, 3] = 255
        flat = data.preprocess_image(image)
        self.assertEqual(flat[2 * 28 + 3], np.float32(1.0))
        self.assertEqual(int(np.count_nonzero(flat)), 1)

    def test_mid_gray_uses_division_by_255(self) -> None:
        image = np.full((28, 28), 128, dtype=np.uint8)
        flat = data.preprocess_image(image)
        expected = np.float32(128.0) / np.float32(255.0)
        self.assertTrue(np.all(flat == expected))
        self.assertGreaterEqual(float(flat.min()), 0.0)
        self.assertLessEqual(float(flat.max()), 1.0)

    def test_rejects_a_non_square_image(self) -> None:
        with self.assertRaises(ValueError):
            data.preprocess_image(np.zeros((27, 28), dtype=np.uint8))

    def test_rejects_float_pixels(self) -> None:
        with self.assertRaises(ValueError):
            data.preprocess_image(np.zeros((28, 28), dtype=np.float32))


class SplitTest(unittest.TestCase):
    def test_sizes_and_repeat_calls(self) -> None:
        first_train, first_validation = data.train_validation_indices()
        second_train, second_validation = data.train_validation_indices()
        self.assertEqual(first_train.shape, (54_000,))
        self.assertEqual(first_validation.shape, (6_000,))
        self.assertTrue(np.array_equal(first_train, second_train))
        self.assertTrue(np.array_equal(first_validation, second_validation))

    def test_indices_cover_the_official_training_set_once(self) -> None:
        train_index, validation_index = data.train_validation_indices()
        combined = np.concatenate([train_index, validation_index])
        self.assertEqual(set(train_index.tolist()).intersection(validation_index.tolist()), set())
        self.assertEqual(combined.shape[0], 60_000)
        self.assertTrue(np.array_equal(np.sort(combined), np.arange(60_000)))


class OfficialMnistTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prepared = data.load_mnist()

    def test_split_counts_and_storage(self) -> None:
        prepared = self.prepared
        self.assertEqual(prepared.train_images.shape, (54_000, 784))
        self.assertEqual(prepared.validation_images.shape, (6_000, 784))
        self.assertEqual(prepared.test_images.shape, (10_000, 784))
        self.assertEqual(prepared.train_images.dtype, np.float32)
        self.assertEqual(prepared.validation_images.dtype, np.float32)
        self.assertEqual(prepared.test_images.dtype, np.float32)
        self.assertEqual(prepared.train_labels.dtype, np.int64)
        self.assertEqual(prepared.test_labels.dtype, np.int64)

    def test_pixels_and_labels_stay_in_range(self) -> None:
        for images, labels in (
            (self.prepared.train_images, self.prepared.train_labels),
            (self.prepared.validation_images, self.prepared.validation_labels),
            (self.prepared.test_images, self.prepared.test_labels),
        ):
            self.assertGreaterEqual(float(images.min()), 0.0)
            self.assertLessEqual(float(images.max()), 1.0)
            self.assertGreaterEqual(int(labels.min()), 0)
            self.assertLessEqual(int(labels.max()), 9)

    def test_real_sample_matches_raw_bytes_divided_by_255(self) -> None:
        from torchvision.datasets import MNIST

        training = MNIST(root=str(self.prepared.cache_dir), train=True, download=False)
        raw = training.data[0].detach().cpu().numpy()
        expected = raw.astype(np.float32) / np.float32(255.0)
        self.assertTrue(np.array_equal(data.preprocess_image(raw), expected.reshape(784)))
        self.assertIn(int(training.targets[0]), range(10))

    def test_split_selects_official_training_labels(self) -> None:
        from torchvision.datasets import MNIST

        training = MNIST(root=str(self.prepared.cache_dir), train=True, download=False)
        official = training.targets.detach().cpu().numpy().astype(np.int64)
        self.assertTrue(np.array_equal(official[self.prepared.train_indices], self.prepared.train_labels))
        self.assertTrue(
            np.array_equal(official[self.prepared.validation_indices], self.prepared.validation_labels)
        )
        self.assertEqual(
            set(self.prepared.train_indices.tolist()).intersection(self.prepared.validation_indices.tolist()),
            set(),
        )

    def test_archives_match_torchvision_checksums(self) -> None:
        for filename, expected, actual in data.archive_md5(self.prepared.cache_dir):
            self.assertEqual(actual, expected, filename)

    def test_second_load_is_the_same_split(self) -> None:
        again = data.load_mnist(download=False)
        self.assertTrue(np.array_equal(again.train_indices, self.prepared.train_indices))
        self.assertTrue(np.array_equal(again.validation_indices, self.prepared.validation_indices))
        self.assertTrue(np.array_equal(again.test_labels, self.prepared.test_labels))


if __name__ == "__main__":
    unittest.main()
