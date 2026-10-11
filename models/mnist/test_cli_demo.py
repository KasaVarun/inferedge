"""Checks for the native CLI demonstration adapter.

Successful inference is the real inferedge process. These tests do not score
the official 10,000-image test set.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import demo_inferedge_cli as demo

LIVE_FIXTURES = ("test_00003", "test_00000", "test_00151", "synthetic")


class FixtureAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = demo.load_manifest()

    def test_selects_the_default_digit_zero_fixture(self) -> None:
        case = demo.select_case(self.manifest, "test_00003")
        self.assertEqual(case["true_class"], 0)
        self.assertEqual(case["expected_predicted_class"], 0)

    def test_unknown_fixture_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            demo.select_case(self.manifest, "not_a_fixture")

    def test_input_length_checksum_and_finite_values(self) -> None:
        case = demo.select_case(self.manifest, "test_00003")
        path = demo.fixture_dir() / case["input"]
        self.assertEqual(path.stat().st_size, demo.INPUT_COUNT * 4)
        self.assertEqual(demo.sha256(path), case["input_sha256"])
        values = demo.load_input(case)
        self.assertEqual(len(values), demo.INPUT_COUNT)
        self.assertTrue(all(value == value for value in values))

    def test_image_argument_has_one_token_per_float(self) -> None:
        argument = demo.image_argument([0.25] * demo.INPUT_COUNT)
        self.assertTrue(argument.startswith("image="))
        tokens = argument.removeprefix("image=").split(",")
        self.assertEqual(len(tokens), demo.INPUT_COUNT)
        self.assertTrue(all(len(token) <= demo.TOKEN_LIMIT for token in tokens))

    def test_parser_reads_ten_logits_and_the_largest_index(self) -> None:
        stdout = "Model: example\nBackend: ONNX Runtime CPU\n\nOutput logits:\n[1, -2, 4, 0, 0, 0, 0, 0, 0, 3]\n"
        logits = demo.parse_logits(stdout)
        self.assertEqual(len(logits), 10)
        self.assertEqual(demo.predicted_class(logits), 2)

    def test_tolerance_accepts_a_tiny_difference(self) -> None:
        reference = [1.0, -2.0]
        actual = [1.0 + 1e-6, -2.0]
        self.assertTrue(demo.within_tolerance(actual, reference))
        self.assertFalse(demo.within_tolerance([1.0, 10.0], reference))


class NativeCliTest(unittest.TestCase):
    def test_nonzero_exit_is_reported(self) -> None:
        executable = demo.executable_path()
        completed = demo.launch(executable, Path("/no/such/mnist-model.onnx"), "image=0")
        self.assertNotEqual(completed.returncode, 0)

    def test_selected_fixtures_match_the_manifest(self) -> None:
        for name in LIVE_FIXTURES:
            result = demo.demonstrate(name)
            self.assertEqual(result["exit_status"], 0, name)
            self.assertEqual(len(result["logits"]), 10, name)
            self.assertTrue(result["prediction_matches"], result)
            self.assertTrue(result["within_tolerance"], result)
            self.assertEqual(result["predicted_class"], result["expected_predicted_class"])


if __name__ == "__main__":
    unittest.main()
