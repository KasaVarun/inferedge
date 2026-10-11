"""Fixture and parity checks for the 14 recorded MNIST inputs.

These tests do not score the official 10,000-image test set.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

import model
import verify_onnx_parity as parity

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "mnist"


def load_manifest() -> dict:
    return json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))


class FixtureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_manifest()
        self.cases = self.manifest["cases"]

    def test_shapes_dtype_and_element_counts(self) -> None:
        self.assertEqual(self.manifest["input_elements"], 784)
        self.assertEqual(self.manifest["logit_elements"], 10)
        self.assertEqual(len(self.cases), 14)
        for case in self.cases:
            image = parity.read_float32(FIXTURE_DIR / case["input"], 784)
            pytorch = parity.read_float32(FIXTURE_DIR / case["pytorch_logits"], 10)
            exported = parity.read_float32(FIXTURE_DIR / case["onnx_logits"], 10)
            for array, count in ((image, 784), (pytorch, 10), (exported, 10)):
                self.assertEqual(array.shape, (count,))
                self.assertEqual(array.dtype, np.dtype("<f4"))
                self.assertTrue(np.issubdtype(array.dtype, np.floating))

    def test_manifest_matches_files_and_predictions(self) -> None:
        names = [case["name"] for case in self.cases]
        self.assertEqual(len(names), len(set(names)))
        plan = parity.case_plan()
        self.assertEqual(
            [(case["name"], case["test_index"], case["true_class"]) for case in self.cases],
            [(case["name"], case["test_index"], case["true_class"]) for case in plan],
        )
        for case in self.cases:
            for key in ("input", "pytorch_logits", "onnx_logits"):
                path = FIXTURE_DIR / case[key]
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(digest, case[f"{key}_sha256"])
            pytorch = parity.read_float32(FIXTURE_DIR / case["pytorch_logits"], 10)
            exported = parity.read_float32(FIXTURE_DIR / case["onnx_logits"], 10)
            self.assertEqual(int(np.argmax(pytorch)), case["expected_predicted_class"])
            self.assertEqual(int(np.argmax(exported)), case["expected_predicted_class"])
            np.testing.assert_allclose(pytorch, exported, rtol=parity.RTOL, atol=parity.ATOL)

    def test_synthetic_fixture_matches_the_formula(self) -> None:
        first = parity.synthetic_image()
        second = parity.synthetic_image()
        self.assertTrue(np.array_equal(first, second))
        payload = parity.float32_bytes(first)
        self.assertEqual(payload, parity.float32_bytes(second))
        synthetic = next(case for case in self.cases if case["name"] == "synthetic")
        stored = (FIXTURE_DIR / synthetic["input"]).read_bytes()
        self.assertEqual(stored, payload)
        self.assertEqual(hashlib.sha256(stored).hexdigest(), synthetic["input_sha256"])
        self.assertEqual(first.shape, (model.INPUT_SIZE,))


class LiveParityTest(unittest.TestCase):
    def test_pytorch_and_onnx_agree_on_the_fixture_inputs(self) -> None:
        parity.require_artifacts()
        network = parity.load_network()
        session = parity.load_session()
        manifest = load_manifest()
        for case in manifest["cases"]:
            image = parity.read_float32(FIXTURE_DIR / case["input"], 784)
            pytorch = parity.pytorch_logits(network, image)
            exported = parity.onnx_logits(session, image)
            np.testing.assert_allclose(pytorch, exported, rtol=parity.RTOL, atol=parity.ATOL)
            predicted = int(np.argmax(pytorch))
            self.assertEqual(predicted, int(np.argmax(exported)))
            self.assertEqual(predicted, case["expected_predicted_class"])


if __name__ == "__main__":
    unittest.main()
