"""Compare the frozen MNIST checkpoint with its ONNX Runtime session.

The 13 real images are the indices recorded in evaluation_report.json.
This script does not reselect them, does not train, does not re-export, and
does not score the rest of the official test set.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

import data
import model

CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "mnist_mlp.pt"
ONNX_PATH = Path(__file__).resolve().parents[1] / "mnist_mlp.onnx"
REPORT_PATH = Path(__file__).resolve().parent / "parity_report.json"
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "mnist"
CHECKPOINT_SHA256 = "b53a41676adfa020e3808617539f7966776fdd20bd7a247f34656e230459f22a"
ONNX_SHA256 = "3473a4b2673fe4353c2b3a04a7d432717a4dfcdc42fc217565c99743542d8cce"
RTOL = 1e-4
ATOL = 1e-5

# Official test indices from models/mnist/evaluation_report.json.
CLASS_EXAMPLES = (
    (3, 0),
    (2, 1),
    (1, 2),
    (18, 3),
    (4, 4),
    (8, 5),
    (11, 6),
    (0, 7),
    (61, 8),
    (7, 9),
)
MISCLASSIFIED_EXAMPLES = (
    (151, 9),
    (217, 6),
    (241, 9),
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def float32_bytes(values: np.ndarray) -> bytes:
    """Little-endian float32 payload with no header."""
    array = np.ascontiguousarray(values, dtype="<f4")
    return array.tobytes()


def read_float32(path: Path, count: int) -> np.ndarray:
    payload = path.read_bytes()
    if len(payload) != count * 4:
        raise ValueError(f"{path} has {len(payload)} bytes, expected {count * 4}")
    return np.frombuffer(payload, dtype="<f4").copy()


def synthetic_image() -> np.ndarray:
    """input[k] = float32(k / 784) for k in 0..783."""
    values = np.empty(model.INPUT_SIZE, dtype=np.float32)
    for index in range(model.INPUT_SIZE):
        values[index] = np.float32(index / model.INPUT_SIZE)
    return values


def case_plan() -> list[dict[str, object]]:
    """Return the 14 cases in the required order. Predictions are not consulted."""
    cases: list[dict[str, object]] = []
    for test_index, true_class in CLASS_EXAMPLES:
        cases.append(
            {
                "name": f"test_{test_index:05d}",
                "kind": "official_test_class_example",
                "test_index": test_index,
                "true_class": true_class,
            }
        )
    for test_index, true_class in MISCLASSIFIED_EXAMPLES:
        cases.append(
            {
                "name": f"test_{test_index:05d}",
                "kind": "official_test_misclassification",
                "test_index": test_index,
                "true_class": true_class,
            }
        )
    cases.append(
        {
            "name": "synthetic",
            "kind": "synthetic",
            "test_index": None,
            "true_class": None,
        }
    )
    return cases


def require_artifacts() -> None:
    checkpoint = file_sha256(CHECKPOINT_PATH)
    exported = file_sha256(ONNX_PATH)
    if checkpoint != CHECKPOINT_SHA256:
        raise SystemExit(f"checkpoint sha256 is {checkpoint}, expected {CHECKPOINT_SHA256}")
    if exported != ONNX_SHA256:
        raise SystemExit(f"onnx sha256 is {exported}, expected {ONNX_SHA256}")


def load_network() -> model.MnistMlp:
    network = model.MnistMlp().to(torch.device("cpu"))
    state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    network.load_state_dict(state)
    network.eval()
    return network


def load_session() -> ort.InferenceSession:
    session = ort.InferenceSession(ONNX_PATH.read_bytes(), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise SystemExit(f"providers are {session.get_providers()}")
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise SystemExit("the ONNX session must have one input and one output")
    image = inputs[0]
    logits = outputs[0]
    if image.name != "image" or image.type != "tensor(float)" or list(image.shape) != [model.INPUT_SIZE]:
        raise SystemExit(f"ONNX input is {image.name} {image.type} {image.shape}")
    if logits.name != "logits" or logits.type != "tensor(float)" or list(logits.shape) != [model.CLASS_COUNT]:
        raise SystemExit(f"ONNX output is {logits.name} {logits.type} {logits.shape}")
    return session


def pytorch_logits(network: model.MnistMlp, image: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
    network.eval()
    with torch.inference_mode():
        logits = network(tensor)
    array = np.ascontiguousarray(logits.detach().cpu().numpy(), dtype=np.float32)
    _require_logits(array, "PyTorch")
    return array


def onnx_logits(session: ort.InferenceSession, image: np.ndarray) -> np.ndarray:
    feed = np.ascontiguousarray(image, dtype=np.float32)
    result = session.run(None, {"image": feed})
    if len(result) != 1:
        raise SystemExit(f"ONNX Runtime returned {len(result)} outputs")
    array = np.ascontiguousarray(result[0], dtype=np.float32)
    _require_logits(array, "ONNX Runtime")
    return array


def _require_logits(logits: np.ndarray, source: str) -> None:
    if logits.shape != (model.CLASS_COUNT,) or logits.dtype != np.float32 or not np.isfinite(logits).all():
        raise SystemExit(f"{source} logits are shape {logits.shape} dtype {logits.dtype}")


def compare_logits(pytorch: np.ndarray, onnx: np.ndarray, case: dict[str, object]) -> dict[str, object]:
    difference = np.abs(pytorch.astype(np.float64) - onnx.astype(np.float64))
    pytorch_class = int(np.argmax(pytorch))
    onnx_class = int(np.argmax(onnx))
    try:
        np.testing.assert_allclose(pytorch, onnx, rtol=RTOL, atol=ATOL)
    except AssertionError as exc:
        _fail(case, pytorch, onnx, difference, f"logits differ beyond rtol={RTOL} atol={ATOL}: {exc}")
    if pytorch_class != onnx_class:
        _fail(case, pytorch, onnx, difference, "argmax predictions differ")
    return {
        "name": case["name"],
        "kind": case["kind"],
        "test_index": case["test_index"],
        "true_class": case["true_class"],
        "pytorch_predicted_class": pytorch_class,
        "onnx_predicted_class": onnx_class,
        "predictions_agree": True,
        "maximum_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
    }


def _fail(
    case: dict[str, object],
    pytorch: np.ndarray,
    onnx: np.ndarray,
    difference: np.ndarray,
    reason: str,
) -> None:
    print(f"parity failed for {case['name']}: {reason}", flush=True)
    print(f"test_index {case['test_index']} true_class {case['true_class']}", flush=True)
    print(f"pytorch_logits {pytorch.tolist()}", flush=True)
    print(f"onnx_logits {onnx.tolist()}", flush=True)
    print(f"absolute_difference {difference.tolist()}", flush=True)
    raise SystemExit(1)


def input_for_case(case: dict[str, object], test_images: np.ndarray, test_labels: np.ndarray) -> np.ndarray:
    if case["kind"] == "synthetic":
        return synthetic_image()
    index = int(case["test_index"])
    true_class = int(case["true_class"])
    if int(test_labels[index]) != true_class:
        raise SystemExit(f"test index {index} has label {int(test_labels[index])}, expected {true_class}")
    image = np.ascontiguousarray(test_images[index], dtype=np.float32)
    if image.shape != (model.INPUT_SIZE,) or image.dtype != np.float32:
        raise SystemExit(f"test index {index} has shape {image.shape} dtype {image.dtype}")
    return image


def write_fixtures(rows: list[dict[str, object]]) -> None:
    input_dir = FIXTURE_DIR / "inputs"
    pytorch_dir = FIXTURE_DIR / "pytorch"
    onnx_dir = FIXTURE_DIR / "onnx"
    for directory in (input_dir, pytorch_dir, onnx_dir):
        directory.mkdir(parents=True, exist_ok=True)
    manifest_cases = []
    for row in rows:
        name = str(row["name"])
        input_path = input_dir / f"{name}.f32"
        pytorch_path = pytorch_dir / f"{name}.f32"
        onnx_path = onnx_dir / f"{name}.f32"
        input_path.write_bytes(float32_bytes(row["image"]))
        pytorch_path.write_bytes(float32_bytes(row["pytorch_logits"]))
        onnx_path.write_bytes(float32_bytes(row["onnx_logits"]))
        manifest_cases.append(
            {
                "name": name,
                "kind": row["kind"],
                "test_index": row["test_index"],
                "true_class": row["true_class"],
                "expected_predicted_class": row["pytorch_predicted_class"],
                "input": f"inputs/{name}.f32",
                "pytorch_logits": f"pytorch/{name}.f32",
                "onnx_logits": f"onnx/{name}.f32",
                "input_sha256": file_sha256(input_path),
                "pytorch_logits_sha256": file_sha256(pytorch_path),
                "onnx_logits_sha256": file_sha256(onnx_path),
            }
        )
    manifest = {
        "format": "little-endian float32 arrays with no header",
        "input_elements": model.INPUT_SIZE,
        "logit_elements": model.CLASS_COUNT,
        "tolerances": {"rtol": RTOL, "atol": ATOL},
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "onnx_sha256": ONNX_SHA256,
        "selection": (
            "Ten class examples and three misclassifications are the official test "
            "indices recorded in models/mnist/evaluation_report.json. "
            "The synthetic input is float32(k / 784)."
        ),
        "cases": manifest_cases,
    }
    (FIXTURE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    require_artifacts()
    prepared = data.load_mnist(download=False)
    network = load_network()
    session = load_session()
    rows: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    all_differences: list[np.ndarray] = []
    for case in case_plan():
        image = input_for_case(case, prepared.test_images, prepared.test_labels)
        pytorch = pytorch_logits(network, image)
        exported = onnx_logits(session, image)
        sample = compare_logits(pytorch, exported, case)
        samples.append(sample)
        all_differences.append(np.abs(pytorch.astype(np.float64) - exported.astype(np.float64)))
        rows.append(
            {
                **case,
                "image": image,
                "pytorch_logits": pytorch,
                "onnx_logits": exported,
                "pytorch_predicted_class": sample["pytorch_predicted_class"],
            }
        )
        print(
            f"{case['name']} true_class {case['true_class']} "
            f"predicted {sample['pytorch_predicted_class']} "
            f"max_abs {sample['maximum_absolute_error']:.8e} "
            f"mean_abs {sample['mean_absolute_error']:.8e}",
            flush=True,
        )

    differences = np.concatenate(all_differences)
    maximum = float(differences.max())
    mean = float(differences.mean())
    agreed = sum(1 for sample in samples if sample["predictions_agree"])
    if agreed != len(samples) or len(samples) != 14:
        raise SystemExit(f"completed {len(samples)} cases with {agreed} agreements")
    report = {
        "checkpoint_sha256": file_sha256(CHECKPOINT_PATH),
        "onnx_sha256": file_sha256(ONNX_PATH),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
        },
        "preprocessing": "uint8 pixels divided by 255, row-major float32[784], no mean or std",
        "tolerances": {"rtol": RTOL, "atol": ATOL},
        "input_count": len(samples),
        "samples": samples,
        "maximum_absolute_error": maximum,
        "mean_absolute_error": mean,
        "predicted_class_agreement": agreed,
        "passed": True,
    }
    if report["checkpoint_sha256"] != CHECKPOINT_SHA256 or report["onnx_sha256"] != ONNX_SHA256:
        raise SystemExit("artifact hash changed during parity verification")
    write_fixtures(rows)
    REPORT_PATH.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"input_count {len(samples)}", flush=True)
    print(f"maximum_absolute_error {maximum:.8e}", flush=True)
    print(f"mean_absolute_error {mean:.8e}", flush=True)
    print(f"predicted_class_agreement {agreed}/{len(samples)}", flush=True)
    print("passed true", flush=True)
    print(f"report {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
