"""Run one MNIST fixture through the native inferedge CLI.

Python reads the little-endian fixture and checks the printed logits.
It does not import a neural-network library and it does not compute logits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

INPUT_COUNT = 784
LOGIT_COUNT = 10
RTOL = 1e-4
ATOL = 1e-5
DEFAULT_FIXTURE = "test_00003"
TOKEN_LIMIT = 64


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def fixture_dir() -> Path:
    return repository_root() / "models" / "fixtures" / "mnist"


def model_path() -> Path:
    return repository_root() / "models" / "mnist_mlp.onnx"


def load_manifest() -> dict:
    return json.loads((fixture_dir() / "manifest.json").read_text(encoding="utf-8"))


def select_case(manifest: dict, name: str) -> dict:
    for case in manifest["cases"]:
        if case["name"] == name:
            return case
    known = ", ".join(case["name"] for case in manifest["cases"])
    raise ValueError(f"unknown fixture {name}. Known cases: {known}")


def read_float32(path: Path, count: int) -> list[float]:
    payload = path.read_bytes()
    expected = count * 4
    if len(payload) != expected:
        raise ValueError(f"{path} has {len(payload)} bytes, expected {expected}")
    values = list(struct.unpack(f"<{count}f", payload))
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{path} contains a non-finite value")
    return values


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_input(case: dict) -> list[float]:
    path = fixture_dir() / case["input"]
    digest = sha256(path)
    if digest != case["input_sha256"]:
        raise ValueError(f"{path} sha256 is {digest}, expected {case['input_sha256']}")
    values = read_float32(path, INPUT_COUNT)
    if len(values) != INPUT_COUNT:
        raise ValueError(f"{path} has {len(values)} floats, expected {INPUT_COUNT}")
    return values


def load_reference_logits(case: dict) -> list[float]:
    path = fixture_dir() / case["onnx_logits"]
    digest = sha256(path)
    if digest != case["onnx_logits_sha256"]:
        raise ValueError(f"{path} sha256 is {digest}, expected {case['onnx_logits_sha256']}")
    return read_float32(path, LOGIT_COUNT)


def format_token(value: float) -> str:
    token = format(value, ".9g")
    if len(token) > TOKEN_LIMIT or not math.isfinite(value):
        raise ValueError(f"cannot encode {value} as a CLI token")
    return token


def image_argument(values: list[float]) -> str:
    if len(values) != INPUT_COUNT:
        raise ValueError(f"image argument needs {INPUT_COUNT} floats, got {len(values)}")
    return "image=" + ",".join(format_token(value) for value in values)


def executable_path(requested: Path | None = None) -> Path:
    if requested is not None:
        return requested
    environment = os.environ.get("INFEREDGE_EXECUTABLE")
    if environment:
        return Path(environment)
    root = repository_root()
    for relative in (Path("build") / "inferedge", Path("build-release") / "inferedge"):
        candidate = root / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("inferedge executable was not found in build/ or build-release/")


def launch(executable: Path, model: Path, image_arg: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["ORT_DISABLE_TELEMETRY"] = "1"
    return subprocess.run(
        [str(executable), "--model", str(model), "--input", image_arg],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )


def parse_logits(stdout: str) -> list[float]:
    marker = "Output logits:\n"
    if marker not in stdout:
        raise ValueError("CLI stdout has no logits output")
    line = stdout.split(marker, 1)[1].splitlines()[0].strip()
    if not line.startswith("[") or not line.endswith("]"):
        raise ValueError(f"CLI logit line is not a bracket list: {line}")
    body = line[1:-1].strip()
    parts = [part.strip() for part in body.split(",")] if body else []
    if len(parts) != LOGIT_COUNT:
        raise ValueError(f"CLI returned {len(parts)} logits, expected {LOGIT_COUNT}")
    values = [float(part) for part in parts]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("CLI logits are not finite")
    return values


def predicted_class(logits: list[float]) -> int:
    if len(logits) != LOGIT_COUNT:
        raise ValueError(f"predicted class needs {LOGIT_COUNT} logits")
    best = 0
    for index, value in enumerate(logits):
        if value > logits[best]:
            best = index
    return best


def absolute_errors(actual: list[float], reference: list[float]) -> list[float]:
    if len(actual) != len(reference):
        raise ValueError("logit lists have different lengths")
    return [abs(left - right) for left, right in zip(actual, reference, strict=True)]


def within_tolerance(actual: list[float], reference: list[float]) -> bool:
    for left, right, difference in zip(actual, reference, absolute_errors(actual, reference), strict=True):
        del left
        if difference > ATOL + RTOL * abs(right):
            return False
    return True


def demonstrate(name: str, executable: Path | None = None) -> dict[str, object]:
    case = select_case(load_manifest(), name)
    values = load_input(case)
    reference = load_reference_logits(case)
    completed = launch(executable_path(executable), model_path(), image_argument(values))
    if completed.returncode != 0:
        raise SystemExit(
            f"inferedge exited {completed.returncode}\n{completed.stderr}{completed.stdout}"
        )
    logits = parse_logits(completed.stdout)
    predicted = predicted_class(logits)
    expected = int(case["expected_predicted_class"])
    differences = absolute_errors(logits, reference)
    return {
        "fixture": case["name"],
        "true_class": case["true_class"],
        "expected_predicted_class": expected,
        "predicted_class": predicted,
        "prediction_matches": predicted == expected,
        "within_tolerance": within_tolerance(logits, reference),
        "exit_status": completed.returncode,
        "maximum_absolute_error": max(differences),
        "mean_absolute_error": sum(differences) / len(differences),
        "logits": logits,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def print_result(result: dict[str, object]) -> None:
    true_class = result["true_class"]
    print(f"fixture {result['fixture']}", flush=True)
    print(f"true_digit {true_class if true_class is not None else 'none'}", flush=True)
    print(f"predicted_digit {result['predicted_class']}", flush=True)
    print(f"manifest_expected {result['expected_predicted_class']}", flush=True)
    print(f"prediction_matches {str(result['prediction_matches']).lower()}", flush=True)
    print(f"within_tolerance {str(result['within_tolerance']).lower()}", flush=True)
    print(f"exit_status {result['exit_status']}", flush=True)
    print(f"maximum_absolute_error {result['maximum_absolute_error']:.8e}", flush=True)
    print(f"mean_absolute_error {result['mean_absolute_error']:.8e}", flush=True)
    stdout = str(result["stdout"])
    native_line = stdout.split("Output logits:\n", 1)[1].splitlines()[0].strip()
    print(f"native_stdout {native_line}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Demonstrate MNIST inference with the native CLI.")
    parser.add_argument("fixture", nargs="?", default=DEFAULT_FIXTURE)
    parser.add_argument("--executable", type=Path)
    args = parser.parse_args(argv)
    result = demonstrate(args.fixture, args.executable)
    print_result(result)
    if not result["prediction_matches"] or not result["within_tolerance"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
