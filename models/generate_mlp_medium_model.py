#!/usr/bin/env python3
"""Write the deterministic medium dynamic-batch MLP and its fixtures.

CMake does not run this script. It requires onnx==1.22.0. An optional
argument is the models directory that receives the ONNX file and fixtures.
The default is the directory that contains this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import onnx

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import generate_mlp_small_model as small
    from weight_rng import require_onnx_version
else:
    from models import generate_mlp_small_model as small
    from models.weight_rng import require_onnx_version

STEM = "mlp_medium_f32"
INPUT_WIDTH = 256
FAN_OUTS = (256, 128, 16)
OUTPUT_NAME = "logits"
PARAMETER_COUNT = 100752
WEIGHT_BYTES = PARAMETER_COUNT * 4
GENERATOR = "models/generate_mlp_medium_model.py"


def build_artifacts() -> dict[str, bytes]:
    drawn = small.draw_mlp(INPUT_WIDTH, FAN_OUTS)
    count = sum(len(values) for _name, _dims, values in drawn)
    if count != PARAMETER_COUNT:
        raise SystemExit(f"parameter count is {count}, expected {PARAMETER_COUNT}")
    model = small.build_model(STEM, INPUT_WIDTH, FAN_OUTS, OUTPUT_NAME, drawn, GENERATOR)
    return small.artifact_map(STEM, INPUT_WIDTH, FAN_OUTS, OUTPUT_NAME, drawn, model)


def write_artifacts(models_dir: Path, files: dict[str, bytes]) -> None:
    small.write_artifacts(models_dir, files)


def main() -> None:
    require_onnx_version(onnx.__version__)
    if len(sys.argv) > 2:
        raise SystemExit("usage: generate_mlp_medium_model.py [models-directory]")
    models_dir = Path(sys.argv[1]) if len(sys.argv) == 2 else Path(__file__).resolve().parent
    write_artifacts(models_dir, build_artifacts())
    sidecar = models_dir / "fixtures" / f"{STEM}.weights"
    if sidecar.stat().st_size != WEIGHT_BYTES:
        raise SystemExit(f"weight sidecar is {sidecar.stat().st_size} bytes, expected {WEIGHT_BYTES}")


if __name__ == "__main__":
    main()
