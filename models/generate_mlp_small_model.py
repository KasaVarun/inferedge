#!/usr/bin/env python3
"""Write the deterministic small dynamic-batch MLP and its fixtures.

CMake does not run this script. It requires onnx==1.22.0. An optional
argument is the models directory that receives the ONNX file and fixtures.
The default is the directory that contains this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from weight_rng import (  # type: ignore[no-redef]
        FIXTURE_BATCH_SIZES,
        MLP_INITIALIZER_NAMES,
        SplitMix64,
        assert_finite,
        draw_vector,
        fixture_attempt_ids,
        input_matrix,
        manifest_text,
        mlp_row,
        pack_f32,
        require_onnx_version,
        rows_are_distinct,
    )
else:
    from models.weight_rng import (
        FIXTURE_BATCH_SIZES,
        MLP_INITIALIZER_NAMES,
        SplitMix64,
        assert_finite,
        draw_vector,
        fixture_attempt_ids,
        input_matrix,
        manifest_text,
        mlp_row,
        pack_f32,
        require_onnx_version,
        rows_are_distinct,
    )

IR_VERSION = 10
OPSET_VERSION = 13
STEM = "mlp_small_f32"
INPUT_WIDTH = 32
FAN_OUTS = (64, 32, 8)
OUTPUT_NAME = "logits"
PARAMETER_COUNT = 4456
WEIGHT_BYTES = PARAMETER_COUNT * 4


def layer_plan(input_width: int, fan_outs: tuple[int, ...]) -> list[tuple[int, int, bool]]:
    if len(fan_outs) != 3 or any(width <= 0 for width in fan_outs):
        raise SystemExit("an MLP plan must contain three positive widths")
    plan: list[tuple[int, int, bool]] = []
    fan_in = input_width
    for index, fan_out in enumerate(fan_outs):
        plan.append((fan_in, fan_out, index < len(fan_outs) - 1))
        fan_in = fan_out
    return plan


def draw_mlp(input_width: int, fan_outs: tuple[int, ...]) -> list[tuple[str, list[int], list[float]]]:
    """Return (name, dims, values) in W1, b1, W2, b2, W3, b3 order."""
    rng = SplitMix64()
    drawn: list[tuple[str, list[int], list[float]]] = []
    name_index = 0
    for fan_in, fan_out, _apply_relu in layer_plan(input_width, fan_outs):
        weight_name = MLP_INITIALIZER_NAMES[name_index]
        bias_name = MLP_INITIALIZER_NAMES[name_index + 1]
        name_index += 2
        weight = draw_vector(rng, fan_in * fan_out, fan_in)
        bias = draw_vector(rng, fan_out, fan_in)
        assert_finite(weight, weight_name)
        assert_finite(bias, bias_name)
        drawn.append((weight_name, [fan_in, fan_out], weight))
        drawn.append((bias_name, [fan_out], bias))
    if [item[0] for item in drawn] != list(MLP_INITIALIZER_NAMES):
        raise SystemExit("initializer names departed from W1, b1, W2, b2, W3, b3")
    count = sum(len(values) for _name, _dims, values in drawn)
    if count != PARAMETER_COUNT and (input_width, fan_outs) == (INPUT_WIDTH, FAN_OUTS):
        raise SystemExit(f"parameter count is {count}, expected {PARAMETER_COUNT}")
    return drawn


def reference_layers(
    drawn: list[tuple[str, list[int], list[float]]],
) -> list[tuple[list[float], list[float], int, int, bool]]:
    layers = []
    for index in range(0, len(drawn), 2):
        _weight_name, weight_dims, weight = drawn[index]
        _bias_name, bias_dims, bias = drawn[index + 1]
        fan_in, fan_out = weight_dims
        if bias_dims != [fan_out]:
            raise SystemExit("bias shape does not match the layer output")
        apply_relu = index + 2 < len(drawn)
        layers.append((weight, bias, fan_in, fan_out, apply_relu))
    return layers


def build_model(
    stem: str,
    input_width: int,
    fan_outs: tuple[int, ...],
    output_name: str,
    drawn: list[tuple[str, list[int], list[float]]],
    generator_script: str,
) -> onnx.ModelProto:
    initializers = [
        helper.make_tensor(name, TensorProto.FLOAT, dims, pack_f32(values), raw=True)
        for name, dims, values in drawn
    ]
    nodes = []
    current = "features"
    for index in range(0, len(drawn), 2):
        layer_number = index // 2 + 1
        weight_name = drawn[index][0]
        bias_name = drawn[index + 1][0]
        is_last = index + 2 == len(drawn)
        multiplied = f"multiplied{layer_number}"
        added = output_name if is_last else f"biased{layer_number}"
        nodes.append(helper.make_node("MatMul", [current, weight_name], [multiplied]))
        nodes.append(helper.make_node("Add", [multiplied, bias_name], [added]))
        if not is_last:
            hidden = f"hidden{layer_number}"
            nodes.append(helper.make_node("Relu", [added], [hidden]))
            current = hidden
    output_width = fan_outs[-1]
    graph = helper.make_graph(
        nodes,
        stem,
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["batch", input_width])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, ["batch", output_width])],
        initializers,
    )
    model = helper.make_model(
        graph,
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET_VERSION)],
        producer_name="inferedge",
        producer_version="0.1.0",
        doc_string=(
            f"Deterministic float32 {stem} MLP. "
            f"Generated by {generator_script}. IR {IR_VERSION}, opset {OPSET_VERSION}."
        ),
    )
    if model.ir_version != IR_VERSION:
        raise SystemExit(f"IR version is {model.ir_version}, expected {IR_VERSION}")
    if len(model.opset_import) != 1 or model.opset_import[0].version != OPSET_VERSION:
        raise SystemExit(f"opset import is not {OPSET_VERSION}")
    for initializer in model.graph.initializer:
        if initializer.data_location != TensorProto.DEFAULT or initializer.external_data:
            raise SystemExit(f"initializer {initializer.name} is not embedded")
    onnx.checker.check_model(model)
    return model


def expected_rows(
    attempt_ids: list[int],
    input_width: int,
    drawn: list[tuple[str, list[int], list[float]]],
) -> list[list[float]]:
    layers = reference_layers(drawn)
    rows = []
    for row in input_matrix(attempt_ids, input_width):
        result = mlp_row(row, layers)
        assert_finite(result, "logits")
        rows.append(result)
    return rows


def artifact_map(
    stem: str,
    input_width: int,
    fan_outs: tuple[int, ...],
    output_name: str,
    drawn: list[tuple[str, list[int], list[float]]],
    model: onnx.ModelProto,
) -> dict[str, bytes]:
    sidecar = b"".join(pack_f32(values) for _name, _dims, values in drawn)
    files = {
        f"{stem}.onnx": model.SerializeToString(),
        f"fixtures/{stem}.weights": sidecar,
        f"fixtures/{stem}/manifest.txt": manifest_text(stem, input_width, fan_outs[-1]).encode(
            "ascii"
        ),
    }
    for batch_size in FIXTURE_BATCH_SIZES:
        attempt_ids = fixture_attempt_ids(batch_size)
        inputs = input_matrix(attempt_ids, input_width)
        if not rows_are_distinct(inputs):
            raise SystemExit(f"{stem} batch {batch_size} input rows are not distinct")
        outputs = expected_rows(attempt_ids, input_width, drawn)
        tag = f"b{batch_size:02d}"
        files[f"fixtures/{stem}/{tag}.input.f32"] = b"".join(pack_f32(row) for row in inputs)
        files[f"fixtures/{stem}/{tag}.expected.f32"] = b"".join(
            pack_f32(row) for row in outputs
        )
    return files


def build_artifacts(
    stem: str = STEM,
    input_width: int = INPUT_WIDTH,
    fan_outs: tuple[int, ...] = FAN_OUTS,
    output_name: str = OUTPUT_NAME,
) -> dict[str, bytes]:
    if (stem, input_width, fan_outs, output_name) != (STEM, INPUT_WIDTH, FAN_OUTS, OUTPUT_NAME):
        raise SystemExit(f"unknown MLP configuration {stem}")
    drawn = draw_mlp(input_width, fan_outs)
    model = build_model(
        stem,
        input_width,
        fan_outs,
        output_name,
        drawn,
        "models/generate_mlp_small_model.py",
    )
    return artifact_map(stem, input_width, fan_outs, output_name, drawn, model)


def write_artifacts(models_dir: Path, files: dict[str, bytes]) -> None:
    root = models_dir.resolve()
    for relative, payload in files.items():
        destination = (root / relative).resolve()
        if destination != root and root not in destination.parents:
            raise SystemExit(f"refusing to write outside {root}: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)


def main() -> None:
    require_onnx_version(onnx.__version__)
    if len(sys.argv) > 2:
        raise SystemExit("usage: generate_mlp_small_model.py [models-directory]")
    models_dir = Path(sys.argv[1]) if len(sys.argv) == 2 else Path(__file__).resolve().parent
    write_artifacts(models_dir, build_artifacts())
    sidecar = models_dir / "fixtures" / f"{STEM}.weights"
    if sidecar.stat().st_size != WEIGHT_BYTES:
        raise SystemExit(f"weight sidecar is {sidecar.stat().st_size} bytes, expected {WEIGHT_BYTES}")


if __name__ == "__main__":
    main()
