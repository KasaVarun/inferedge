"""Export the selected MNIST checkpoint to an ONNX file.

torch.onnx.export defaults to dynamo=True. That path needs onnxscript, which
is not installed, and this script does not install it. The call below uses the
TorchScript exporter with dynamo=False, opset_version=13, and weights stored
inside the file.

That exporter writes IR 7. The graph is only MatMul, Add, and Relu on float
tensors with fixed shapes, which IR 10 can represent. After
onnx.checker.check_model passes, the saved file uses IR 10. A newer IR is not
rewritten downward.

The checkpoint bytes are not modified. This script does not train and does not
score the official test set.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

import model

CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "mnist_mlp.pt"
ONNX_PATH = Path(__file__).resolve().parents[1] / "mnist_mlp.onnx"
SELECTED_SHA256 = "b53a41676adfa020e3808617539f7966776fdd20bd7a247f34656e230459f22a"
OPSET_VERSION = 13
IR_VERSION = 10
ALLOWED_OPERATORS = frozenset({"MatMul", "Add", "Relu"})


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def example_image() -> torch.Tensor:
    """Deterministic float32[784] trace and smoke input. Values are k/784."""
    return torch.arange(model.INPUT_SIZE, dtype=torch.float32) / float(model.INPUT_SIZE)


def load_network() -> model.MnistMlp:
    digest = file_sha256(CHECKPOINT_PATH)
    if digest != SELECTED_SHA256:
        raise SystemExit(f"checkpoint sha256 is {digest}, expected {SELECTED_SHA256}")
    network = model.MnistMlp().to(torch.device("cpu"))
    state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    expected = ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias")
    if tuple(state) != expected:
        raise SystemExit(f"checkpoint tensors are {tuple(state)}")
    if not all(torch.isfinite(tensor).all() for tensor in state.values()):
        raise SystemExit("checkpoint has a non-finite parameter")
    network.load_state_dict(state)
    counted = model.parameter_count(network)
    if counted != model.PARAMETER_COUNT:
        raise SystemExit(f"parameter count is {counted}, expected {model.PARAMETER_COUNT}")
    network.eval()
    return network


def static_shape(value_info: onnx.ValueInfoProto) -> list[int]:
    tensor_type = value_info.type.tensor_type
    if tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise SystemExit(f"{value_info.name} is not float32")
    dims: list[int] = []
    for dim in tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            raise SystemExit(f"{value_info.name} has a dynamic dimension")
        dims.append(int(dim.dim_value))
    return dims


def prepare_model(path: Path) -> onnx.ModelProto:
    exported = onnx.load(path)
    if exported.ir_version > IR_VERSION:
        raise SystemExit(
            f"exporter wrote IR {exported.ir_version}, which is newer than IR {IR_VERSION}"
        )
    opsets = [(item.domain, item.version) for item in exported.opset_import]
    if opsets != [("", OPSET_VERSION)]:
        raise SystemExit(f"opset imports are {opsets}, expected [('', {OPSET_VERSION})]")
    operators = [node.op_type for node in exported.graph.node]
    if not operators or not set(operators) <= ALLOWED_OPERATORS:
        raise SystemExit(f"operator types are {operators}")
    if len(exported.graph.input) != 1 or len(exported.graph.output) != 1:
        raise SystemExit("the graph must have one input and one output")
    image = exported.graph.input[0]
    logits = exported.graph.output[0]
    if image.name != "image" or static_shape(image) != [model.INPUT_SIZE]:
        raise SystemExit(f"input is {image.name} {static_shape(image)}")
    if logits.name != "logits" or static_shape(logits) != [model.CLASS_COUNT]:
        raise SystemExit(f"output is {logits.name} {static_shape(logits)}")
    elements = 0
    for initializer in exported.graph.initializer:
        array = onnx.numpy_helper.to_array(initializer)
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise SystemExit(f"initializer {initializer.name} is not finite float32")
        elements += int(array.size)
    if elements != model.PARAMETER_COUNT:
        raise SystemExit(f"initializer elements are {elements}, expected {model.PARAMETER_COUNT}")
    if exported.ir_version != IR_VERSION:
        exported.ir_version = IR_VERSION
    onnx.checker.check_model(exported)
    if exported.ir_version != IR_VERSION:
        raise SystemExit(f"IR version is {exported.ir_version}, expected {IR_VERSION}")
    return exported


def smoke_inference(path: Path, image: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(path.read_bytes(), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise SystemExit(f"providers are {session.get_providers()}")
    result = session.run(None, {"image": image})
    if len(result) != 1:
        raise SystemExit(f"session returned {len(result)} outputs")
    logits = np.asarray(result[0])
    if logits.shape != (model.CLASS_COUNT,) or logits.dtype != np.float32 or not np.isfinite(logits).all():
        raise SystemExit(f"ONNX logits are shape {logits.shape} dtype {logits.dtype}")
    return logits


def main() -> None:
    before = file_sha256(CHECKPOINT_PATH)
    network = load_network()
    image = example_image()
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory) / "mnist_mlp.onnx"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.onnx.export(
                network,
                (image,),
                str(temporary),
                dynamo=False,
                opset_version=OPSET_VERSION,
                external_data=False,
                input_names=["image"],
                output_names=["logits"],
                export_params=True,
                keep_initializers_as_inputs=False,
            )
        for item in caught:
            print(f"warning: {item.category.__name__}: {item.message}", flush=True)
        exported = prepare_model(temporary)
        onnx.save(exported, ONNX_PATH)

    logits = smoke_inference(ONNX_PATH, image.numpy())
    after = file_sha256(CHECKPOINT_PATH)
    if after != before or after != SELECTED_SHA256:
        raise SystemExit(f"checkpoint sha256 changed from {before} to {after}")
    model_bytes = ONNX_PATH.read_bytes()
    digest = hashlib.sha256(model_bytes).hexdigest()
    checked = onnx.load(ONNX_PATH)
    print("exporter torch.onnx.export dynamo=False opset_version=13 external_data=False", flush=True)
    print(f"ir_version {checked.ir_version}", flush=True)
    print(f"opset {[(item.domain, item.version) for item in checked.opset_import]}", flush=True)
    print(f"operators {[node.op_type for node in checked.graph.node]}", flush=True)
    print(f"bytes {len(model_bytes)}", flush=True)
    print(f"sha256 {digest}", flush=True)
    print(f"checkpoint_sha256 {after}", flush=True)
    print("synthetic_input float32[784] values k/784 for k in 0..783", flush=True)
    print(f"onnx_logits {logits.tolist()}", flush=True)
    print(f"onnx {ONNX_PATH}", flush=True)


if __name__ == "__main__":
    main()
