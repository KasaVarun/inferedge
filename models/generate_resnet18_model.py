#!/usr/bin/env python3
"""Write the deterministic dynamic-batch ResNet-18 ONNX graph and fixtures.

CMake does not run this script. It requires onnx==1.22.0 and NumPy. An optional
argument is the models directory that receives the ONNX file and fixtures.
The default is the directory that contains this script.

Weights are synthetic SplitMix64 draws (seed 20261017). They are not
ImageNet-pretrained. There is no weight sidecar: the ONNX file embeds
initializers, and known-answer logits come from an independent NumPy
float32 forward pass. ONNX Runtime is not used to write expected files.

NumPy float32 matmul uses the host BLAS. Regeneration is not promised
bit-identical across BLAS libraries. Committed fixture bytes are canonical.

BN uses gamma=1, beta=0, running_mean=0, running_var=1, epsilon=1e-5.
That is not arithmetic identity: y = x / sqrt(1 + 1e-5).
"""

from __future__ import annotations

import hashlib
import math
import platform
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from weight_rng import SplitMix64, assert_finite, draw_vector, f32, pack_f32, require_onnx_version
else:
    from models.weight_rng import (
        SplitMix64,
        assert_finite,
        draw_vector,
        f32,
        pack_f32,
        require_onnx_version,
    )

IR_VERSION = 10
OPSET_VERSION = 13
STEM = "resnet18_f32"
SEED = 20261017
ATTEMPT_PERIOD = 251
INPUT_CODE_MASK = 0xFFFFFF
INPUT_NAME = "images"
OUTPUT_NAME = "logits"
CHANNELS = 3
HEIGHT = 224
WIDTH = 224
CLASSES = 1000
ROW_ELEMS = CHANNELS * HEIGHT * WIDTH
LEARNABLE_PARAMETERS = 11_689_512
BN_EPSILON = 1e-5
FIXTURE_BATCH_SIZES = (1, 2, 4, 8)
FIXTURE_ATTEMPT_BASE = 1000
STAGE_PLAN = (
    (64, 64, 2, 1),
    (64, 128, 2, 2),
    (128, 256, 2, 2),
    (256, 512, 2, 2),
)


def bn_inv_std(variance: float = 1.0, epsilon: float = BN_EPSILON) -> float:
    """float32(1 / sqrt(float64(variance) + float64(epsilon)))."""
    return f32(1.0 / math.sqrt(float(variance) + float(epsilon)))


def conv2d_nchw(images: np.ndarray, weight: np.ndarray, stride: int, pad: int) -> np.ndarray:
    """NCHW Conv, OIHW kernel, zero pad. im2col then float32 matmul.

    No Python loop over spatial pixels. BLAS accumulation order is host-defined.
    """
    if images.dtype != np.float32 or weight.dtype != np.float32:
        raise ValueError("conv2d_nchw requires float32")
    batch, in_c, height, width = images.shape
    out_c, weight_in, kernel_h, kernel_w = weight.shape
    if weight_in != in_c:
        raise ValueError("convolution channel mismatch")
    if pad:
        padded = np.zeros(
            (batch, in_c, height + 2 * pad, width + 2 * pad), dtype=np.float32
        )
        padded[:, :, pad : pad + height, pad : pad + width] = images
    else:
        padded = images
    padded_h = padded.shape[2]
    padded_w = padded.shape[3]
    out_h = (padded_h - kernel_h) // stride + 1
    out_w = (padded_w - kernel_w) // stride + 1
    stride_n, stride_c, stride_h, stride_w = padded.strides
    windows = np.lib.stride_tricks.as_strided(
        padded,
        shape=(batch, in_c, out_h, out_w, kernel_h, kernel_w),
        strides=(stride_n, stride_c, stride_h * stride, stride_w * stride, stride_h, stride_w),
    )
    columns = np.ascontiguousarray(
        windows.transpose(0, 2, 3, 1, 4, 5).reshape(batch * out_h * out_w, in_c * kernel_h * kernel_w)
    )
    kernels = np.ascontiguousarray(weight.reshape(out_c, in_c * kernel_h * kernel_w).T)
    product = columns @ kernels
    return np.ascontiguousarray(
        product.reshape(batch, out_h, out_w, out_c).transpose(0, 3, 1, 2)
    )


def maxpool2d_nchw(
    images: np.ndarray, kernel: int = 3, stride: int = 2, pad: int = 1
) -> np.ndarray:
    """MaxPool with zero padding (valid after ReLU, which is non-negative)."""
    if images.dtype != np.float32:
        raise ValueError("maxpool2d_nchw requires float32")
    batch, channels, height, width = images.shape
    if pad:
        padded = np.zeros(
            (batch, channels, height + 2 * pad, width + 2 * pad), dtype=np.float32
        )
        padded[:, :, pad : pad + height, pad : pad + width] = images
    else:
        padded = images
    padded_h = padded.shape[2]
    padded_w = padded.shape[3]
    out_h = (padded_h - kernel) // stride + 1
    out_w = (padded_w - kernel) // stride + 1
    stride_n, stride_c, stride_h, stride_w = padded.strides
    windows = np.lib.stride_tricks.as_strided(
        padded,
        shape=(batch, channels, out_h, out_w, kernel, kernel),
        strides=(stride_n, stride_c, stride_h * stride, stride_w * stride, stride_h, stride_w),
    )
    return np.ascontiguousarray(np.max(windows, axis=(-1, -2)))


def batch_norm_nchw(
    images: np.ndarray,
    scale: np.ndarray,
    bias: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    epsilon: float = BN_EPSILON,
) -> np.ndarray:
    """Eval BatchNormalization. inv_std is float32(1/sqrt(var+eps)) per channel."""
    if images.dtype != np.float32:
        raise ValueError("batch_norm_nchw requires float32")
    inv_std = np.array(
        [bn_inv_std(float(value), epsilon) for value in variance], dtype=np.float32
    )
    coeff = (scale * inv_std).reshape(1, -1, 1, 1)
    shift = bias.reshape(1, -1, 1, 1)
    center = mean.reshape(1, -1, 1, 1)
    return np.ascontiguousarray((images - center) * coeff + shift)


def relu_nchw(images: np.ndarray) -> np.ndarray:
    return np.maximum(images, np.float32(0.0))


def global_average_pool_nchw(images: np.ndarray) -> np.ndarray:
    """Sum in float32 over H,W then multiply by float32(1/(H*W))."""
    height = images.shape[2]
    width = images.shape[3]
    summed = np.sum(images, axis=(2, 3), dtype=np.float32, keepdims=True)
    scale = np.float32(1.0 / float(height * width))
    return np.ascontiguousarray(summed * scale)


def gemm_logits(features: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """features [N,512] @ weight [512,1000] + bias [1000], all float32."""
    return np.ascontiguousarray(features @ weight + bias)


def image_from_attempt(attempt_id: int) -> np.ndarray:
    """One NCHW image [1,3,224,224] from the frozen attempt_id mapping."""
    if not isinstance(attempt_id, int) or isinstance(attempt_id, bool) or attempt_id < 0:
        raise ValueError("attempt_id must be a non-negative int")
    indices = np.arange(ROW_ELEMS, dtype=np.int64)
    codes = ((attempt_id % ATTEMPT_PERIOD) + indices) & INPUT_CODE_MASK
    values = codes.astype(np.float32) * np.float32(2.0**-24)
    if not np.all(np.isfinite(values)):
        raise ValueError("generated image is not finite")
    return values.reshape(1, CHANNELS, HEIGHT, WIDTH)


def fixture_attempt_ids(batch_size: int) -> list[int]:
    if batch_size not in FIXTURE_BATCH_SIZES:
        raise ValueError(f"unsupported fixture batch size {batch_size}")
    return [FIXTURE_ATTEMPT_BASE + row for row in range(batch_size)]


def stack_images(attempt_ids: list[int]) -> np.ndarray:
    return np.concatenate([image_from_attempt(attempt_id) for attempt_id in attempt_ids], axis=0)


class ResNet18:
    """Holds named float32 initializers and runs the NumPy and ONNX graphs."""

    def __init__(self, rng: SplitMix64) -> None:
        self.rng = rng
        self.initializers: list[tuple[str, list[int], list[float]]] = []
        self.nodes: list[onnx.NodeProto] = []
        self.learnable = 0
        self.arrays: dict[str, np.ndarray] = {}

    def _add_tensor(self, name: str, dims: list[int], values: list[float], learnable: bool) -> None:
        assert_finite(values, name)
        self.initializers.append((name, dims, values))
        self.arrays[name] = np.array(values, dtype=np.float32).reshape(dims)
        if learnable:
            self.learnable += len(values)

    def add_conv(
        self, source: str, name: str, out_c: int, in_c: int, kernel: int, stride: int, pad: int
    ) -> str:
        weight_name = name + ".weight"
        fan_in = in_c * kernel * kernel
        values = draw_vector(self.rng, out_c * fan_in, fan_in)
        self._add_tensor(weight_name, [out_c, in_c, kernel, kernel], values, True)
        output = name + ".conv"
        self.nodes.append(
            helper.make_node(
                "Conv",
                [source, weight_name],
                [output],
                kernel_shape=[kernel, kernel],
                strides=[stride, stride],
                pads=[pad, pad, pad, pad],
            )
        )
        return output

    def add_bn(self, source: str, name: str, channels: int) -> str:
        ones = [f32(1.0)] * channels
        zeros = [f32(0.0)] * channels
        self._add_tensor(name + ".scale", [channels], ones, True)
        self._add_tensor(name + ".B", [channels], zeros, True)
        self._add_tensor(name + ".mean", [channels], zeros, False)
        self._add_tensor(name + ".var", [channels], ones, False)
        output = name + ".bn"
        self.nodes.append(
            helper.make_node(
                "BatchNormalization",
                [source, name + ".scale", name + ".B", name + ".mean", name + ".var"],
                [output],
                epsilon=BN_EPSILON,
            )
        )
        return output

    def add_relu(self, source: str, name: str) -> str:
        output = name + ".relu"
        self.nodes.append(helper.make_node("Relu", [source], [output]))
        return output

    def basic_block(self, source: str, name: str, in_c: int, out_c: int, stride: int) -> str:
        conv1 = self.add_conv(source, name + ".conv1", out_c, in_c, 3, stride, 1)
        bn1 = self.add_bn(conv1, name + ".bn1", out_c)
        relu1 = self.add_relu(bn1, name + ".inner")
        conv2 = self.add_conv(relu1, name + ".conv2", out_c, out_c, 3, 1, 1)
        bn2 = self.add_bn(conv2, name + ".bn2", out_c)
        residual = source
        if stride != 1 or in_c != out_c:
            down = self.add_conv(source, name + ".downsample.conv", out_c, in_c, 1, stride, 0)
            residual = self.add_bn(down, name + ".downsample.bn", out_c)
        added = name + ".add"
        self.nodes.append(helper.make_node("Add", [bn2, residual], [added]))
        return self.add_relu(added, name)

    def build_graph(self) -> str:
        current = self.add_conv(INPUT_NAME, "conv1", 64, 3, 7, 2, 3)
        current = self.add_bn(current, "bn1", 64)
        current = self.add_relu(current, "stem")
        current_name = "maxpool"
        self.nodes.append(
            helper.make_node(
                "MaxPool",
                [current],
                [current_name],
                kernel_shape=[3, 3],
                strides=[2, 2],
                pads=[1, 1, 1, 1],
            )
        )
        current = current_name
        in_c = 64
        for stage_index, (stage_in, stage_out, blocks, first_stride) in enumerate(STAGE_PLAN):
            if stage_in != in_c:
                raise SystemExit(f"stage {stage_index} in_c mismatch")
            for block_index in range(blocks):
                stride = first_stride if block_index == 0 else 1
                block_in = in_c if block_index == 0 else stage_out
                current = self.basic_block(
                    current, f"layer{stage_index + 1}.{block_index}", block_in, stage_out, stride
                )
            in_c = stage_out
        pooled = "avgpool"
        self.nodes.append(helper.make_node("GlobalAveragePool", [current], [pooled]))
        flat = "flatten"
        self.nodes.append(helper.make_node("Flatten", [pooled], [flat], axis=1))
        weight_values = draw_vector(self.rng, 512 * CLASSES, 512)
        bias_values = draw_vector(self.rng, CLASSES, 512)
        self._add_tensor("fc.weight", [512, CLASSES], weight_values, True)
        self._add_tensor("fc.bias", [CLASSES], bias_values, True)
        self.nodes.append(
            helper.make_node(
                "Gemm",
                [flat, "fc.weight", "fc.bias"],
                [OUTPUT_NAME],
                transA=0,
                transB=0,
                alpha=1.0,
                beta=1.0,
            )
        )
        if in_c != 512:
            raise SystemExit("final residual width is not 512")
        if self.learnable != LEARNABLE_PARAMETERS:
            raise SystemExit(
                f"learnable parameter count is {self.learnable}, expected {LEARNABLE_PARAMETERS}"
            )
        return OUTPUT_NAME

    def numpy_forward(self, images: np.ndarray) -> np.ndarray:
        if images.dtype != np.float32 or images.ndim != 4:
            raise ValueError("numpy_forward requires NCHW float32")
        if images.shape[1:] != (CHANNELS, HEIGHT, WIDTH):
            raise ValueError("numpy_forward spatial contract mismatch")
        arrays = self.arrays
        current = conv2d_nchw(images, arrays["conv1.weight"], 2, 3)
        current = batch_norm_nchw(
            current, arrays["bn1.scale"], arrays["bn1.B"], arrays["bn1.mean"], arrays["bn1.var"]
        )
        current = relu_nchw(current)
        current = maxpool2d_nchw(current)
        in_c = 64
        for stage_index, (stage_in, stage_out, blocks, first_stride) in enumerate(STAGE_PLAN):
            if stage_in != in_c:
                raise SystemExit(f"numpy stage {stage_index} in_c mismatch")
            for block_index in range(blocks):
                stride = first_stride if block_index == 0 else 1
                block_in = in_c if block_index == 0 else stage_out
                prefix = f"layer{stage_index + 1}.{block_index}"
                residual = current
                inner = conv2d_nchw(current, arrays[prefix + ".conv1.weight"], stride, 1)
                inner = batch_norm_nchw(
                    inner,
                    arrays[prefix + ".bn1.scale"],
                    arrays[prefix + ".bn1.B"],
                    arrays[prefix + ".bn1.mean"],
                    arrays[prefix + ".bn1.var"],
                )
                inner = relu_nchw(inner)
                inner = conv2d_nchw(inner, arrays[prefix + ".conv2.weight"], 1, 1)
                inner = batch_norm_nchw(
                    inner,
                    arrays[prefix + ".bn2.scale"],
                    arrays[prefix + ".bn2.B"],
                    arrays[prefix + ".bn2.mean"],
                    arrays[prefix + ".bn2.var"],
                )
                if stride != 1 or block_in != stage_out:
                    residual = conv2d_nchw(
                        residual, arrays[prefix + ".downsample.conv.weight"], stride, 0
                    )
                    residual = batch_norm_nchw(
                        residual,
                        arrays[prefix + ".downsample.bn.scale"],
                        arrays[prefix + ".downsample.bn.B"],
                        arrays[prefix + ".downsample.bn.mean"],
                        arrays[prefix + ".downsample.bn.var"],
                    )
                current = relu_nchw(inner + residual)
            in_c = stage_out
        pooled = global_average_pool_nchw(current)
        features = pooled.reshape(images.shape[0], 512)
        logits = gemm_logits(features, arrays["fc.weight"], arrays["fc.bias"])
        if not np.all(np.isfinite(logits)):
            raise SystemExit("NumPy reference produced a non-finite logit")
        return logits


def build_model(net: ResNet18) -> onnx.ModelProto:
    net.build_graph()
    initializers = [
        helper.make_tensor(name, TensorProto.FLOAT, dims, pack_f32(values), raw=True)
        for name, dims, values in net.initializers
    ]
    graph = helper.make_graph(
        net.nodes,
        STEM,
        [
            helper.make_tensor_value_info(
                INPUT_NAME, TensorProto.FLOAT, ["batch", CHANNELS, HEIGHT, WIDTH]
            )
        ],
        [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, ["batch", CLASSES])],
        initializers,
    )
    model = helper.make_model(
        graph,
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET_VERSION)],
        producer_name="inferedge",
        producer_version="0.1.0",
        doc_string=(
            f"Deterministic synthetic float32 ResNet-18 ({STEM}). "
            f"Generated by models/generate_resnet18_model.py. "
            f"IR {IR_VERSION}, opset {OPSET_VERSION}. Seed {SEED}. "
            "Weights are not ImageNet-pretrained."
        ),
    )
    if model.ir_version != IR_VERSION:
        raise SystemExit(f"IR version is {model.ir_version}, expected {IR_VERSION}")
    if len(model.opset_import) != 1 or model.opset_import[0].version != OPSET_VERSION:
        raise SystemExit(f"opset import is not {OPSET_VERSION}")
    for initializer in model.graph.initializer:
        if initializer.data_location != TensorProto.DEFAULT or initializer.external_data:
            raise SystemExit(f"initializer {initializer.name} is not embedded")
        roundtrip = numpy_helper.to_array(initializer)
        expected = net.arrays[initializer.name]
        if roundtrip.shape != expected.shape or roundtrip.dtype != expected.dtype:
            raise SystemExit(f"initializer {initializer.name} round-trip shape mismatch")
        if not np.array_equal(roundtrip, expected):
            raise SystemExit(f"initializer {initializer.name} round-trip bytes mismatch")
    onnx.checker.check_model(model)
    return model


def blas_identity() -> str:
    """Record the NumPy build BLAS without printing np.__config__.show()."""
    parts = [f"numpy={np.__version__}"]
    config = getattr(np, "__config__", None)
    show = getattr(config, "show", None)
    if callable(show):
        try:
            payload = show(mode="dicts")
        except TypeError:
            payload = None
        if isinstance(payload, dict):
            build = payload.get("Build Dependencies", payload)
            if isinstance(build, dict):
                blas = build.get("blas") or build.get("openblas") or build.get("accelerate")
                parts.append(f"blas={blas}")
            parts.append(f"build_keys={sorted(payload.keys())}")
        else:
            parts.append("blas=unreported")
    return "; ".join(parts)


def manifest_text(net: ResNet18, environment: dict[str, str]) -> str:
    lines = [
        "# inferedge float32 fixtures",
        "# little-endian float32 values, no file header, no padding",
        "# batch input_file expected_file input_elems output_elems attempt_base",
        f"# model {STEM}",
        "# architecture ResNet-v1 BasicBlock [2,2,2,2] synthetic SplitMix64 weights",
        "# not ImageNet-pretrained",
        f"# seed {SEED}",
        f"# opset {OPSET_VERSION} ir {IR_VERSION} producer inferedge",
        f"# input {INPUT_NAME} float32 [batch,3,224,224]",
        f"# output {OUTPUT_NAME} float32 [batch,1000]",
        f"# learnable_parameters {net.learnable}",
        f"# initializer_count {len(net.initializers)}",
        "# initializer_order " + ",".join(name for name, _dims, _values in net.initializers),
        f"# python {environment['python']}",
        f"# numpy {environment['numpy']}",
        f"# onnx {environment['onnx']}",
        f"# platform {environment['platform']}",
        f"# blas {environment['blas']}",
        "# bn gamma=1 beta=0 mean=0 var=1 epsilon=1e-5 (not arithmetic identity)",
        "# numpy GAP: float32 sum over H,W then * float32(1/(H*W))",
        "# numpy Conv: im2col + float32 matmul; BLAS order not bit-stable across hosts",
        "# numerical coverage: attempt_ids 1000..1007 only; fixture replay",
        "# no weight sidecar",
    ]
    for batch_size in FIXTURE_BATCH_SIZES:
        tag = f"b{batch_size:02d}"
        lines.append(
            f"{batch_size} {tag}.input.f32 {tag}.expected.f32 "
            f"{ROW_ELEMS} {CLASSES} {FIXTURE_ATTEMPT_BASE}"
        )
    lines.append("")
    return "\n".join(lines)


def environment_record() -> dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "onnx": onnx.__version__,
        "platform": platform.platform() + " " + platform.machine(),
        "blas": blas_identity(),
    }


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_artifacts() -> dict[str, bytes]:
    require_onnx_version(onnx.__version__)
    net = ResNet18(SplitMix64(SEED))
    model = build_model(net)
    environment = environment_record()
    files: dict[str, bytes] = {
        f"{STEM}.onnx": model.SerializeToString(),
        f"fixtures/{STEM}/manifest.txt": manifest_text(net, environment).encode("ascii"),
    }
    for batch_size in FIXTURE_BATCH_SIZES:
        attempt_ids = fixture_attempt_ids(batch_size)
        images = stack_images(attempt_ids)
        if images.shape != (batch_size, CHANNELS, HEIGHT, WIDTH):
            raise SystemExit(f"batch {batch_size} input shape is {images.shape}")
        if batch_size >= 2 and np.array_equal(images[0], images[1]):
            raise SystemExit(f"batch {batch_size} images 0 and 1 are identical")
        logits = net.numpy_forward(images)
        if logits.shape != (batch_size, CLASSES):
            raise SystemExit(f"batch {batch_size} logits shape is {logits.shape}")
        if not np.all(np.isfinite(images)) or not np.all(np.isfinite(logits)):
            raise SystemExit(f"batch {batch_size} contains a non-finite value")
        tag = f"b{batch_size:02d}"
        files[f"fixtures/{STEM}/{tag}.input.f32"] = images.tobytes(order="C")
        files[f"fixtures/{STEM}/{tag}.expected.f32"] = logits.tobytes(order="C")
    return files


def write_artifacts(models_dir: Path, files: dict[str, bytes]) -> None:
    root = models_dir.resolve()
    for relative, payload in files.items():
        destination = (root / relative).resolve()
        if destination != root and root not in destination.parents:
            raise SystemExit(f"refusing to write outside {root}: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)


def main() -> None:
    if len(sys.argv) > 2:
        raise SystemExit("usage: generate_resnet18_model.py [models-directory]")
    models_dir = Path(sys.argv[1]) if len(sys.argv) == 2 else Path(__file__).resolve().parent
    files = build_artifacts()
    write_artifacts(models_dir, files)
    onnx_path = models_dir / f"{STEM}.onnx"
    total = sum(len(payload) for payload in files.values())
    print(f"learnable_parameters {LEARNABLE_PARAMETERS}")
    print(f"onnx_bytes {len(files[f'{STEM}.onnx'])}")
    print(f"total_bytes {total}")
    print(f"onnx_sha256 {sha256_bytes(files[f'{STEM}.onnx'])}")
    print(f"manifest_sha256 {sha256_bytes(files[f'fixtures/{STEM}/manifest.txt'])}")
    for batch_size in FIXTURE_BATCH_SIZES:
        tag = f"b{batch_size:02d}"
        print(
            f"{tag}_input_sha256 {sha256_bytes(files[f'fixtures/{STEM}/{tag}.input.f32'])}"
        )
        print(
            f"{tag}_expected_sha256 {sha256_bytes(files[f'fixtures/{STEM}/{tag}.expected.f32'])}"
        )
    print(f"onnx_path {onnx_path}")
    if total > 60_000_000:
        raise SystemExit(f"artifact total {total} bytes exceeds the 54-56 MB budget")


if __name__ == "__main__":
    main()
