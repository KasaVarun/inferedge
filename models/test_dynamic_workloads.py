"""Deterministic generation checks for the dynamic MLP and GEMM workloads."""

from __future__ import annotations

import hashlib
import math
import re
import struct
import tempfile
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto

from models import generate_gemm_model as gemm_gen
from models import generate_mlp_medium_model as medium_gen
from models import generate_mlp_small_model as small_gen
from models import weight_rng as rng

REPO_MODELS = Path(__file__).resolve().parent
RESNET18_HOST_MANIFEST_KEYS = frozenset({"python", "numpy", "onnx", "platform", "blas"})
RESNET18_STABLE_MANIFEST_PREFIXES = (
    "# inferedge float32 fixtures",
    "# little-endian float32 values, no file header, no padding",
    "# batch input_file expected_file input_elems output_elems attempt_base",
    "# model ",
    "# architecture ",
    "# not ImageNet-pretrained",
    "# seed ",
    "# opset ",
    "# input ",
    "# output ",
    "# learnable_parameters ",
    "# initializer_count ",
    "# initializer_order ",
    "# bn ",
    "# numpy GAP:",
    "# numpy Conv:",
    "# numerical coverage:",
    "# no weight sidecar",
)
RESNET18_MANIFEST_RECORD = re.compile(
    r"^(\d+) ([A-Za-z0-9._-]+) ([A-Za-z0-9._-]+) (\d+) (\d+) (\d+)$"
)
RESNET18_REGENERATED_RELATIVES = frozenset(
    {
        "resnet18_f32.onnx",
        "fixtures/resnet18_f32/manifest.txt",
        "fixtures/resnet18_f32/b01.input.f32",
        "fixtures/resnet18_f32/b01.expected.f32",
        "fixtures/resnet18_f32/b02.input.f32",
        "fixtures/resnet18_f32/b02.expected.f32",
        "fixtures/resnet18_f32/b04.input.f32",
        "fixtures/resnet18_f32/b04.expected.f32",
        "fixtures/resnet18_f32/b08.input.f32",
        "fixtures/resnet18_f32/b08.expected.f32",
    }
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def resnet18_host_manifest_key(line: str) -> str | None:
    """Return the provenance key if this comment is host-specific."""
    if line.startswith("# python "):
        return "python"
    if line.startswith("# onnx "):
        return "onnx"
    if line.startswith("# platform "):
        return "platform"
    if line.startswith("# blas "):
        return "blas"
    if line.startswith("# numpy ") and not (
        line.startswith("# numpy GAP:") or line.startswith("# numpy Conv:")
    ):
        return "numpy"
    return None


def resnet18_manifest_stable_lines(text: str) -> list[str]:
    """Return contract lines after requiring known host provenance comments.

    Host keys python, numpy, onnx, platform, and blas may differ across
    machines. `# numpy GAP:` and `# numpy Conv:` stay in the stable contract.
    Every other allowed comment and every fixture record must match.
    An unknown comment or a malformed line is an error, not a skip.
    """
    if not isinstance(text, str):
        raise ValueError("manifest text must be str")
    if not text.endswith("\n"):
        raise ValueError("manifest must end with a newline")
    lines = text.split("\n")
    if lines[-1] != "":
        raise ValueError("manifest must end with an empty line")
    stable: list[str] = []
    host_keys: list[str] = []
    for line in lines[:-1]:
        host_key = resnet18_host_manifest_key(line)
        if host_key is not None:
            value = line.split(" ", 2)[-1] if line.startswith("# ") else ""
            if not value:
                raise ValueError(f"empty host provenance value for {host_key}")
            host_keys.append(host_key)
            continue
        if line.startswith("#"):
            if not line.startswith("# "):
                raise ValueError(f"malformed manifest comment: {line!r}")
            if not any(line == prefix or line.startswith(prefix) for prefix in RESNET18_STABLE_MANIFEST_PREFIXES):
                raise ValueError(f"unknown manifest comment: {line!r}")
            stable.append(line)
            continue
        if RESNET18_MANIFEST_RECORD.fullmatch(line) is None:
            raise ValueError(f"malformed manifest record: {line!r}")
        stable.append(line)
    if len(host_keys) != len(set(host_keys)):
        raise ValueError(f"duplicate host provenance keys: {host_keys}")
    if set(host_keys) != RESNET18_HOST_MANIFEST_KEYS:
        raise ValueError(
            "host provenance keys are "
            f"{sorted(set(host_keys))}, expected {sorted(RESNET18_HOST_MANIFEST_KEYS)}"
        )
    return stable


def unpack_f32(payload: bytes) -> list[float]:
    if len(payload) % 4 != 0:
        raise AssertionError(f"payload length {len(payload)} is not a multiple of 4")
    return [
        struct.unpack_from("<f", payload, offset)[0] for offset in range(0, len(payload), 4)
    ]


def initializer_bytes(tensor: onnx.TensorProto) -> bytes:
    if tensor.data_type != TensorProto.FLOAT:
        raise AssertionError(f"{tensor.name} is not float32")
    if tensor.raw_data:
        return bytes(tensor.raw_data)
    if tensor.float_data:
        return b"".join(struct.pack("<f", value) for value in tensor.float_data)
    raise AssertionError(f"{tensor.name} has no float payload")


def dynamic_static_shape(value_info: onnx.ValueInfoProto) -> tuple[str, int]:
    dims = value_info.type.tensor_type.shape.dim
    if len(dims) != 2:
        raise AssertionError(f"{value_info.name} rank is {len(dims)}")
    if value_info.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise AssertionError(f"{value_info.name} is not float32")
    if dims[0].dim_param != "batch" or dims[0].HasField("dim_value"):
        raise AssertionError(f"{value_info.name} axis 0 is not the dynamic batch")
    if not dims[1].HasField("dim_value") or dims[1].dim_value <= 0:
        raise AssertionError(f"{value_info.name} axis 1 is not a positive static width")
    return dims[0].dim_param, int(dims[1].dim_value)


class RngAndReferenceTest(unittest.TestCase):
    def test_onnx_pin_and_rejection(self) -> None:
        self.assertEqual(rng.REQUIRED_ONNX_VERSION, "1.22.0")
        self.assertEqual(onnx.__version__, "1.22.0")
        rng.require_onnx_version("1.22.0")
        with self.assertRaises(SystemExit):
            rng.require_onnx_version("1.21.0")

    def test_splitmix_vectors(self) -> None:
        generator = rng.SplitMix64(20261015)
        expected_state = (
            0x9E3779B9807FA4AC,
            0x3C6EF372FFCA20C1,
            0xDAA66D2C7F149CD6,
            0x78DDE6E5FE5F18EB,
        )
        expected_draw = (
            0x68BE0AF9EDB90DBE,
            0x06E0F5B77DE996D7,
            0xBA55FA73FB737E6A,
            0x567D75658A8BB78B,
        )
        for state, draw in zip(expected_state, expected_draw, strict=True):
            actual = generator.next_u64()
            self.assertEqual(actual, draw)
            self.assertEqual(generator.state, state)
            self.assertEqual(rng.high_24_bits(actual), (draw >> 40) & 0xFFFFFF)

    def test_float32_little_endian_one(self) -> None:
        self.assertEqual(rng.f32_le(1.0), b"\x00\x00\x80\x3f")
        self.assertEqual(rng.f32(1.0), 1.0)

    def test_square_fan_in_bound_is_exact(self) -> None:
        self.assertEqual(rng.fan_in_bound(64), 0.125)
        self.assertEqual(rng.fan_in_bound(256), 0.0625)

    def test_row_major_stride_disagrees_with_a_shifted_index(self) -> None:
        weight = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        actual = rng.matmul_row([0.0, 1.0], weight, 2, 3)
        self.assertEqual(actual, [weight[1 * 3 + column] for column in range(3)])
        self.assertEqual(actual, [4.0, 5.0, 6.0])
        ignored_row = [weight[column] for column in range(3)]
        column_major = [weight[column * 2 + 1] for column in range(3)]
        self.assertEqual(ignored_row, [1.0, 2.0, 3.0])
        self.assertEqual(column_major, [2.0, 4.0, 6.0])
        self.assertNotEqual(actual, ignored_row)
        self.assertNotEqual(actual, column_major)

    def test_bias_is_added_after_the_dot_product(self) -> None:
        weight = [1.0]
        graph_order = rng.add_bias(rng.matmul_row([1.0], weight, 1, 1), [5.0])
        self.assertEqual(graph_order, [6.0])

    def test_gemm_omits_a_zero_add(self) -> None:
        weight = [2.0]
        self.assertEqual(rng.gemm_row([3.0], weight, 1, 1), [6.0])

    def test_input_period_is_251_and_rows_are_not_globally_unique(self) -> None:
        self.assertEqual(rng.input_value(0, 4), rng.input_value(251, 4))
        self.assertNotEqual(rng.input_value(0, 4), rng.input_value(1, 4))
        self.assertTrue(math.isfinite(rng.input_value(1000, 511)))
        self.assertEqual(rng.input_value(1000, 0), rng.f32(float((1000 % 251)) * rng.f32(2.0**-10)))

    def test_unknown_configuration_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            small_gen.build_artifacts("mlp_large_f32", 32, (64, 32, 8), "logits")


class GeneratedWorkloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng.require_onnx_version(onnx.__version__)
        cls.builds = {
            "mlp_small_f32": small_gen.build_artifacts(),
            "mlp_medium_f32": medium_gen.build_artifacts(),
            "gemm_f32": gemm_gen.build_artifacts(),
        }
        cls.repeated: dict[str, tuple[dict[str, bytes], dict[str, bytes]]] = {}
        for name, builder in (
            ("mlp_small_f32", small_gen.build_artifacts),
            ("mlp_medium_f32", medium_gen.build_artifacts),
            ("gemm_f32", gemm_gen.build_artifacts),
        ):
            with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
                left_dir = Path(first)
                right_dir = Path(second)
                left = builder()
                right = builder()
                small_gen.write_artifacts(left_dir, left)
                small_gen.write_artifacts(right_dir, right)
                left_files = {path: payload for path, payload in left.items()}
                right_files = {path: payload for path, payload in right.items()}
                cls.repeated[name] = (left_files, right_files)
                for relative, payload in left.items():
                    self_left = (left_dir / relative).read_bytes()
                    self_right = (right_dir / relative).read_bytes()
                    if self_left != payload or self_right != right[relative] or self_left != self_right:
                        raise AssertionError(f"{name} {relative} was not repeatable on disk")

    def test_temporary_generations_match_byte_for_byte(self) -> None:
        for name, (left, right) in self.repeated.items():
            self.assertEqual(set(left), set(right), name)
            for relative, payload in left.items():
                self.assertEqual(payload, right[relative], f"{name}:{relative}")
                self.assertEqual(payload, self.builds[name][relative], f"{name}:{relative}")

    def test_committed_artifacts_match_regeneration(self) -> None:
        for name, files in self.builds.items():
            for relative, payload in files.items():
                committed = REPO_MODELS / relative
                self.assertTrue(committed.is_file(), str(committed))
                self.assertEqual(committed.read_bytes(), payload, str(committed))

    def test_contracts(self) -> None:
        expectations = {
            "mlp_small_f32": {
                "input": "features",
                "output": "logits",
                "input_width": 32,
                "output_width": 8,
                "operators": ["MatMul", "Add", "Relu", "MatMul", "Add", "Relu", "MatMul", "Add"],
                "initializers": [("W1", [32, 64]), ("b1", [64]), ("W2", [64, 32]), ("b2", [32]), ("W3", [32, 8]), ("b3", [8])],
                "parameters": 4456,
                "bytes": 17824,
            },
            "mlp_medium_f32": {
                "input": "features",
                "output": "logits",
                "input_width": 256,
                "output_width": 16,
                "operators": ["MatMul", "Add", "Relu", "MatMul", "Add", "Relu", "MatMul", "Add"],
                "initializers": [
                    ("W1", [256, 256]),
                    ("b1", [256]),
                    ("W2", [256, 128]),
                    ("b2", [128]),
                    ("W3", [128, 16]),
                    ("b3", [16]),
                ],
                "parameters": 100752,
                "bytes": 403008,
            },
            "gemm_f32": {
                "input": "features",
                "output": "product",
                "input_width": 512,
                "output_width": 512,
                "operators": ["Gemm"],
                "initializers": [("weight", [512, 512])],
                "parameters": 262144,
                "bytes": 1048576,
            },
        }
        for name, expected in expectations.items():
            model = onnx.load_from_string(self.builds[name][f"{name}.onnx"])
            onnx.checker.check_model(model)
            self.assertEqual(model.ir_version, 10, name)
            self.assertEqual([(item.domain, item.version) for item in model.opset_import], [("", 13)], name)
            self.assertEqual(model.producer_name, "inferedge", name)
            self.assertEqual(len(model.graph.input), 1, name)
            self.assertEqual(len(model.graph.output), 1, name)
            self.assertEqual(model.graph.input[0].name, expected["input"])
            self.assertEqual(model.graph.output[0].name, expected["output"])
            self.assertEqual(dynamic_static_shape(model.graph.input[0]), ("batch", expected["input_width"]))
            self.assertEqual(
                dynamic_static_shape(model.graph.output[0]),
                ("batch", expected["output_width"]),
            )
            self.assertEqual([node.op_type for node in model.graph.node], expected["operators"], name)
            if name == "gemm_f32":
                attributes = {item.name: item for item in model.graph.node[0].attribute}
                self.assertEqual(attributes["transA"].i, 0)
                self.assertEqual(attributes["transB"].i, 0)
                self.assertEqual(attributes["alpha"].f, 1.0)
                self.assertEqual(attributes["beta"].f, 0.0)
            names_and_dims = [
                (tensor.name, list(tensor.dims)) for tensor in model.graph.initializer
            ]
            self.assertEqual(names_and_dims, expected["initializers"], name)
            raw = b"".join(initializer_bytes(tensor) for tensor in model.graph.initializer)
            self.assertEqual(len(raw), expected["bytes"], name)
            self.assertEqual(len(raw) // 4, expected["parameters"], name)
            self.assertEqual(self.builds[name][f"fixtures/{name}.weights"], raw, name)
            self.assertTrue(all(math.isfinite(value) for value in unpack_f32(raw)))
            for tensor in model.graph.initializer:
                self.assertEqual(tensor.data_location, TensorProto.DEFAULT)
                self.assertEqual(len(tensor.external_data), 0)

    def test_fixtures(self) -> None:
        widths = {
            "mlp_small_f32": (32, 8),
            "mlp_medium_f32": (256, 16),
            "gemm_f32": (512, 512),
        }
        for name, (input_width, output_width) in widths.items():
            files = self.builds[name]
            manifest = files[f"fixtures/{name}/manifest.txt"].decode("ascii")
            self.assertIn("little-endian float32", manifest)
            single = unpack_f32(files[f"fixtures/{name}/b01.input.f32"])
            single_expected = unpack_f32(files[f"fixtures/{name}/b01.expected.f32"])
            for batch_size in (1, 2, 4, 8, 16, 32):
                tag = f"b{batch_size:02d}"
                inputs = unpack_f32(files[f"fixtures/{name}/{tag}.input.f32"])
                expected = unpack_f32(files[f"fixtures/{name}/{tag}.expected.f32"])
                self.assertEqual(len(inputs), batch_size * input_width, f"{name}:{tag}")
                self.assertEqual(len(expected), batch_size * output_width, f"{name}:{tag}")
                self.assertTrue(all(math.isfinite(value) for value in inputs))
                self.assertTrue(all(math.isfinite(value) for value in expected))
                rows = [
                    tuple(inputs[row * input_width : (row + 1) * input_width])
                    for row in range(batch_size)
                ]
                self.assertEqual(len(rows), len(set(rows)), f"{name}:{tag}")
                self.assertIn(
                    f"{batch_size} {tag}.input.f32 {tag}.expected.f32 "
                    f"{input_width} {output_width} 1000",
                    manifest,
                )
            paired = unpack_f32(files[f"fixtures/{name}/b02.input.f32"])
            paired_expected = unpack_f32(files[f"fixtures/{name}/b02.expected.f32"])
            self.assertEqual(paired[:input_width], single)
            self.assertEqual(paired_expected[:output_width], single_expected)
            self.assertNotEqual(paired[:input_width], paired[input_width:])

    def test_fixture_hashes_are_stable_across_builds(self) -> None:
        for name, files in self.builds.items():
            for relative, payload in files.items():
                self.assertEqual(sha256(payload), sha256(self.repeated[name][0][relative]))


class ResNet18CommittedArtifactTest(unittest.TestCase):
    """Inspect committed ResNet-18 files. Collection does not import NumPy."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.onnx_path = REPO_MODELS / "resnet18_f32.onnx"
        cls.fixture_dir = REPO_MODELS / "fixtures" / "resnet18_f32"
        if not cls.onnx_path.is_file():
            raise unittest.SkipTest("resnet18_f32.onnx has not been generated")
        cls.model = onnx.load(str(cls.onnx_path))

    def test_onnx_contract(self) -> None:
        onnx.checker.check_model(self.model)
        self.assertEqual(self.model.ir_version, 10)
        self.assertEqual([(item.domain, item.version) for item in self.model.opset_import], [("", 13)])
        self.assertEqual(self.model.producer_name, "inferedge")
        self.assertEqual(self.model.graph.input[0].name, "images")
        self.assertEqual(self.model.graph.output[0].name, "logits")
        input_dims = self.model.graph.input[0].type.tensor_type.shape.dim
        output_dims = self.model.graph.output[0].type.tensor_type.shape.dim
        self.assertEqual(self.model.graph.input[0].type.tensor_type.elem_type, TensorProto.FLOAT)
        self.assertEqual(self.model.graph.output[0].type.tensor_type.elem_type, TensorProto.FLOAT)
        self.assertEqual(input_dims[0].dim_param, "batch")
        self.assertFalse(input_dims[0].HasField("dim_value"))
        self.assertEqual([dim.dim_value for dim in input_dims[1:]], [3, 224, 224])
        self.assertEqual(output_dims[0].dim_param, "batch")
        self.assertFalse(output_dims[0].HasField("dim_value"))
        self.assertEqual(output_dims[1].dim_value, 1000)
        operators = [node.op_type for node in self.model.graph.node]
        self.assertEqual(operators[0], "Conv")
        self.assertEqual(operators[1], "BatchNormalization")
        self.assertEqual(operators[2], "Relu")
        self.assertEqual(operators[3], "MaxPool")
        self.assertEqual(operators[-3], "GlobalAveragePool")
        self.assertEqual(operators[-2], "Flatten")
        self.assertEqual(operators[-1], "Gemm")
        self.assertEqual(operators.count("Conv"), 20)
        self.assertEqual(operators.count("BatchNormalization"), 20)
        self.assertEqual(operators.count("Add"), 8)
        self.assertEqual(operators.count("Relu"), 17)
        self.assertEqual(operators.count("MaxPool"), 1)
        self.assertEqual(operators.count("GlobalAveragePool"), 1)
        self.assertEqual(operators.count("Flatten"), 1)
        self.assertEqual(operators.count("Gemm"), 1)
        stem = {item.name: item for item in self.model.graph.node[0].attribute}
        self.assertEqual(list(stem["kernel_shape"].ints), [7, 7])
        self.assertEqual(list(stem["strides"].ints), [2, 2])
        self.assertEqual(list(stem["pads"].ints), [3, 3, 3, 3])
        pool = {item.name: item for item in self.model.graph.node[3].attribute}
        self.assertEqual(list(pool["kernel_shape"].ints), [3, 3])
        self.assertEqual(list(pool["strides"].ints), [2, 2])
        self.assertEqual(list(pool["pads"].ints), [1, 1, 1, 1])
        gemm = {item.name: item for item in self.model.graph.node[-1].attribute}
        self.assertEqual(gemm["transA"].i, 0)
        self.assertEqual(gemm["transB"].i, 0)
        flatten = {item.name: item for item in self.model.graph.node[-2].attribute}
        self.assertEqual(flatten["axis"].i, 1)
        names = [tensor.name for tensor in self.model.graph.initializer]
        self.assertIn("conv1.weight", names)
        self.assertIn("fc.weight", names)
        self.assertIn("fc.bias", names)
        self.assertIn("layer1.0.conv1.weight", names)
        self.assertIn("layer2.0.downsample.conv.weight", names)
        self.assertIn("layer3.0.downsample.conv.weight", names)
        self.assertIn("layer4.0.downsample.conv.weight", names)
        self.assertNotIn("layer1.0.downsample.conv.weight", names)
        learnable = 0
        for tensor in self.model.graph.initializer:
            self.assertEqual(tensor.data_type, TensorProto.FLOAT)
            self.assertEqual(tensor.data_location, TensorProto.DEFAULT)
            self.assertEqual(len(tensor.external_data), 0)
            count = 1
            for dim in tensor.dims:
                count *= int(dim)
            if tensor.name.endswith(".mean") or tensor.name.endswith(".var"):
                continue
            learnable += count
        self.assertEqual(learnable, 11_689_512)
        fc_weight = next(tensor for tensor in self.model.graph.initializer if tensor.name == "fc.weight")
        self.assertEqual(list(fc_weight.dims), [512, 1000])

    def test_fixture_files_and_sizes(self) -> None:
        manifest = (self.fixture_dir / "manifest.txt").read_text(encoding="ascii")
        self.assertIn("seed 20261017", manifest)
        self.assertIn("not ImageNet-pretrained", manifest)
        self.assertIn("no weight sidecar", manifest)
        self.assertIn("attempt_ids 1000..1007 only", manifest)
        for batch_size in (1, 2, 4, 8):
            tag = f"b{batch_size:02d}"
            input_path = self.fixture_dir / f"{tag}.input.f32"
            expected_path = self.fixture_dir / f"{tag}.expected.f32"
            self.assertTrue(input_path.is_file(), str(input_path))
            self.assertTrue(expected_path.is_file(), str(expected_path))
            self.assertEqual(input_path.stat().st_size, batch_size * 3 * 224 * 224 * 4)
            self.assertEqual(expected_path.stat().st_size, batch_size * 1000 * 4)
            inputs = unpack_f32(input_path.read_bytes())
            outputs = unpack_f32(expected_path.read_bytes())
            self.assertTrue(all(math.isfinite(value) for value in inputs))
            self.assertTrue(all(math.isfinite(value) for value in outputs))
            self.assertIn(
                f"{batch_size} {tag}.input.f32 {tag}.expected.f32 150528 1000 1000",
                manifest,
            )
        single = unpack_f32((self.fixture_dir / "b01.input.f32").read_bytes())
        paired = unpack_f32((self.fixture_dir / "b02.input.f32").read_bytes())
        self.assertEqual(paired[:150528], single)
        self.assertNotEqual(paired[:150528], paired[150528:])
        sidecar = REPO_MODELS / "fixtures" / "resnet18_f32.weights"
        self.assertFalse(sidecar.exists())

    def test_manifest_stable_lines_ignore_host_provenance_values(self) -> None:
        committed = (self.fixture_dir / "manifest.txt").read_text(encoding="ascii")
        stable = resnet18_manifest_stable_lines(committed)
        self.assertIn("# seed 20261017", stable)
        self.assertIn("# model resnet18_f32", stable)
        self.assertIn("# initializer_count 102", stable)
        self.assertTrue(any(line.startswith("# initializer_order ") for line in stable))
        self.assertIn("1 b01.input.f32 b01.expected.f32 150528 1000 1000", stable)
        self.assertIn("8 b08.input.f32 b08.expected.f32 150528 1000 1000", stable)
        self.assertFalse(any("python" in line for line in stable))
        drifted = committed.replace(
            "# python 3.14.7 (main, Aug  5 2026, 10:29:49) [Clang 21.0.0 (clang-2100.1.1.101)]",
            "# python 3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]",
            1,
        )
        self.assertNotEqual(committed, drifted)
        self.assertEqual(resnet18_manifest_stable_lines(drifted), stable)

    def test_manifest_rejects_unknown_or_malformed_fields(self) -> None:
        committed = (self.fixture_dir / "manifest.txt").read_text(encoding="ascii")
        with self.assertRaises(ValueError):
            resnet18_manifest_stable_lines(committed + "# extra unknown field\n")
        with self.assertRaises(ValueError):
            resnet18_manifest_stable_lines(re.sub(r"^# python .*\n", "", committed, count=1, flags=re.M))
        seeded = committed.replace("# seed 20261017", "# seed 0", 1)
        self.assertNotEqual(
            resnet18_manifest_stable_lines(seeded),
            resnet18_manifest_stable_lines(committed),
        )
        with self.assertRaises(ValueError):
            resnet18_manifest_stable_lines(committed.replace("# python ", "# ", 1))
        with self.assertRaises(ValueError):
            resnet18_manifest_stable_lines("1 b01.input.f32 b01.expected.f32 150528 1000 1000\n")


class ResNet18OperatorTest(unittest.TestCase):
    """Focused NumPy operator checks. Skipped when NumPy is absent (CI onnx venv)."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import numpy as np
        except ImportError as error:
            raise unittest.SkipTest("numpy is not installed") from error
        from models import generate_resnet18_model as resnet_gen

        cls.np = np
        cls.resnet = resnet_gen

    def test_conv_padding_stride_oihw_nchw(self) -> None:
        np = self.np
        images = np.array(
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]],
            dtype=np.float32,
        )
        kernel = np.array([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=np.float32)
        strided = self.resnet.conv2d_nchw(images, kernel, stride=2, pad=0)
        self.assertEqual(strided.shape, (1, 1, 2, 2))
        np.testing.assert_array_equal(strided[0, 0], np.array([[1.0, 3.0], [9.0, 11.0]], dtype=np.float32))
        padded = self.resnet.conv2d_nchw(images, kernel, stride=1, pad=1)
        self.assertEqual(padded.shape, (1, 1, 5, 5))
        self.assertEqual(float(padded[0, 0, 0, 0]), 0.0)
        self.assertEqual(float(padded[0, 0, 1, 1]), 1.0)
        crossed = np.zeros((1, 2, 1, 1), dtype=np.float32)
        crossed[0, 1, 0, 0] = 1.0
        oihw = np.zeros((1, 2, 1, 1), dtype=np.float32)
        oihw[0, 1, 0, 0] = 3.0
        product = self.resnet.conv2d_nchw(crossed, oihw, stride=1, pad=0)
        self.assertEqual(float(product[0, 0, 0, 0]), 3.0)

    def test_batch_norm_epsilon_is_not_identity(self) -> None:
        np = self.np
        images = np.array([[[[2.0]]]], dtype=np.float32)
        ones = np.array([1.0], dtype=np.float32)
        zeros = np.array([0.0], dtype=np.float32)
        actual = self.resnet.batch_norm_nchw(images, ones, zeros, zeros, ones)
        expected = np.float32(2.0) * self.resnet.bn_inv_std(1.0, 1e-5)
        self.assertTrue(math.isfinite(float(actual[0, 0, 0, 0])))
        self.assertNotEqual(float(actual[0, 0, 0, 0]), 2.0)
        self.assertEqual(float(actual[0, 0, 0, 0]), float(expected))

    def test_identity_and_projection_residual_add(self) -> None:
        np = self.np
        left = np.array([[[[1.0, -2.0]]]], dtype=np.float32)
        right = np.array([[[[3.0, 4.0]]]], dtype=np.float32)
        added = left + right
        np.testing.assert_array_equal(added, np.array([[[[4.0, 2.0]]]], dtype=np.float32))
        activated = self.resnet.relu_nchw(added)
        np.testing.assert_array_equal(activated, np.array([[[[4.0, 2.0]]]], dtype=np.float32))
        source = np.ones((1, 2, 4, 4), dtype=np.float32)
        proj = np.zeros((3, 2, 1, 1), dtype=np.float32)
        proj[0, 0, 0, 0] = 1.0
        projected = self.resnet.conv2d_nchw(source, proj, stride=2, pad=0)
        self.assertEqual(projected.shape, (1, 3, 2, 2))
        self.assertEqual(float(projected[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(projected[0, 1, 0, 0]), 0.0)

    def test_maxpool_and_global_average_pool(self) -> None:
        np = self.np
        images = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], dtype=np.float32)
        pooled = self.resnet.maxpool2d_nchw(images, kernel=2, stride=1, pad=0)
        np.testing.assert_array_equal(pooled[0, 0], np.array([[5.0, 6.0], [8.0, 9.0]], dtype=np.float32))
        gap = self.resnet.global_average_pool_nchw(images)
        expected = np.float32(45.0) * np.float32(1.0 / 9.0)
        self.assertEqual(float(gap[0, 0, 0, 0]), float(expected))

    def test_classifier_ordering_and_finite_inputs(self) -> None:
        np = self.np
        features = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        weight = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32)
        bias = np.array([1.0, 2.0], dtype=np.float32)
        logits = self.resnet.gemm_logits(features, weight, bias)
        np.testing.assert_array_equal(logits, np.array([[11.0, 22.0], [31.0, 42.0]], dtype=np.float32))
        image = self.resnet.image_from_attempt(1000)
        self.assertEqual(image.shape, (1, 3, 224, 224))
        self.assertTrue(bool(np.all(np.isfinite(image))))
        first = float(image.reshape(-1)[0])
        self.assertEqual(first, float(np.float32((1000 % 251) & 0xFFFFFF) * np.float32(2.0**-24)))
        self.assertEqual(self.resnet.LEARNABLE_PARAMETERS, 11_689_512)
        self.assertEqual(self.resnet.SEED, 20261017)

    def test_committed_artifacts_match_regeneration_when_numpy_present(self) -> None:
        files = self.resnet.build_artifacts()
        self.assertEqual(set(files), RESNET18_REGENERATED_RELATIVES)
        for relative, payload in files.items():
            committed = REPO_MODELS / relative
            self.assertTrue(committed.is_file(), str(committed))
            if relative == "fixtures/resnet18_f32/manifest.txt":
                committed_text = committed.read_text(encoding="ascii")
                regenerated_text = payload.decode("ascii")
                self.assertEqual(
                    resnet18_manifest_stable_lines(committed_text),
                    resnet18_manifest_stable_lines(regenerated_text),
                    relative,
                )
                continue
            self.assertEqual(committed.read_bytes(), payload, relative)
        self.assertEqual(self.resnet.LEARNABLE_PARAMETERS, 11_689_512)


if __name__ == "__main__":
    unittest.main()
