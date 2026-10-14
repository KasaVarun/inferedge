"""Deterministic generation checks for the dynamic MLP and GEMM workloads."""

from __future__ import annotations

import hashlib
import math
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


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


if __name__ == "__main__":
    unittest.main()
