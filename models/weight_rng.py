"""Deterministic float32 weights and a scalar reference for dynamic workloads.

Unsigned 64-bit overflow is emulated by masking with 2**64 - 1 after every
addition and multiplication. Python integers do not wrap on their own.

Float32 values are rounded with struct.pack("<f", number), which is
IEEE-754 round-to-nearest-even, and then unpacked. The "<f" format is
little-endian on the wire, so the stored bytes do not depend on the host
byte order. A Python float returned by unpack is exactly the float32 value.

This host can repeat a generation. That is not, by itself, a claim that
another libm produces the same sqrt for a non-square fan-in.
"""

from __future__ import annotations

import math
import struct

REQUIRED_ONNX_VERSION = "1.22.0"
SEED = 20261015
MASK64 = (1 << 64) - 1
UINT32_MASK = (1 << 32) - 1
GAMMA = 0x9E3779B97F4A7C15
MIX_1 = 0xBF58476D1CE4E5B9
MIX_2 = 0x94D049BB133111EB
ATTEMPT_PERIOD = 251

MLP_INITIALIZER_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")
GEMM_INITIALIZER_NAMES = ("weight",)


def require_onnx_version(actual: str) -> None:
    """Reject every ONNX version other than the repository pin."""
    if actual != REQUIRED_ONNX_VERSION:
        raise SystemExit(
            f"onnx {REQUIRED_ONNX_VERSION} is required, found {actual}"
        )


def f32(value: float) -> float:
    """Round one real value to float32, then return that exact value."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def f32_le(value: float) -> bytes:
    """Little-endian IEEE-754 bytes of a float32-rounded value."""
    return struct.pack("<f", f32(value))


def f32_mul(left: float, right: float) -> float:
    """Float32 multiply.

    Both operands are float32. Their product is exact in float64, and the
    following conversion is the float32 rounding of that product.
    """
    return f32(f32(left) * f32(right))


def f32_add(left: float, right: float) -> float:
    """Float32 add.

    The sum of two float32 values is exact in float64. The following
    conversion is the float32 rounding of that sum.
    """
    return f32(f32(left) + f32(right))


def f32_sub(left: float, right: float) -> float:
    """Float32 subtract. Same exactness argument as f32_add."""
    return f32(f32(left) - f32(right))


class SplitMix64:
    """Steele's SplitMix64. The seed is the state before the first draw."""

    def __init__(self, seed: int = SEED) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an int")
        if seed < 0 or seed > MASK64:
            raise ValueError("seed does not fit in uint64")
        self.state = seed

    def next_u64(self) -> int:
        self.state = (self.state + GAMMA) & MASK64
        mixed = self.state
        mixed = ((mixed ^ (mixed >> 30)) * MIX_1) & MASK64
        mixed = ((mixed ^ (mixed >> 27)) * MIX_2) & MASK64
        return (mixed ^ (mixed >> 31)) & MASK64


def high_24_bits(draw: int) -> int:
    if draw < 0 or draw > MASK64:
        raise ValueError("draw does not fit in uint64")
    return (draw >> 40) & 0xFFFFFF


def unit_interval(draw: int) -> float:
    """Map a draw to float32(u) * float32(2**-24), with u in 0 .. 2**24-1.

    Integers in that range are exact in float32. 2**-24 is exact. Their
    product is exact, so the float32 rounding does not change it.
    """
    unit = high_24_bits(draw)
    return f32_mul(f32(float(unit)), f32(2.0**-24))


def fan_in_bound(fan_in: int) -> float:
    """One float32 rounding of the float64 value 1/sqrt(fan_in).

    math.sqrt is the platform's correctly rounded float64 square root.
    The division happens in float64. There is one conversion to float32
    after that division, not a separate float32 square root and float32
    reciprocal.
    """
    if not isinstance(fan_in, int) or isinstance(fan_in, bool) or fan_in <= 0:
        raise ValueError("fan_in must be a positive int")
    return f32(1.0 / math.sqrt(fan_in))


def weight_from_draw(draw: int, bound: float) -> float:
    """Scale a draw into float32 [-bound, bound).

    Rounding points, in order:
    unit = float32(float32(u) * float32(2**-24))
    two_bound = float32(float32(2) * bound)
    product = float32(unit * two_bound)
    result = float32(product - bound)
    """
    unit = unit_interval(draw)
    two_bound = f32_mul(f32(2.0), bound)
    return f32_sub(f32_mul(unit, two_bound), bound)


def draw_vector(rng: SplitMix64, count: int, fan_in: int) -> list[float]:
    if count < 0:
        raise ValueError("count must be non-negative")
    bound = fan_in_bound(fan_in)
    return [weight_from_draw(rng.next_u64(), bound) for _ in range(count)]


def pack_f32(values: list[float]) -> bytes:
    return b"".join(f32_le(value) for value in values)


def input_code(attempt_id: int, index: int) -> int:
    """uint32(attempt_id % 251 + index). The residue repeats every 251 ids."""
    if (
        not isinstance(attempt_id, int)
        or isinstance(attempt_id, bool)
        or attempt_id < 0
        or not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
    ):
        raise ValueError("attempt_id and index must be non-negative ints")
    return ((attempt_id % ATTEMPT_PERIOD) + index) & UINT32_MASK


def input_value(attempt_id: int, index: int) -> float:
    """float32(uint32(attempt_id % 251 + index)) * 0x1p-10.

    The code fits in the float32 mantissa for every width in this milestone.
    0x1p-10 is the exact power of two 2**-10. The product is therefore exact
    and finite. Rows that share a residue of attempt_id are not distinct.
    """
    return f32_mul(f32(float(input_code(attempt_id, index))), f32(2.0**-10))


def input_row(attempt_id: int, width: int) -> list[float]:
    if width <= 0:
        raise ValueError("width must be positive")
    return [input_value(attempt_id, index) for index in range(width)]


def input_matrix(attempt_ids: list[int], width: int) -> list[list[float]]:
    return [input_row(attempt_id, width) for attempt_id in attempt_ids]


def matmul_row(row: list[float], weight: list[float], fan_in: int, fan_out: int) -> list[float]:
    """Row-major matmul. Flattened weight index is k * fan_out + n.

    The accumulator starts at +0. Each product is rounded to float32, then
    added into the accumulator with another float32 rounding. k runs from
    0 upward. This is the MatMul node, not a bias-initialized dot product.
    """
    if len(row) != fan_in:
        raise ValueError("row width does not match fan_in")
    if len(weight) != fan_in * fan_out:
        raise ValueError("weight size does not match the matrix shape")
    output: list[float] = []
    for column in range(fan_out):
        accumulator = f32(0.0)
        for reduction in range(fan_in):
            product = f32_mul(row[reduction], weight[reduction * fan_out + column])
            accumulator = f32_add(accumulator, product)
        output.append(accumulator)
    return output


def add_bias(row: list[float], bias: list[float]) -> list[float]:
    """Elementwise float32 add after MatMul. This is the Add node."""
    if len(row) != len(bias):
        raise ValueError("bias width does not match the row")
    return [f32_add(value, bias[index]) for index, value in enumerate(row)]


def relu(row: list[float]) -> list[float]:
    """ReLU after Add. A non-positive finite value becomes +0."""
    activated: list[float] = []
    for value in row:
        if not math.isfinite(value):
            raise ValueError("non-finite activation")
        activated.append(value if value > 0.0 else f32(0.0))
    return activated


def mlp_row(
    row: list[float],
    layers: list[tuple[list[float], list[float], int, int, bool]],
) -> list[float]:
    """Apply MatMul, Add, and an optional ReLU for each layer, in order.

    Each layer tuple is (weight, bias, fan_in, fan_out, apply_relu).
    The last layer of the approved MLPs sets apply_relu False, matching
    MatMul, Add with no following ReLU.
    """
    current = row
    for weight, bias, fan_in, fan_out, apply_relu in layers:
        current = matmul_row(current, weight, fan_in, fan_out)
        current = add_bias(current, bias)
        if apply_relu:
            current = relu(current)
    return current


def gemm_row(row: list[float], weight: list[float], fan_in: int, fan_out: int) -> list[float]:
    """Gemm with alpha 1, beta 0, and no C tensor.

    The beta * C term is omitted. Adding float32 0 is not performed.
    The result is the MatMul accumulation above.
    """
    return matmul_row(row, weight, fan_in, fan_out)


def assert_finite(values: list[float], label: str) -> None:
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(f"{label}[{index}] is not finite")


FIXTURE_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
FIXTURE_ATTEMPT_BASE = 1000


def fixture_attempt_ids(batch_size: int, base: int = FIXTURE_ATTEMPT_BASE) -> list[int]:
    if batch_size not in FIXTURE_BATCH_SIZES:
        raise ValueError(f"unsupported fixture batch size {batch_size}")
    return [base + row for row in range(batch_size)]


def manifest_text(model_name: str, input_width: int, output_width: int) -> str:
    """ASCII manifest. Widths are repeated so a C++ reader need not parse JSON."""
    lines = [
        "# inferedge float32 fixtures",
        "# little-endian float32 values, no file header, no padding",
        "# batch input_file expected_file input_width output_width attempt_base",
        f"# model {model_name}",
    ]
    for batch_size in FIXTURE_BATCH_SIZES:
        tag = f"b{batch_size:02d}"
        lines.append(
            f"{batch_size} {tag}.input.f32 {tag}.expected.f32 "
            f"{input_width} {output_width} {FIXTURE_ATTEMPT_BASE}"
        )
    lines.append("")
    return "\n".join(lines)


def rows_are_distinct(rows: list[list[float]]) -> bool:
    encoded = [tuple(row) for row in rows]
    return len(encoded) == len(set(encoded))
