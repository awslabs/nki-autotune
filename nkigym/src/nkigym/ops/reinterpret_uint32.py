"""Reinterpret a binary32 tile through one native integer tensor copy."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import NKIOp, _operand_role
from nkigym.ops.float32_fma import finish_binary32_sqrt


class NKIReinterpretUInt32(NKIOp):
    """Copy float32 storage as uint32 bit patterns without numeric conversion."""

    NAME: ClassVar[str] = "tensor_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf"})}
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"float32"})}
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"src": "float32"}
    REINTERPRET_INPUT_DTYPES: ClassVar[dict[str, str]] = {"src": "uint32"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_DTYPE: ClassVar[str | None] = "uint32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "uint32"

    def __init__(self) -> None:
        """Select bit-accurate Vector Engine copy."""
        super().__init__(engine="vector")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require float32 storage in SBUF."""
        if _operand_role(kwargs["src"]) not in {None, "sbuf"} or np.asarray(kwargs["src"]).dtype != np.float32:
            raise TypeError("NKIReinterpretUInt32 requires SBUF float32")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Copy the input's exact binary32 representation."""
        return np.asarray(kwargs["src"]).view(np.uint32).copy()


def emit_binary32_parts(emit: TorchArithmetic, source: str, positive_normal: bool = False) -> tuple[str, str, str, str]:
    """Extract bits, normalized integer significand, exponent, and raw exponent.

    Integer significands are positive normal FP32 values after zero is
    replaced temporarily. Their own exponent determines the normalization
    shift; replacing it with 150 produces an exact 24-bit integer significand.
    No floating arithmetic consumes the original subnormal input.
    """
    op = emit.binary
    bits = emit.cast("NKIReinterpretUInt32", source)
    field = emit.cast("NKIFloat32Cast", op("right_shift", op("bitwise_and", bits, 0x7FFFFFFF), 23))
    if positive_normal:
        significand = emit.cast("NKIFloat32Cast", op("bitwise_or", op("bitwise_and", bits, 0x7FFFFF), 0x800000))
        return bits, significand, op("subtract", field, 127), field
    fraction = emit.cast("NKIFloat32Cast", op("bitwise_and", bits, 0x7FFFFF))
    significand = op("add", fraction, op("multiply", op("greater_equal", field, 1), 8388608))
    significand = op("add", significand, op("multiply", op("equal", significand, 0), 8388608))
    significand_bits = emit.cast("NKIReinterpretUInt32", significand)
    significand_field = emit.cast("NKIFloat32Cast", op("right_shift", significand_bits, 23))
    exponent = op("subtract", op("add", op("maximum", field, 1), significand_field), 277)
    normalized = op("bitwise_or", op("bitwise_and", significand_bits, 0x7FFFFF), 150 << 23)
    significand = emit.cast("NKIReinterpretFloat32", normalized)
    return bits, significand, exponent, field


def emit_binary32_sqrt(
    source: TorchValue, name: str, body: list[str], imports: set[str], positive_normal: bool = False
) -> TorchValue:
    """Emit correctly rounded square roots using a refined native estimate.

    One Newton iteration bounds the integer root estimate to one unit.
    Column inputs use a single-rounded FMA residual; other shapes retain
    the exact modulo-word square. The residual selects the correctly rounded root. Final bit assembly
    preserves zero signs and NaN payloads.
    """
    emit = TorchArithmetic(name, body, imports)
    op, integer = emit.binary, emit.integer
    data = source.name if source.storage_dtype == "float32" else emit.cast("NKIFloat32Cast", source.name)
    bits, significand, exponent, field = emit_binary32_parts(emit, data, positive_normal)
    biased = emit.cast("NKIUInt32Cast", op("add", exponent, 254))
    parity = op("bitwise_and", biased, 1)
    exponent_bits = op("left_shift", op("right_shift", biased, 1), 23)
    zero = op("bitwise_and", bits, 0)
    parity_float = emit.cast("NKIFloat32Cast", parity)
    scaled = op("multiply", significand, op("add", parity_float, 1))
    estimate = emit.emit("NKIActivation", f"data={scaled}", "op='sqrt', scale=8388608.0")
    reciprocal = emit.emit("NKIReciprocal", f"data={estimate}")
    radicand = emit.emit("NKIActivation", f"data={scaled}", "op='copy', scale=8388608.0")
    quotient = op("multiply", radicand, reciprocal)
    combined = op("add", estimate, quotient)
    refined = emit.emit("NKIActivation", f"data={combined}", "op='copy', scale=0.5")
    root = emit.cast("NKIUInt32Cast", refined)
    if len(source.shape) == 1 or source.shape[1] == 1:
        rounded = emit.cast("NKIFloat32Cast", root)
        negative_root = op("multiply", rounded, -1.0)
        remainder = emit.emit("NKIFloat32FMA", f"data={rounded}, scale={negative_root}, bias={radicand}")
        increment = emit.cast("NKIUInt32Cast", op("greater", remainder, rounded))
        decrement = emit.cast("NKIUInt32Cast", emit.inverse(op("greater", remainder, negative_root)))
    else:
        mantissa = emit.cast("NKIUInt32Cast", significand)
        low = op("left_shift", mantissa, 23)
        low = emit.select(parity, op("left_shift", low, 1), low)
        one = op("bitwise_or", zero, 1)
        upper = emit.cast("NKIFloat32Cast", op("right_shift", root, 12))
        lower = emit.cast("NKIFloat32Cast", op("bitwise_and", root, 4095))
        upper_square = emit.cast("NKIUInt32Cast", op("multiply", upper, upper))
        lower_square = emit.cast("NKIUInt32Cast", op("multiply", lower, lower))
        cross = emit.cast("NKIUInt32Cast", op("multiply", upper, lower))
        square = integer("add", op("left_shift", upper_square, 24), op("left_shift", cross, 13))
        square = integer("add", square, lower_square)
        remainder = integer("subtract", low, square)
        upward = integer("subtract", integer("subtract", remainder, root), one)
        downward = integer("subtract", integer("add", remainder, root), one)
        increment = op("bitwise_xor", op("right_shift", upward, 31), 1)
        decrement = op("right_shift", downward, 31)
    root = integer("subtract", integer("add", root, increment), decrement)
    fraction = integer("subtract", root, op("bitwise_or", zero, 0x800000))
    result = integer("add", exponent_bits, fraction)
    result = finish_binary32_sqrt(emit, result, bits, field, zero, positive_normal)
    imports.add("NKIReinterpretFloat32")
    body.append(f"{name} = NKIReinterpretFloat32()(src={result})")
    return TorchValue(name, source.shape, source.transposed, storage_dtype="float32")
