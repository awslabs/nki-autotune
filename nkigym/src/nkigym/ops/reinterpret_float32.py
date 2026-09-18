"""Reinterpret packed binary32 fields through one native float32 tensor copy."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.ops.base import NKIOp, _operand_role


class NKIReinterpretFloat32(NKIOp):
    """Copy uint32 storage as float32 bit patterns without numeric conversion."""

    NAME: ClassVar[str] = "tensor_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf"})}
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"uint32"})}
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"src": "uint32"}
    REINTERPRET_INPUT_DTYPES: ClassVar[dict[str, str]] = {"src": "float32"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"

    def __init__(self) -> None:
        """Select bit-accurate Vector Engine copy."""
        super().__init__(engine="vector")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require uint32 storage in SBUF."""
        if _operand_role(kwargs["src"]) not in {None, "sbuf"} or np.asarray(kwargs["src"]).dtype != np.uint32:
            raise TypeError("NKIReinterpretFloat32 requires SBUF uint32")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Copy the packed binary32 representation."""
        return np.asarray(kwargs["src"]).view(np.float32).copy()


def emit_binary32_round(emit: TorchArithmetic, quotient: str, remainder: str, exponent: str, divisor: int) -> str:
    """Round an exact quotient/remainder once onto the binary32 output grid.

    For normal output, compare twice the remainder with the divisor.
    For subnormal output, retain the discarded quotient bits and a nonzero
    remainder flag, then round directly to the destination precision.
    This avoids double rounding at underflow and preserves even ties.
    """
    op = emit.binary
    qi = emit.cast("NKIUInt32Cast", quotient)
    parity = emit.cast("NKIFloat32Cast", op("bitwise_and", qi, 1))
    twice = op("multiply", remainder, 2)
    increment = op("maximum", op("greater", twice, divisor), op("multiply", op("equal", twice, divisor), parity))
    rounded = op("add", quotient, increment)
    normal_exp = op("add", op("add", exponent, 127), op("equal", rounded, 16777216))
    fraction = emit.cast("NKIFloat32Cast", op("bitwise_and", emit.cast("NKIUInt32Cast", rounded), 0x7FFFFF))
    fraction = emit.cast("NKIUInt32Cast", op("multiply", fraction, op("less", normal_exp, 255)))
    field = emit.cast("NKIUInt32Cast", op("minimum", op("maximum", normal_exp, 0), 255))
    normal = op("bitwise_or", op("left_shift", field, 23), fraction)
    shift = op("minimum", op("maximum", op("subtract", op("multiply", exponent, -1), 126), 1), 24)
    shift = emit.cast("NKIUInt32Cast", shift)
    truncated = op("right_shift", qi, shift)
    one = op("bitwise_or", op("bitwise_and", qi, 0), 1)
    unit = op("left_shift", one, shift)
    half = emit.cast("NKIFloat32Cast", op("right_shift", unit, 1))
    mask = emit.cast("NKIUInt32Cast", op("subtract", emit.cast("NKIFloat32Cast", unit), 1))
    discarded = emit.cast("NKIFloat32Cast", op("bitwise_and", qi, mask))
    odd = emit.cast("NKIFloat32Cast", op("bitwise_and", truncated, 1))
    tie = op("multiply", op("equal", discarded, half), op("maximum", op("greater", remainder, 0), odd))
    increment = op("maximum", op("greater", discarded, half), tie)
    subnormal = op("add", emit.cast("NKIFloat32Cast", truncated), increment)
    subnormal = emit.cast("NKIUInt32Cast", op("multiply", subnormal, op("greater_equal", exponent, -150)))
    return emit.select(op("greater_equal", exponent, -126), normal, subnormal)


def emit_binary32_sign(emit: TorchArithmetic, magnitude: str, bits: str, field: str, negative: bool) -> str:
    """Restore signed zero, overflow signs, infinities, and quiet NaN payloads."""
    op = emit.binary
    zero = op("bitwise_and", bits, 0)
    absolute = emit.cast("NKIFloat32Cast", op("bitwise_and", bits, 0x7FFFFFFF))
    magnitude = emit.select(op("equal", absolute, 0), zero, magnitude)
    sign = op("bitwise_xor", op("bitwise_and", bits, 0x80000000), 0x80000000 if negative else 0)
    result = op("bitwise_or", magnitude, sign)
    fraction = emit.cast("NKIFloat32Cast", op("bitwise_and", bits, 0x7FFFFF))
    infinity = op("bitwise_xor", bits, 0x80000000 if negative else 0)
    nonfinite = emit.select(op("greater", fraction, 0), op("bitwise_or", bits, 0x400000), infinity)
    result = emit.select(op("equal", field, 255), nonfinite, result)
    return result
