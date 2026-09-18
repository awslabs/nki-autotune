"""Float32 activation scaling and correctly rounded constant division emission."""

import math
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import NKIOp, PointwiseContract, _operand_role
from nkigym.ops.reinterpret_float32 import emit_binary32_round, emit_binary32_sign
from nkigym.ops.reinterpret_uint32 import emit_binary32_parts


def emit_tensor_scalar_or_divide(
    source: TorchValue,
    operation: str,
    operand: str,
    reverse: bool,
    name: str,
    body: list[str],
    imports: set[str],
    native: bool,
) -> TorchValue:
    """Emit scalar math, using reciprocal scaling only under the native arithmetic contract.

    Normalized significands are integers below 2**24. Each of 23 restoring
    steps doubles the remainder, subtracts the divisor when needed, and
    appends one quotient bit. All these operations are exact in float32.
    The exact remainder then determines nearest-even rounding, including
    subnormal outputs, without a rounded reciprocal or a fused operation.

    Scalars are interpreted as binary32; zero and nonfinite divisors are
    rejected. Infinite numerators retain the quotient sign. NaN numerators
    are quieted while retaining their sign and payload.
    """
    target = TorchValue(name, source.shape, source.transposed, storage_dtype="float32")
    if operation != "divide":
        target = TorchValue(name, source.shape, source.transposed, storage_dtype=source.storage_dtype)
        reverse_argument = ", reverse0=True" if reverse and operation == "subtract" else ""
        imports.add("NKITensorScalar")
        body.append(
            f'{target.name} = NKITensorScalar(op0="{operation}"{reverse_argument})'
            f"(data={source.name}, operand0={operand})"
        )
        return target
    if reverse:
        raise ValueError("Torch scalar-over-tensor division is unsupported")
    divisor = float(np.float32(float(operand)))
    if divisor == 0.0:
        raise ValueError("Torch tensor division requires a nonzero scalar")
    if not math.isfinite(divisor):
        raise ValueError("Torch tensor division requires a finite binary32 scalar")
    if native:
        scale = float(np.float32(1.0) / np.float32(divisor))
        imports.add("NKIFloat32Scale")
        body.append(f"{target.name} = NKIFloat32Scale(scale={scale!r})(data={source.name})")
        return target
    emit = TorchArithmetic(name, body, imports)
    data = source.name if source.storage_dtype == "float32" else emit.cast("NKIFloat32Cast", source.name)
    bits, significand, exponent, field = emit_binary32_parts(emit, data)
    mantissa, power = math.frexp(abs(divisor))
    denominator = int(mantissa * 2**24)
    op = emit.binary
    below_one = op("less", significand, denominator)
    remainder = op("subtract", op("multiply", significand, op("add", below_one, 1)), denominator)
    exponent = op("subtract", op("subtract", exponent, power - 1), below_one)
    quotient = op("greater_equal", significand, 0)
    for _ in range(23):
        twice = op("multiply", remainder, 2)
        bit = op("greater_equal", twice, denominator)
        remainder = op("subtract", twice, op("multiply", bit, denominator))
        quotient = op("add", op("multiply", quotient, 2), bit)
    magnitude = emit_binary32_round(emit, quotient, remainder, exponent, denominator)
    result = emit_binary32_sign(emit, magnitude, bits, field, divisor < 0)
    body.append(f"{target.name} = NKIReinterpretFloat32()(src={result})")
    imports.add("NKIReinterpretFloat32")
    return target


class NKIFloat32Scale(NKIOp):
    """Scale one on-chip tensor into a float32 activation destination."""

    NAME: ClassVar[str] = "activation"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"data": frozenset({"sbuf", "psum"})}
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {"data": frozenset({"bfloat16", "float16", "float32"})}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"

    def __init__(self, scale: float) -> None:
        """Configure a native copy activation with one scalar scale."""
        super().__init__(op="copy", scale=scale)

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PointwiseContract:
        """Return the activation-scale pointwise contract."""
        return PointwiseContract(
            operator="copy", input_operands=("data",), output_operand="dst", scale=float(kwargs["scale"])
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one on-chip input."""
        role = _operand_role(kwargs["data"])
        if role is not None and role not in {"sbuf", "psum"}:
            raise TypeError(f"NKIFloat32Scale(data=<role={role}>) expects sbuf or psum")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the float32 scaled values."""
        return np.asarray(kwargs["data"], dtype=np.float32) * float(kwargs["scale"])


__all__ = ["NKIFloat32Scale"]
