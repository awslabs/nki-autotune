"""Single-rounded binary32 multiply-add through native activation."""

from collections.abc import Callable
from ctypes import CDLL, CFUNCTYPE, c_float
from typing import Any, ClassVar

import numpy as np
from torch.fx import Node

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.ops.base import NKIOp, _operand_role

_FMA = CFUNCTYPE(c_float, c_float, c_float, c_float)(("fmaf", CDLL("libm.so.6")))


class NKIFloat32FMA(NKIOp):
    """Compute one FP32 multiply-add per partition with a single rounding."""

    NAME: ClassVar[str] = "activation"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "data": ("P", "F"),
        "scale": ("P", "F"),
        "bias": ("P", "F"),
        "dst": ("P", "F"),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "scale", "bias"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {
        "data": frozenset({"sbuf", "psum"}),
        "scale": frozenset({"sbuf"}),
        "bias": frozenset({"sbuf"}),
    }
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {operand: "float32" for operand in INPUT_OPERANDS}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": 1}
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self) -> None:
        """Select the activation engine's fused affine operation."""
        super().__init__(op="copy")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require compatible on-chip binary32 columns."""
        shape = np.asarray(kwargs["data"]).shape
        if len(shape) not in {1, 2} or len(shape) == 2 and shape[1] != 1:
            raise ValueError("NKIFloat32FMA requires one value per partition")
        for operand in self.INPUT_OPERANDS:
            role = _operand_role(kwargs[operand])
            if role is not None and role not in self.INPUT_LOCATIONS[operand]:
                raise TypeError(f"NKIFloat32FMA.{operand} requires {self.INPUT_LOCATIONS[operand]}")
            if np.asarray(kwargs[operand]).dtype != np.float32:
                raise TypeError(f"NKIFloat32FMA.{operand} requires float32")
            if np.asarray(kwargs[operand]).shape != shape:
                raise ValueError("NKIFloat32FMA operands must have matching shapes")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Use the platform's correctly rounded binary32 FMA for CPU evaluation."""
        data = np.asarray(kwargs["data"], dtype=np.float32)
        operands = [data]
        for name in ("scale", "bias"):
            value = np.asarray(kwargs[name], dtype=np.float32)
            operands.append(value[..., None] if value.ndim + 1 == data.ndim else value)
        return np.asarray(np.frompyfunc(_FMA, 3, 1)(*operands), dtype=np.float32)


__all__ = ["NKIFloat32FMA"]


def positive_normal_sqrt_input(node: Node, fp32: Callable[[Node], bool]) -> bool:
    """Prove a nonnegative expression plus an epsilon in the normal FP32 range."""
    source = node.args[0] if node.args else None
    if not isinstance(source, Node) or _operation(source) not in {"add", "iadd"} or len(source.args) != 2:
        return False
    if not fp32(source) or source.kwargs.get("alpha", 1) != 1:
        return False
    return any(
        isinstance(constant, (int, float))
        and 2.0**-126 <= constant <= float(np.finfo(np.float32).max)
        and _nonnegative(value, fp32)
        for value, constant in (source.args, source.args[::-1])
    )


def _operation(node: Node) -> str:
    """Return the normalized FX operation name."""
    return str(getattr(node.target, "__name__", node.target)).removeprefix("wrapped_")


def _nonnegative(value: object, fp32: Callable[[Node], bool]) -> bool:
    """Recognize expressions that cannot produce a negative finite value."""
    if not isinstance(value, Node):
        return isinstance(value, (int, float)) and value >= 0
    if not fp32(value):
        return False
    operation = _operation(value)
    if operation in {"square", "abs", "absolute"} or operation == "var" and value.kwargs.get("correction") == 0:
        return True
    if operation in {"sum", "mean", "reshape", "view", "contiguous", "detach", "unsqueeze", "squeeze", "float"}:
        return bool(value.args) and _nonnegative(value.args[0], fp32)
    return bool(
        operation in {"truediv", "div", "true_divide"}
        and len(value.args) == 2
        and isinstance(value.args[1], (int, float))
        and value.args[1] > 0
        and _nonnegative(value.args[0], fp32)
    )


def finish_binary32_sqrt(
    emit: TorchArithmetic, result: str, bits: str, field: str, zero: str, positive_normal: bool
) -> str:
    """Preserve exceptional encodings after exact square-root rounding."""
    op, integer = emit.binary, emit.integer
    if not positive_normal:
        negative = op("right_shift", bits, 31)
        nan = op("bitwise_or", zero, 0xFFC00000)
        result = emit.select(negative, nan, result)
        absolute = op("bitwise_and", bits, 0x7FFFFFFF)
        nonzero = op("right_shift", op("bitwise_or", absolute, integer("subtract", zero, absolute)), 31)
        result = emit.select(op("bitwise_xor", nonzero, 1), bits, result)
        payload = op("bitwise_and", bits, 0x7FFFFF)
        nonfinite = emit.select(negative, nan, bits)
    else:
        payload = op("bitwise_and", bits, 0x7FFFFF)
        nonfinite = bits
    nonzero = op("right_shift", op("bitwise_or", payload, integer("subtract", zero, payload)), 31)
    nonfinite = emit.select(nonzero, op("bitwise_or", bits, 0x400000), nonfinite)
    return emit.select(op("equal", field, 255), nonfinite, result)
