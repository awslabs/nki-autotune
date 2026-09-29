"""Integer bit-pattern operations through native ``tensor_scalar``."""

import operator
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import numpy as np
import torch
from torch.fx import Node

from nkigym.ops.base import NKIOp, PointwiseContract, _operand_role

_BITWISE = {
    "bitwise_and": np.bitwise_and,
    "bitwise_or": np.bitwise_or,
    "bitwise_xor": np.bitwise_xor,
    "left_shift": np.left_shift,
    "right_shift": np.right_shift,
}


def bounded_index_quotient(
    call: Callable[..., Node], cast_op: Callable[..., object], indices: Node, width: int, divisor: int
) -> tuple[Node, Node]:
    """Compute unsigned quotient and exact FP32 values for bounded indices."""
    if not 0 < divisor <= width <= 2**24:
        raise ValueError("index quotient requires an exact FP32 index bound")
    shape = (1, width)
    if divisor & (divisor - 1) == 0:
        quotient = call(operator.rshift, (indices, divisor.bit_length() - 1), shape)
        return quotient, call(cast_op, (quotient, "float32"), shape)
    positions = call(cast_op, (indices, "float32"), shape)
    scaled = call(operator.mul, (positions, 1.0 / divisor), shape)
    rounded = call(cast_op, (call(cast_op, (scaled, "uint32"), shape), "float32"), shape)
    excess = call(operator.sub, (call(operator.mul, (rounded, divisor), shape), positions), shape)
    correction = call(torch.clamp, (excess,), shape, min=0.0, max=1.0)
    values = call(operator.sub, (rounded, correction), shape)
    return call(cast_op, (values, "uint32"), shape), values


class NKIBitwiseScalar(NKIOp):
    """Apply one or two native bitwise operators to uint32 data."""

    NAME: ClassVar[str] = "tensor_scalar"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"data": frozenset({"sbuf"})}
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {"data": frozenset({"uint32"})}
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"data": "uint32"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_DTYPE: ClassVar[str | None] = "uint32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "uint32"

    @staticmethod
    def native_parameters(kwargs: dict[str, Any], operands: frozenset[str]) -> tuple[str, dict[str, Any]]:
        """Encode logical right shifts using the native shift-and-mask pair."""
        if kwargs.get("op0") == "right_shift":
            shift = int(kwargs["operand0"])
            if shift < 0 or kwargs.get("op1") not in {None, "bitwise_and"}:
                raise ValueError("uint32 right shift requires a nonnegative count and an optional bitwise_and")
            mask = int(kwargs["operand1"]) if kwargs.get("op1") == "bitwise_and" else 0xFFFFFFFF
            kwargs.update(
                op0="right_shift" if shift < 32 else "bitwise_and",
                operand0=shift if shift < 32 else 0,
                op1="bitwise_and",
                operand1=(0xFFFFFFFF >> shift) & mask,
            )
        return "tensor_scalar", kwargs

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PointwiseContract | None:
        """Describe a single mapping; keep compound integer expressions opaque."""
        return (
            PointwiseContract(operator=str(kwargs["op0"]), input_operands=("data",), output_operand="dst")
            if kwargs.get("op1") is None
            else None
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require SBUF uint32 input and a supported integer scalar operator."""
        if _operand_role(kwargs["data"]) not in {None, "sbuf"}:
            raise TypeError("NKIBitwiseScalar.data requires SBUF")
        if np.asarray(kwargs["data"]).dtype != np.uint32 or kwargs["op0"] not in _BITWISE:
            raise TypeError("NKIBitwiseScalar requires uint32 data and a bitwise operator")
        if kwargs.get("op1") is not None and (
            kwargs["op1"] not in {"bitwise_and", "bitwise_or", "bitwise_xor"} or "operand1" not in kwargs
        ):
            raise ValueError("a second operator requires bitwise_and/or/xor and operand1")
        if kwargs["op0"] == "right_shift" and kwargs.get("op1") not in {None, "bitwise_and"}:
            raise ValueError("logical right shift supports only a following bitwise_and")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Apply the ordered native uint32 bit-pattern operations."""
        result = _BITWISE[kwargs["op0"]](kwargs["data"], np.uint32(kwargs["operand0"]))
        if kwargs.get("op1") is not None:
            result = _BITWISE[kwargs["op1"]](result, np.uint32(kwargs["operand1"]))
        return result
