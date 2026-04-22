"""Tensor-scalar op: nisa.tensor_scalar.

(data <op0> operand0) <op1> operand1 -> output(P, F).
operand0/operand1: scalar constant or (P,) column vector,
broadcast across the free axis.
"""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp

_OPS: dict[str, Any] = {"multiply": np.multiply, "subtract": np.subtract, "add": np.add}

VE_PARTITION_MAX = 128
VE_FREE_MAX = 512


class NKITensorScalar(NKIOp):
    """Element-wise tensor-scalar arithmetic.

    Computes ``(data <op0> operand0) <op1> operand1`` where each
    operand is either a scalar or a ``(P,)`` column vector that
    broadcasts across the free axis.

    Attributes:
        NAME: ``"tensor_scalar"``.
        OPERAND_AXES: data is ``(P, F)``.
        OUTPUT_AXES: output is ``(P, F)``.
    """

    NAME: ClassVar[str] = "tensor_scalar"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"output": ("P", "F")}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": VE_PARTITION_MAX, "F": VE_FREE_MAX}
    ISA_LOC: ClassVar[str] = "sbuf"
    PSUM_DTYPE: ClassVar[str | None] = None
    INPUT_LOCS: ClassVar[dict[str, str]] = {"data": "sbuf"}
    FLOAT32_KWARGS: ClassVar[frozenset[str]] = frozenset({"operand0", "operand1"})

    def __call__(self, **kwargs: Any) -> np.ndarray:
        """CPU simulation: (data <op0> operand0) <op1> operand1.

        Kwargs:
            data: Array of shape (P, F) or (P,).
            op0: First operation name.
            operand0: Scalar or (P,) vector.
            reverse0: Swap operand order for op0.
            op1: Optional second operation name.
            operand1: Scalar or (P,) vector for op1.
            reverse1: Swap operand order for op1.

        Returns:
            Result array, same shape as data.
        """
        data: np.ndarray = kwargs["data"]
        op0: str = kwargs["op0"]
        operand0 = kwargs["operand0"]
        reverse0: bool = kwargs.get("reverse0", False)
        op1: str | None = kwargs.get("op1")
        operand1 = kwargs.get("operand1")
        reverse1: bool = kwargs.get("reverse1", False)
        b = operand0[..., np.newaxis] if isinstance(operand0, np.ndarray) else operand0
        result = _OPS[op0](b, data) if reverse0 else _OPS[op0](data, b)
        if op1 is not None:
            c = operand1[..., np.newaxis] if isinstance(operand1, np.ndarray) else operand1
            result = _OPS[op1](c, result) if reverse1 else _OPS[op1](result, c)
        return result

    @classmethod
    def format_isa_call(
        cls, dst_expr: str, operand_exprs: dict[str, str], scalar_kwargs: dict[str, str] | None = None
    ) -> str:
        """Format nisa.tensor_scalar(dst, data, ...)."""
        extra = cls._format_scalar_kwargs(scalar_kwargs, set(cls.OPERAND_AXES))
        return f"nisa.tensor_scalar({dst_expr}, {operand_exprs['data']}{extra})"

    @classmethod
    def propagate_mask_value(cls, op_kwargs: dict[str, str], input_value: float) -> float | None:
        """Apply ``(input <op0> operand0) <op1> operand1`` at the scalar level.

        Returns ``None`` if any operand is tensor-valued (we can't
        resolve per-partition scales at propagation time) or if
        the op name isn't in the supported-literal set.
        """
        chain_ops = [("op0", "operand0"), ("op1", "operand1")]
        current: float | None = input_value
        for op_name_key, operand_key in chain_ops:
            if op_name_key not in op_kwargs or op_kwargs[op_name_key] is None:
                continue
            raw_op = op_kwargs[op_name_key]
            op_name = raw_op[1:-1] if raw_op.startswith("'") and raw_op.endswith("'") else raw_op
            operand_raw = op_kwargs.get(operand_key)
            operand = _parse_scalar_literal(operand_raw)
            if operand is None or current is None:
                current = None
                break
            reverse = op_kwargs.get(op_name_key.replace("op", "reverse")) == "True"
            current = _apply_binary(op_name, current, operand, reverse)
            if current is None:
                break
        return current


def _parse_scalar_literal(raw: str | None) -> float | None:
    """Parse a Python-literal scalar source string to ``float``; ``None`` if not a literal."""
    result: float | None = None
    if raw is None:
        result = None
    elif raw.startswith("np.float"):
        inner = raw[raw.find("(") + 1 : raw.rfind(")")]
        try:
            result = float(inner)
        except ValueError:
            result = None
    else:
        try:
            result = float(raw)
        except ValueError:
            result = None
    return result


def _apply_binary(op_name: str, lhs: float, rhs: float, reverse: bool) -> float | None:
    """Evaluate ``op_name`` on two scalars; ``None`` if ``op_name`` is unsupported."""
    a, b = (rhs, lhs) if reverse else (lhs, rhs)
    mapping: dict[str, Any] = {"add": a + b, "subtract": a - b, "multiply": a * b}
    return mapping.get(op_name)
