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
