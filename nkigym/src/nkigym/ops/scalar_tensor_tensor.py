"""Scalar-tensor-tensor op: nisa.scalar_tensor_tensor.

Fused ``(data <op0> operand0) <op1> operand1`` where ``operand0`` is
a scalar constant or ``(P, 1)`` column vector (broadcast along the
free axis) and ``operand1`` is a full-shape tile (element-wise).

Used by online fusion to implement the correction + new-bias fused
multiply-add on the running accumulator without materializing the
intermediate: ``running_sum = sigma * running_sum + tile_sum`` maps
to ``scalar_tensor_tensor(data=running_sum, op0=multiply,
operand0=sigma, op1=add, operand1=tile_sum)``. Matches the
reference attention CTE write-back path (lines 2559-2564).
"""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp

_OPS: dict[str, Any] = {
    "add": np.add,
    "subtract": np.subtract,
    "multiply": np.multiply,
    "maximum": np.maximum,
    "minimum": np.minimum,
}

VE_PARTITION_MAX = 128
VE_FREE_MAX = 512


class NKIScalarTensorTensor(NKIOp):
    """Fused scalar-broadcast + elementwise: ``(data op0 operand0) op1 operand1``.

    Attributes:
        NAME: ``"scalar_tensor_tensor"``.
        OPERAND_AXES: data ``(P, F)``, operand0 ``(P,)``, operand1 ``(P, F)``.
        OUTPUT_AXES: output ``(P, F)``.
    """

    NAME: ClassVar[str] = "scalar_tensor_tensor"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "data": ("P", "F"),
        "operand0": ("P",),
        "operand1": ("P", "F"),
    }
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"output": ("P", "F")}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": VE_PARTITION_MAX, "F": VE_FREE_MAX}
    ISA_LOC: ClassVar[str] = "sbuf"
    PSUM_DTYPE: ClassVar[str | None] = None
    INPUT_LOCS: ClassVar[dict[str, str]] = {"data": "sbuf_or_psum", "operand0": "sbuf", "operand1": "sbuf_or_psum"}
    FLOAT32_KWARGS: ClassVar[frozenset[str]] = frozenset({"operand0"})

    def __call__(self, **kwargs: Any) -> np.ndarray:
        """CPU simulation: ``(data <op0> operand0) <op1> operand1``.

        Kwargs:
            data: Array of shape (P, F).
            op0: First operator name (arithmetic).
            operand0: Scalar or (P,) column vector.
            op1: Second operator name (arithmetic).
            operand1: Array of shape (P, F) for element-wise.
            reverse0: Swap operand order for op0.
            reverse1: Swap operand order for op1.

        Returns:
            Result array of shape (P, F).
        """
        data: np.ndarray = kwargs["data"]
        op0: str = kwargs["op0"]
        op1: str = kwargs["op1"]
        operand0 = kwargs["operand0"]
        operand1: np.ndarray = kwargs["operand1"]
        reverse0: bool = kwargs.get("reverse0", False)
        reverse1: bool = kwargs.get("reverse1", False)
        b = operand0[..., np.newaxis] if isinstance(operand0, np.ndarray) else operand0
        temp = _OPS[op0](b, data) if reverse0 else _OPS[op0](data, b)
        return _OPS[op1](operand1, temp) if reverse1 else _OPS[op1](temp, operand1)

    @classmethod
    def format_isa_call(
        cls, dst_expr: str, operand_exprs: dict[str, str], scalar_kwargs: dict[str, str] | None = None
    ) -> str:
        """Format ``nisa.scalar_tensor_tensor(dst, data, op0, operand0, op1, operand1, ...)``."""
        sk = dict(scalar_kwargs or {})
        op0 = cls._to_nl(sk.pop("op0", "nl.multiply"))
        op1 = cls._to_nl(sk.pop("op1", "nl.add"))
        data = operand_exprs["data"]
        operand0 = operand_exprs.get("operand0", sk.pop("operand0", "0.0"))
        operand1 = operand_exprs["operand1"]
        extra = cls._format_scalar_kwargs(sk, set(cls.OPERAND_AXES))
        return f"nisa.scalar_tensor_tensor({dst_expr}, {data}, {op0}, {operand0}, {op1}, {operand1}{extra})"
