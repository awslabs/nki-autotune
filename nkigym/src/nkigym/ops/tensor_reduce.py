"""Tensor-reduce op: nisa.tensor_reduce.

Reduce along the free axis with optional negation.
data(P, F) -> output(P,).
"""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp

_REDUCE_FNS = {"maximum": np.max, "add": np.sum}

VE_PARTITION_MAX = 128
VE_FREE_MAX = 512


class NKITensorReduce(NKIOp):
    """Reduce along the free axis.

    Applies a reduction (max or add) along the free dimension,
    with optional negation of the result.

    Attributes:
        NAME: ``"tensor_reduce"``.
        OPERAND_AXES: data is ``(P, F)``.
        OUTPUT_AXES: output is ``(P,)``.
    """

    NAME: ClassVar[str] = "tensor_reduce"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"output": ("P",)}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset({"F"})
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": VE_PARTITION_MAX, "F": VE_FREE_MAX}
    ISA_LOC: ClassVar[str] = "sbuf"
    PSUM_DTYPE: ClassVar[str | None] = None
    INPUT_LOCS: ClassVar[dict[str, str]] = {"data": "sbuf"}

    def __call__(self, **kwargs: Any) -> np.ndarray:
        """CPU simulation: reduce along axis with optional negation.

        Kwargs:
            data: Array of shape (P, F).
            op: Reduction operation (``"maximum"`` or ``"add"``).
            axis: Axis to reduce (default 1 = free axis).
            negate: Negate the result.
            keepdims: Keep reduced dimensions.

        Returns:
            Reduced array of shape (P,) or (P, 1) if keepdims.
        """
        data: np.ndarray = kwargs["data"]
        op: str = kwargs["op"]
        axis: int = kwargs.get("axis", 1)
        negate: bool = kwargs.get("negate", False)
        keepdims: bool = kwargs.get("keepdims", False)
        result = _REDUCE_FNS[op](data, axis=axis, keepdims=keepdims)
        if negate:
            result = -result
        return result

    @classmethod
    def format_isa_call(
        cls, dst_expr: str, operand_exprs: dict[str, str], scalar_kwargs: dict[str, str] | None = None
    ) -> str:
        """Format nisa.tensor_reduce(dst, op, data, axis, ...)."""
        sk = scalar_kwargs or {}
        op_arg = cls._to_nl(sk.get("op", "nl.add"))
        axis = sk.get("axis", "1")
        extra = cls._format_scalar_kwargs(sk, set(cls.OPERAND_AXES) | {"op", "axis"})
        return f"nisa.tensor_reduce({dst_expr}, {op_arg}, {operand_exprs['data']}, {axis}{extra})"
