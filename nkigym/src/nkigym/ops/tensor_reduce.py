"""Tensor-reduce op: nisa.tensor_reduce.

Reduce along the free axis with optional negation.
data(P, F) -> output(P,).
"""

from typing import ClassVar

import numpy as np

from nkigym.ops.base import NKIOp

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

    def __call__(
        self, data: np.ndarray, op: str, axis: int = 1, negate: bool = False, keepdims: bool = False, **_: object
    ) -> np.ndarray:
        """CPU simulation: reduce along axis with optional negation.

        Args:
            data: Array of shape (P, F).
            op: Reduction operation (``"max"`` or ``"add"``).
            axis: Axis to reduce (default 1 = free axis).
            negate: Negate the result.
            keepdims: Keep reduced dimensions.

        Returns:
            Reduced array of shape (P,) or (P, 1) if keepdims.
        """
        reduce_fns = {"max": np.max, "add": np.sum}
        result = reduce_fns[op](data, axis=axis, keepdims=keepdims)
        if negate:
            result = -result
        return result

    @classmethod
    def format_isa_call(cls, dst_expr: str, operand_exprs: dict[str, str]) -> str:
        """Format nisa.tensor_reduce(dst, data, ...)."""
        return f"nisa.tensor_reduce({dst_expr}," f" {operand_exprs['data']}, ...)"
