"""Reciprocal op: nisa.reciprocal.

Element-wise ``1 / data``. Dedicated ISA call (Vector Engine)
distinct from ``nisa.activation(op=nl.reciprocal)``: reciprocal is
an atomic instruction with no bias / scale composition. Used in
online-fusion rewrites to compute the final normalization factor
``1 / running_sum`` at the last section (reference attention CTE
lines 2472, 2481).
"""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp

VE_PARTITION_MAX = 128
VE_FREE_MAX = 512


class NKIReciprocal(NKIOp):
    """Element-wise reciprocal ``1 / data``.

    Attributes:
        NAME: ``"reciprocal"``.
        OPERAND_AXES: data ``(P, F)``.
        OUTPUT_AXES: output ``(P, F)``.
    """

    NAME: ClassVar[str] = "reciprocal"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"output": ("P", "F")}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": VE_PARTITION_MAX, "F": VE_FREE_MAX}
    ISA_LOC: ClassVar[str] = "sbuf"
    PSUM_DTYPE: ClassVar[str | None] = None
    INPUT_LOCS: ClassVar[dict[str, str]] = {"data": "sbuf_or_psum"}

    def __call__(self, **kwargs: Any) -> np.ndarray:
        """CPU simulation: ``1.0 / data``.

        Kwargs:
            data: Array of shape (P, F).

        Returns:
            Reciprocal array, same shape as input.
        """
        data: np.ndarray = kwargs["data"]
        return 1.0 / data

    @classmethod
    def format_isa_call(
        cls, dst_expr: str, operand_exprs: dict[str, str], scalar_kwargs: dict[str, str] | None = None
    ) -> str:
        """Format ``nisa.reciprocal(dst, data)``."""
        extra = cls._format_scalar_kwargs(scalar_kwargs, set(cls.OPERAND_AXES))
        return f"nisa.reciprocal({dst_expr}, {operand_exprs['data']}{extra})"
