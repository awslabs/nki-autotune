"""Memset op: nisa.memset.

Fill a tile with a constant value. No operands, one output, a
``value`` scalar kwarg. Used by online-fusion rewrites to
initialize the running reduction state (``running_x := 0`` for
sum, ``-inf`` for max) at the top of the group that owns the
online loop.

The renderer also emits ``nisa.memset`` directly for PSUM zeroing
before matmul accumulation (see ``codegen/nki_ops.py::_memset_lines``).
That path is separate — it doesn't go through this NKIOp class.
"""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp

VE_PARTITION_MAX = 128
VE_FREE_MAX = 512


class NKIMemset(NKIOp):
    """Fill a tile with a constant.

    Attributes:
        NAME: ``"memset"``.
        OPERAND_AXES: no operands.
        OUTPUT_AXES: output ``(P, F)``.
    """

    NAME: ClassVar[str] = "memset"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"output": ("P", "F")}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": VE_PARTITION_MAX, "F": VE_FREE_MAX}
    ISA_LOC: ClassVar[str] = "sbuf"
    PSUM_DTYPE: ClassVar[str | None] = None
    INPUT_LOCS: ClassVar[dict[str, str]] = {}

    def __call__(self, **kwargs: Any) -> np.ndarray:
        """CPU simulation: return a filled array matching ``shape``.

        Kwargs:
            shape: Output shape tuple.
            value: Fill value.
            dtype: Output dtype (optional; defaults to float32).

        Returns:
            Filled array.
        """
        shape = kwargs["shape"]
        value = kwargs["value"]
        dtype = kwargs.get("dtype", np.float32)
        return np.full(shape, value, dtype=dtype)

    @classmethod
    def format_isa_call(
        cls, dst_expr: str, operand_exprs: dict[str, str], scalar_kwargs: dict[str, str] | None = None
    ) -> str:
        """Format ``nisa.memset(dst, value)``."""
        sk = scalar_kwargs or {}
        value = sk.get("value", "0.0")
        return f"nisa.memset({dst_expr}, {value})"
