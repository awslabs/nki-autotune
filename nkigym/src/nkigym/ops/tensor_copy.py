"""Tensor-copy op: nisa.tensor_copy.

Copy an input tile into an output tile element-wise with no
computation. Used by online-fusion rewrites to snapshot the
previous running state (``prev_running := running``) before the
new X update, so the scale coefficient can be computed from both.

The renderer also emits ``nisa.tensor_copy`` directly for PSUM→SBUF
staging (see ``codegen/dma.py``); that path is separate — it
doesn't go through this NKIOp class. This class only covers
explicit copies that participate in the op graph as producer-
consumer edges.
"""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp

VE_PARTITION_MAX = 128
VE_FREE_MAX = 512


class NKITensorCopy(NKIOp):
    """Element-wise tile copy.

    Attributes:
        NAME: ``"tensor_copy"``.
        OPERAND_AXES: src ``(P, F)``.
        OUTPUT_AXES: output ``(P, F)``.
    """

    NAME: ClassVar[str] = "tensor_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"output": ("P", "F")}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": VE_PARTITION_MAX, "F": VE_FREE_MAX}
    ISA_LOC: ClassVar[str] = "sbuf"
    PSUM_DTYPE: ClassVar[str | None] = None
    INPUT_LOCS: ClassVar[dict[str, str]] = {"src": "sbuf_or_psum"}

    def __call__(self, **kwargs: Any) -> np.ndarray:
        """CPU simulation: copy ``src`` unchanged."""
        src: np.ndarray = kwargs["src"]
        return np.asarray(src).copy()

    @classmethod
    def format_isa_call(
        cls, dst_expr: str, operand_exprs: dict[str, str], scalar_kwargs: dict[str, str] | None = None
    ) -> str:
        """Format ``nisa.tensor_copy(dst, src)``."""
        extra = cls._format_scalar_kwargs(scalar_kwargs, set(cls.OPERAND_AXES))
        return f"nisa.tensor_copy({dst_expr}, {operand_exprs['src']}{extra})"
