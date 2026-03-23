"""DMA transfer op definition for schedule-based rendering."""

from typing import ClassVar

from nkigym.ops.base import NKIOp


class NKIDmaCopy(NKIOp):
    """DMA copy: ``nisa.dma_copy(dst=..., src=...)``.

    Transfers data between HBM and SBUF. Direction determined by
    input/output tensor memory spaces.

    Attributes:
        op_name: Registry key ``"dma_copy"``.
        OPERAND_AXES: Single operand ``src`` with axes ``(P, F)``.
        OUTPUT_AXES: Output axes ``(P, F)``.
        TILE_LIMITS: Partition axis capped at 128.
    """

    op_name: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F")}
    OUTPUT_AXES: ClassVar[tuple[str, ...]] = ("P", "F")
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": 128}
