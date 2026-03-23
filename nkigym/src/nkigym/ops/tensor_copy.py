"""Memory-space copy op definition for schedule-based rendering."""

from typing import ClassVar

from nkigym.ops.base import NKIOp


class NKITensorCopy(NKIOp):
    """Tensor copy: ``nisa.tensor_copy(dst=..., src=...)``.

    Copies data between memory spaces (e.g. PSUM to SBUF staging).

    Attributes:
        op_name: Registry key ``"tensor_copy"``.
        OPERAND_AXES: Single operand ``src`` with axes ``(P, F)``.
        OUTPUT_AXES: Output axes ``(P, F)``.
        TILE_LIMITS: Partition axis capped at 128.
    """

    op_name: ClassVar[str] = "tensor_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F")}
    OUTPUT_AXES: ClassVar[tuple[str, ...]] = ("P", "F")
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": 128}
