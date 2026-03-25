"""PE array transpose op definition for schedule-based rendering."""

from typing import ClassVar

from nkigym.ops.base import NKIOp


class NKITranspose(NKIOp):
    """Transpose: ``nisa.nc_transpose(data=...)`` with PSUM copy-back.

    Swaps partition and free dimensions via the PE array.
    Output axes are ``(F, P)`` — reversed from input ``(P, F)``.
    Analysis unification automatically swaps dim IDs via OUTPUT_AXES.

    Rendered as a 3-line sequence (alloc psum, nc_transpose, nl.copy)
    handled by the multi-pass renderer.

    Attributes:
        op_name: Registry key ``"transpose"``.
        OPERAND_AXES: Single operand ``data`` with axes ``(P, F)``.
        OUTPUT_AXES: Output axes ``(F, P)`` --- swapped.
        TILE_LIMITS: Partition axis capped at 128.
    """

    op_name: ClassVar[str] = "transpose"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[tuple[str, str]] = ("F", "P")
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": 128}
