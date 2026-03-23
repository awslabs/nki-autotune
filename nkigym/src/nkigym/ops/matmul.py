"""Matrix multiplication op definition for schedule-based rendering."""

from typing import ClassVar

from nkigym.ops.base import NKIOp


class NKIMatmul(NKIOp):
    """Matrix multiply: ``nisa.nc_matmul(dst=..., stationary=..., moving=...)``.

    Attributes:
        op_name: Registry key ``"nc_matmul"``.
        OPERAND_AXES: Stationary is ``(K, M)``, moving is ``(K, N)``.
        OUTPUT_AXES: Output is ``(M, N)``.
        TILE_LIMITS: K and M capped at 128, N capped at 512.
    """

    op_name: ClassVar[str] = "nc_matmul"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"stationary": ("K", "M"), "moving": ("K", "N")}
    OUTPUT_AXES: ClassVar[tuple[str, str]] = ("M", "N")
    TILE_LIMITS: ClassVar[dict[str, int]] = {"K": 128, "M": 128, "N": 512}

    @classmethod
    def render_compute(cls, dst_expr: str, operand_exprs: dict[str, str]) -> str:
        """Render schedule compute line for matmul.

        Args:
            dst_expr: Destination expression for the accumulator.
            operand_exprs: Maps ``"stationary"`` and ``"moving"`` to expressions.

        Returns:
            NKI matmul source line.
        """
        return f"nisa.nc_matmul(dst={dst_expr}, stationary={operand_exprs['stationary']}, moving={operand_exprs['moving']})"
