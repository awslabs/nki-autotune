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
        ACC_FREE_DIM_LIMIT: Max PSUM free-dim elements per partition
            for the accumulator.  On v2/v3 each PSUM bank holds 512
            float32 elements (``nc_matmul`` moving free-dim limit).
            The compiler reload path supports up to 4 banks = 2048.
    """

    op_name: ClassVar[str] = "nc_matmul"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"stationary": ("K", "M"), "moving": ("K", "N")}
    OUTPUT_AXES: ClassVar[tuple[str, str]] = ("M", "N")
    TILE_LIMITS: ClassVar[dict[str, int]] = {"K": 128, "M": 128, "N": 512}
    ACC_FREE_DIM_LIMIT: ClassVar[int] = 2048

    @classmethod
    def render_compute(
        cls, dst_expr: str, operand_exprs: dict[str, str], config_kwargs: tuple[tuple[str, object], ...]
    ) -> str:
        """Render schedule compute line for matmul.

        Args:
            dst_expr: Destination expression for the accumulator.
            operand_exprs: Maps ``"stationary"`` and ``"moving"`` to expressions.
            config_kwargs: Unused for matmul.

        Returns:
            NKI matmul source line.
        """
        return f"nisa.nc_matmul(dst={dst_expr}, stationary={operand_exprs['stationary']}, moving={operand_exprs['moving']})"
