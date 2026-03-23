"""Element-wise binary multiplication op definition for schedule-based rendering."""

from typing import ClassVar

from nkigym.ops.base import NKIOp


class NKIMultiply(NKIOp):
    """Element-wise multiply: ``nisa.tensor_tensor(dst=..., data1=..., data2=..., op=nl.multiply)``.

    Attributes:
        op_name: Registry key ``"multiply"``.
        OPERAND_AXES: Two operands ``data1`` and ``data2`` with axes ``(P, F)``.
        OUTPUT_AXES: Output axes ``(P, F)`` — element-wise, no reduction.
        TILE_LIMITS: No tile size overrides.
    """

    op_name: ClassVar[str] = "multiply"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"data1": ("P", "F"), "data2": ("P", "F")}
    OUTPUT_AXES: ClassVar[tuple[str, str]] = ("P", "F")
    TILE_LIMITS: ClassVar[dict[str, int]] = {}

    @classmethod
    def render_post_compute(
        cls, dst_expr: str, operand_exprs: dict[str, str], config_kwargs: tuple[tuple[str, object], ...]
    ) -> str:
        """Render element-wise multiply as a post-compute op.

        Args:
            dst_expr: Destination SBUF expression.
            operand_exprs: Must contain ``"data1"`` and ``"data2"`` keys.
            config_kwargs: Unused for multiply.

        Returns:
            NKI tensor_tensor source line with ``op=nl.multiply``.
        """
        return f"nisa.tensor_tensor(dst={dst_expr}, data1={operand_exprs['data1']}, data2={operand_exprs['data2']}, op=nl.multiply)"
