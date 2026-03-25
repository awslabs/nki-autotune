"""Element-wise tile-scalar op definition for schedule-based rendering."""

from typing import ClassVar

from nkigym.ops.base import NKIOp, _op_display_name


class NKITensorScalar(NKIOp):
    """Tensor-scalar: ``nisa.tensor_scalar(dst=..., data=..., operand0=..., op0=...)``.

    Element-wise op between a tile and a column vector.
    The ``operand0`` operand is 1D ``(P,)``; it broadcasts across the free axis.

    Attributes:
        op_name: Registry key ``"tensor_scalar"``.
        OPERAND_AXES: ``data`` is ``(P, F)``, ``operand0`` is ``(P,)``.
        OUTPUT_AXES: Output axes ``(P, F)``.
        TILE_LIMITS: Partition axis capped at 128.
    """

    op_name: ClassVar[str] = "tensor_scalar"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "operand0": ("P",)}
    OUTPUT_AXES: ClassVar[tuple[str, ...]] = ("P", "F")
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": 128}

    @classmethod
    def render_post_compute(
        cls, dst_expr: str, operand_exprs: dict[str, str], config_kwargs: tuple[tuple[str, object], ...]
    ) -> str:
        """Render tensor_scalar as a post-compute op.

        Args:
            dst_expr: Destination SBUF expression.
            operand_exprs: Must contain ``"data"`` and ``"operand0"`` keys.
            config_kwargs: Must contain ``("op0", <func>)`` for the operation.

        Returns:
            NKI tensor_scalar source line.
        """
        kwargs = dict(config_kwargs)
        name = _op_display_name(kwargs.get("op0", "add"))
        return f"nisa.tensor_scalar(dst={dst_expr}, data={operand_exprs['data']}, operand0={operand_exprs['operand0']}, op0=nl.{name})"
