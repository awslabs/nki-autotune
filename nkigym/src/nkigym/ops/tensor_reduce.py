"""Free-axis reduction op definition for schedule-based rendering."""

from typing import ClassVar

from nkigym.ops.base import NKIOp, _op_display_name


class NKITensorReduce(NKIOp):
    """Tensor reduce: ``nisa.tensor_reduce(dst=..., data=..., op=...)``.

    Collapses the free axis, producing a 1D output.

    Attributes:
        op_name: Registry key ``"tensor_reduce"``.
        OPERAND_AXES: Single operand ``data`` with axes ``(P, F)``.
        OUTPUT_AXES: Output axes ``(P,)`` --- free axis collapsed.
        TILE_LIMITS: Partition axis capped at 128.
    """

    op_name: ClassVar[str] = "tensor_reduce"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[tuple[str, ...]] = ("P",)
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": 128}

    @classmethod
    def render_post_compute(
        cls, dst_expr: str, operand_exprs: dict[str, str], config_kwargs: tuple[tuple[str, object], ...]
    ) -> str:
        """Render tensor_reduce as a post-compute op.

        Args:
            dst_expr: Destination SBUF expression.
            operand_exprs: Must contain ``"data"`` key with source expression.
            config_kwargs: Must contain ``("op", <func>)`` for the reduction.

        Returns:
            NKI tensor_reduce source line.
        """
        kwargs = dict(config_kwargs)
        name = _op_display_name(kwargs.get("op", "add"))
        return f"nisa.tensor_reduce(dst={dst_expr}, data={operand_exprs['data']}, op=nl.{name})"
