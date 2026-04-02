"""Position-predicated element select op for schedule-based rendering.

Generates an affine index pattern per element and selects data or a
fallback value based on a comparison.  Maps to ``nisa.affine_select``.
"""

from typing import ClassVar

from nkigym.ops.base import NKIOp


class NKIAffineSelect(NKIOp):
    """Affine select: ``nisa.affine_select(dst=..., on_true_tile=..., ...)``.

    Position-predicated element select using GpSimd Engine.
    Generates affine_value = offset + p*channel_multiplier + f*step
    per element, compares to 0, selects data or on_false_value.

    Attributes:
        op_name: Registry key ``"affine_select"``.
        OPERAND_AXES: Single operand ``data`` with axes ``(P, F)``.
        OUTPUT_AXES: Output axes ``(P, F)`` --- same shape as input.
        TILE_LIMITS: Partition axis capped at 128.
        NEEDS_TILE_POSITION: Renderer must inject tile start indices.
    """

    op_name: ClassVar[str] = "affine_select"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[tuple[str, ...]] = ("P", "F")
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": 128}
    NEEDS_TILE_POSITION: ClassVar[bool] = True

    @classmethod
    def render_post_compute(
        cls, dst_expr: str, operand_exprs: dict[str, str], config_kwargs: tuple[tuple[str, object], ...]
    ) -> str:
        """Render affine_select as a post-compute op.

        Args:
            dst_expr: Destination SBUF expression.
            operand_exprs: Must contain ``"data"`` key.
            config_kwargs: Contains cmp_op, on_false_value,
                channel_multiplier, step.

        Returns:
            NKI affine_select source line.
        """
        kwargs = dict(config_kwargs)
        cmp_op = kwargs.get("cmp_op", "greater_equal")
        on_false_value = kwargs.get("on_false_value", 0.0)
        channel_multiplier = kwargs.get("channel_multiplier", 1)
        step = kwargs.get("step", -1)
        data = operand_exprs["data"]
        return (
            f"nisa.affine_select(dst={dst_expr}, "
            f"on_true_tile={data}, "
            f"on_false_value={on_false_value}, "
            f"cmp_op=nl.{cmp_op}, "
            f"channel_multiplier={channel_multiplier}, "
            f"step={step})"
        )
