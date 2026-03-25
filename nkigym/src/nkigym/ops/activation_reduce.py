"""Compound activation+reduce op definition for schedule-based rendering.

Applies an element-wise activation then reduces across the free axis,
producing a 1D column vector.  Maps to ``nisa.activation`` with
``reduce_op`` and ``reduce_res`` keyword arguments.
"""

from typing import ClassVar

from nkigym.ops.base import NKIOp, _op_display_name


class NKIActivationReduce(NKIOp):
    """Activation+reduce: ``nisa.activation(dst=..., data=..., op=..., reduce_op=..., reduce_res=...)``.

    Applies element-wise activation then reduces across the free axis.
    Output is 1D ``(P,)`` — free axis collapsed (barrier op for pass 0).

    Attributes:
        op_name: Registry key ``"activation_reduce"``.
        OPERAND_AXES: Single operand ``data`` with axes ``(P, F)``.
        OUTPUT_AXES: Output axes ``(P,)`` --- free axis collapsed.
        TILE_LIMITS: Partition axis capped at 128.
    """

    op_name: ClassVar[str] = "activation_reduce"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[tuple[str, ...]] = ("P",)
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": 128}

    @classmethod
    def render_compute(
        cls, dst_expr: str, operand_exprs: dict[str, str], config_kwargs: tuple[tuple[str, object], ...]
    ) -> str:
        """Render activation+reduce compute line.

        Args:
            dst_expr: PSUM reduce_res destination expression.
            operand_exprs: Must contain ``"data"`` and ``"_scratch_dst"`` keys.
            config_kwargs: Must contain ``("op", ...)``, ``("reduce_op", ...)``.

        Returns:
            NKI activation source line with reduce_op and reduce_res.
        """
        kwargs = dict(config_kwargs)
        op_name = _op_display_name(kwargs.get("op", "square"))
        reduce_name = _op_display_name(kwargs.get("reduce_op", "add"))
        scratch = operand_exprs.get("_scratch_dst", "sbuf_scratch")
        data = operand_exprs["data"]
        return f"nisa.activation(dst={scratch}, op=nl.{op_name}, data={data}, reduce_op=np.{reduce_name}, reduce_res={dst_expr})"
