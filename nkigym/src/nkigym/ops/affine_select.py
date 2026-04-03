"""Position-predicated element select op: nisa.affine_select.

Generates affine_value = offset + p*channel_multiplier + f*step
per element, compares to 0. Selects data or on_false_value.
"""

from typing import ClassVar

import numpy as np

from nkigym.codegen.ir import RenderContext
from nkigym.ops.base import NKIOp


class NKIAffineSelect(NKIOp):
    """Affine select: position-predicated element select (GpSimd Engine).

    Generates an affine value per element and selects between
    the input tile and a constant based on comparison.
    Renderer computes offset from tile_start at render time.

    Attributes:
        NAME: ``"affine_select"``.
        OPERAND_AXES: Single operand ``data`` with axes ``(P, F)``.
        OUTPUT_AXES: Output axes ``(P, F)``.
        MAX_TILE_SIZES: Partition axis capped at 128.
        ENGINE: GpSimd.
    """

    NAME: ClassVar[str] = "affine_select"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"output": ("P", "F")}
    MAX_TILE_SIZES: ClassVar[dict[str, int]] = {"P": 128}
    ENGINE: ClassVar[str] = "GpSimd"
    NEEDS_TILE_POSITION: ClassVar[bool] = True

    _CMP_FNS: ClassVar[dict[str, object]] = {"greater_equal": np.greater_equal}

    def __call__(
        self, data: np.ndarray, cmp_op: str, on_false_value: float, channel_multiplier: int, step: int, **_: object
    ) -> np.ndarray:
        """CPU simulation: affine select with comparison.

        Args:
            data: Array of shape (P, F).
            cmp_op: Comparison operation.
            on_false_value: Value when predicate is false.
            channel_multiplier: P-axis scale factor.
            step: F-axis step.

        Returns:
            Masked array with same shape as data.
        """
        rows, cols = data.shape
        p_idx = np.arange(rows)[:, np.newaxis]
        f_idx = np.arange(cols)[np.newaxis, :]
        mask = self._CMP_FNS[cmp_op](p_idx * channel_multiplier + f_idx * step, 0)
        return np.where(mask, data, on_false_value)

    def render_isa(self, ctx: RenderContext) -> str:
        """Emit nisa.affine_select call with tile-position offset.

        The offset accounts for the tile's position in the full tensor:
        ``offset = tile_start_P * channel_multiplier + tile_start_F * step``
        so local indices produce correct global affine values.

        Args:
            ctx: Render context.

        Returns:
            NKI source line for affine_select.
        """
        dst = ctx.outputs["output"]
        data = ctx.operands["data"]
        cmp_op = ctx.config_kwargs["cmp_op"]
        on_false_value = ctx.config_kwargs["on_false_value"]
        channel_multiplier = ctx.config_kwargs.get("channel_multiplier", 1)
        step = ctx.config_kwargs.get("step", -1)
        offset_expr = self._compute_offset(data, ctx, channel_multiplier, step)
        free_axis = data.axes[1]
        free_size = data.tile_size[free_axis]
        return (
            f"nisa.affine_select(dst={dst.default_indexed_slice()}, "
            f"pattern=[[{step}, {free_size}]], "
            f"offset={offset_expr}, "
            f"channel_multiplier={channel_multiplier}, "
            f"cmp_op=nl.{cmp_op}, "
            f"on_true_tile={data.default_indexed_slice()}, "
            f"on_false_value={on_false_value})"
        )

    def _compute_offset(
        self, data: "nkigym.codegen.ir.Tensor", ctx: RenderContext, channel_multiplier: int, step: int
    ) -> str:
        """Compute the offset expression from tile position.

        Args:
            data: Data operand Tensor.
            ctx: Render context with tile_start expressions.
            channel_multiplier: P-axis scale factor.
            step: F-axis step.

        Returns:
            Python expression string for offset.
        """
        parts: list[str] = []
        if len(data.axes) >= 1 and channel_multiplier != 0:
            p_dim = data.axes[0]
            p_start = ctx.tile_start.get(p_dim)
            if p_start is not None:
                parts.append(f"{p_start} * {channel_multiplier}")
        if len(data.axes) >= 2 and step != 0:
            f_dim = data.axes[1]
            f_start = ctx.tile_start.get(f_dim)
            if f_start is not None:
                parts.append(f"{f_start} * {step}")
        if not parts:
            result = "0"
        else:
            result = " + ".join(parts)
        return result
