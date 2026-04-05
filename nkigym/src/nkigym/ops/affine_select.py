"""Position-predicated element select op: nisa.affine_select.

Generates affine_value = offset + p*channel_multiplier + f*step
per element, compares to 0. Selects on_true_tile or on_false_value.
"""

import math
from typing import ClassVar

import numpy as np

from nkigym.codegen.ir import RenderContext
from nkigym.ops.base import NKIOp


def _format_float(value: float) -> str:
    """Format a float for NKI code generation.

    Converts special float values (inf, -inf, nan) to np.* expressions
    so the generated kernel uses valid Python.

    Args:
        value: The float value to format.

    Returns:
        String representation valid in generated NKI code.
    """
    if math.isinf(value):
        if value > 0:
            result = "np.inf"
        else:
            result = "-np.inf"
    elif math.isnan(value):
        result = "np.nan"
    else:
        result = repr(value)
    return result


class NKIAffineSelect(NKIOp):
    """Affine select: position-predicated element select.

    Generates an affine value per element and selects between
    the input tile and a constant based on comparison.
    Renderer computes offset from tile_start at render time.

    Attributes:
        NAME: ``"affine_select"``.
        OPERAND_AXES: Single operand ``on_true_tile`` with axes ``(P, F)``.
        OUTPUT_AXES: Output axes ``(P, F)``.
        MAX_TILE_SIZES: Partition axis capped at 128.
    """

    NAME: ClassVar[str] = "affine_select"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"on_true_tile": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"output": ("P", "F")}
    AXIS_ROLES: ClassVar[dict[str, str]] = {"P": "partition", "F": "free"}
    MAX_TILE_SIZES: ClassVar[dict[str, int]] = {"P": 128}
    NEEDS_TILE_POSITION: ClassVar[bool] = True

    _CMP_FNS: ClassVar[dict[str, object]] = {"greater_equal": np.greater_equal, "equal": np.equal}

    def __call__(
        self,
        pattern: list[list[int]],
        channel_multiplier: int,
        on_true_tile: np.ndarray,
        on_false_value: float,
        cmp_op: str = "equal",
        offset: int = 0,
        **_: object,
    ) -> np.ndarray:
        """CPU simulation: affine select with comparison.

        Args:
            pattern: List of [step, count] pairs defining free-axis layout.
            channel_multiplier: P-axis scale factor.
            on_true_tile: Array of shape (P, F).
            on_false_value: Value when predicate is false.
            cmp_op: Comparison operation name.
            offset: Base offset for the affine expression.

        Returns:
            Masked array with same shape as on_true_tile.
        """
        P = on_true_tile.shape[0]
        F = int(np.prod([n for _, n in pattern]))
        p_idx = np.arange(P)[:, np.newaxis]
        f_vals = np.array([0])
        for step, count in pattern:
            f_vals = (f_vals[:, np.newaxis] + np.arange(count) * step).ravel()
        affine = offset + p_idx * channel_multiplier + f_vals[np.newaxis, :]
        mask = self._CMP_FNS[cmp_op](affine, 0)
        return np.where(mask, on_true_tile.reshape(P, F), on_false_value)

    def render(self, ctx: RenderContext) -> list[str]:
        """Emit nisa.affine_select call with tile-position offset.

        The offset accounts for the tile's position in the full tensor:
        ``offset = tile_start_P * channel_multiplier + tile_start_F * step``
        so local indices produce correct global affine values.

        Args:
            ctx: Render context.

        Returns:
            NKI source lines for affine_select.
        """
        dst = ctx.outputs["output"]
        on_true = ctx.operands["on_true_tile"]
        cmp_op = ctx.config_kwargs["cmp_op"]
        on_false_value = _format_float(ctx.config_kwargs["on_false_value"])
        channel_multiplier = ctx.config_kwargs.get("channel_multiplier", 1)
        pattern = ctx.config_kwargs.get("pattern", [[-1, on_true.tile_size[on_true.axes[1]]]])
        step = pattern[0][0]
        free_axis = on_true.axes[1]
        free_size = on_true.tile_size[free_axis]
        offset_expr = self._compute_offset(on_true, ctx, channel_multiplier, step)
        return [
            f"nisa.affine_select(dst={dst.default_indexed_slice()}, "
            f"pattern=[[{step}, {free_size}]], "
            f"offset={offset_expr}, "
            f"channel_multiplier={channel_multiplier}, "
            f"cmp_op=nl.{cmp_op}, "
            f"on_true_tile={on_true.default_indexed_slice()}, "
            f"on_false_value={on_false_value})"
        ]

    def _compute_offset(
        self, on_true: "nkigym.codegen.ir.Tensor", ctx: RenderContext, channel_multiplier: int, step: int
    ) -> str:
        """Compute the offset expression from tile position.

        Args:
            on_true: on_true_tile operand Tensor.
            ctx: Render context with tile_start expressions.
            channel_multiplier: P-axis scale factor.
            step: F-axis step.

        Returns:
            Python expression string for offset.
        """
        parts: list[str] = []
        if len(on_true.axes) >= 1 and channel_multiplier != 0:
            p_dim = on_true.axes[0]
            p_start = ctx.tile_start.get(p_dim)
            if p_start is not None:
                parts.append(f"{p_start} * {channel_multiplier}")
        if len(on_true.axes) >= 2 and step != 0:
            f_dim = on_true.axes[1]
            f_start = ctx.tile_start.get(f_dim)
            if f_start is not None:
                parts.append(f"{f_start} * {step}")
        if not parts:
            result = "0"
        else:
            result = " + ".join(parts)
        return result
