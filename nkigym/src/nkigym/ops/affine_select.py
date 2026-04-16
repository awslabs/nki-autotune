"""Affine-select op: nisa.affine_select.

Position-predicated element select.
affine_value = offset + p * channel_multiplier + sum(idx_i * step_i).
Compares affine_value to 0; selects on_true_tile or on_false_value.
"""

from typing import ClassVar

import numpy as np

from nkigym.ops.base import NKIOp

VE_PARTITION_MAX = 128
VE_FREE_MAX = 512


class NKIAffineSelect(NKIOp):
    """Position-predicated element select.

    Builds an affine index from partition position and free-axis
    pattern, compares to zero, and selects between ``on_true_tile``
    and ``on_false_value`` per element.

    Attributes:
        NAME: ``"affine_select"``.
        OPERAND_AXES: on_true_tile is ``(P, F)``.
        OUTPUT_AXES: output is ``(P, F)``.
    """

    NAME: ClassVar[str] = "affine_select"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"on_true_tile": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"output": ("P", "F")}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": VE_PARTITION_MAX, "F": VE_FREE_MAX}
    ISA_LOC: ClassVar[str] = "sbuf"
    PSUM_DTYPE: ClassVar[str | None] = None
    INPUT_LOCS: ClassVar[dict[str, str]] = {"on_true_tile": "sbuf"}

    def __call__(
        self,
        on_true_tile: np.ndarray,
        pattern: list[list[int]],
        channel_multiplier: int,
        on_false_value: float,
        cmp_op: str = "equal",
        offset: int = 0,
        **_: object,
    ) -> np.ndarray:
        """CPU simulation: affine-predicated select.

        Args:
            on_true_tile: Array of shape (P, F).
            pattern: List of [step, count] pairs for the free axis.
            channel_multiplier: Coefficient for partition index.
            on_false_value: Scalar fill for false positions.
            cmp_op: Comparison operator (``"equal"`` or ``"greater_equal"``).
            offset: Constant offset in the affine expression.

        Returns:
            Result array of shape (P, F).
        """
        p_count = on_true_tile.shape[0]
        f_count = int(np.prod([n for _, n in pattern]))
        p_idx = np.arange(p_count)[:, np.newaxis]
        f_vals = np.array([0])
        for step, count in pattern:
            f_vals = (f_vals[:, np.newaxis] + np.arange(count) * step).ravel()
        affine = offset + p_idx * channel_multiplier + f_vals[np.newaxis, :]
        cmps = {"greater_equal": np.greater_equal, "equal": np.equal}
        mask = cmps[cmp_op](affine, 0)
        return np.where(mask, on_true_tile.reshape(p_count, f_count), on_false_value)

    @classmethod
    def format_isa_call(
        cls, dst_expr: str, operand_exprs: dict[str, str], scalar_kwargs: dict[str, str] | None = None
    ) -> str:
        """Format nisa.affine_select(dst, pattern, channel_multiplier, on_true_tile, on_false_value, ...)."""
        sk = scalar_kwargs or {}
        pattern = sk.get("pattern", "[]")
        ch_mul = sk.get("channel_multiplier", "0")
        on_false = sk.get("on_false_value", "0.0")
        extra = cls._format_scalar_kwargs(
            sk, set(cls.OPERAND_AXES) | {"pattern", "channel_multiplier", "on_false_value"}
        )
        return (
            f"nisa.affine_select({dst_expr}, {pattern}, {ch_mul}, {operand_exprs['on_true_tile']}, {on_false}{extra})"
        )
