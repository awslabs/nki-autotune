"""Affine-select op: nisa.affine_select.

Position-predicated element select.
affine_value = offset + p * channel_multiplier + sum(idx_i * step_i).
Compares affine_value to 0; selects on_true_tile or on_false_value.
"""

import ast
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp

_CMP_OPS: dict[str, Any] = {"greater_equal": np.greater_equal, "equal": np.equal}

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

    def __call__(self, **kwargs: Any) -> np.ndarray:
        """CPU simulation: affine-predicated select.

        Kwargs:
            on_true_tile: Array of shape (P, F).
            pattern: List of [step, count] pairs for the free axis.
            channel_multiplier: Coefficient for partition index.
            on_false_value: Scalar fill for false positions.
            cmp_op: Comparison operator (``"equal"`` or ``"greater_equal"``).
            offset: Constant offset in the affine expression.

        Returns:
            Result array of shape (P, F).
        """
        on_true_tile: np.ndarray = kwargs["on_true_tile"]
        pattern: list[list[int]] = kwargs["pattern"]
        channel_multiplier: int = kwargs["channel_multiplier"]
        on_false_value: float = kwargs["on_false_value"]
        cmp_op: str = kwargs.get("cmp_op", "equal")
        offset: int = kwargs.get("offset", 0)
        p_count = on_true_tile.shape[0]
        f_count = int(np.prod([n for _, n in pattern]))
        p_idx = np.arange(p_count)[:, np.newaxis]
        f_vals = np.array([0])
        for step, count in pattern:
            f_vals = (f_vals[:, np.newaxis] + np.arange(count) * step).ravel()
        affine = offset + p_idx * channel_multiplier + f_vals[np.newaxis, :]
        mask = _CMP_OPS[cmp_op](affine, 0)
        return np.where(mask, on_true_tile.reshape(p_count, f_count), on_false_value)

    @classmethod
    def format_isa_call(
        cls, dst_expr: str, operand_exprs: dict[str, str], scalar_kwargs: dict[str, str] | None = None
    ) -> str:
        """Format nisa.affine_select(dst, pattern, channel_multiplier, on_true_tile, on_false_value, ...).

        The user writes ``pattern`` and ``offset`` in GLOBAL coordinates
        (as if the affine expression ran over the full tensor). This
        formatter rewrites them to per-tile coordinates using the
        geometry injected by the renderer: free-axis counts are clamped
        to the op tile's free size, and ``offset`` is augmented with
        ``channel_multiplier * p_tile_start`` plus per-axis
        ``step * f_tile_start`` so the tile's boundary reproduces the
        same global affine value. Only 1D free-axis patterns are
        supported today.
        """
        sk = dict(scalar_kwargs or {})
        pattern = cls._parse_pattern(sk.get("pattern", "[]"))
        ch_mul = int(sk.get("channel_multiplier", "0"))
        on_false = sk.get("on_false_value", "0.0")
        offset = int(sk.get("offset", "0"))
        f_tile_size = int(sk.get("__tile_size_F", "0"))
        p_tile_start = sk.get("__tile_start_P", "0")
        f_tile_start = sk.get("__tile_start_F", "0")

        new_pattern, offset_terms = cls._rewrite_pattern(pattern, f_tile_size, f_tile_start)
        offset_expr = cls._build_offset_expr(offset, ch_mul, p_tile_start, offset_terms)
        sk["pattern"] = repr(new_pattern)
        sk["offset"] = offset_expr
        extra = cls._format_scalar_kwargs(
            sk,
            set(cls.OPERAND_AXES)
            | {"pattern", "channel_multiplier", "on_false_value", "offset"}
            | {k for k in sk if k.startswith("__")},
        )
        return (
            f"nisa.affine_select({dst_expr}, {repr(new_pattern)}, {ch_mul}, "
            f"{operand_exprs['on_true_tile']}, {on_false}, offset={offset_expr}{extra})"
        )

    @staticmethod
    def _parse_pattern(pattern_src: str) -> list[list[int]]:
        """Parse a traced ``pattern`` literal into a list of ``[step, count]`` pairs."""
        parsed = ast.literal_eval(pattern_src)
        return [[int(step), int(count)] for step, count in parsed]

    @staticmethod
    def _rewrite_pattern(
        pattern: list[list[int]], f_tile_size: int, f_tile_start: str
    ) -> tuple[list[list[int]], list[str]]:
        """Clamp free-axis counts to one tile and collect per-axis offset contributions.

        Supports the common 1D case where a single ``[step, count]``
        describes the whole free axis. When ``count`` exceeds the
        tile's free size, it is clamped to ``f_tile_size`` and the
        omitted leading offset (``step * f_tile_start``) is returned
        so the caller can fold it into the instruction's ``offset``.
        Patterns already sized to the tile are left untouched and
        still contribute ``step * f_tile_start`` so the per-tile
        call reproduces the global affine value.
        """
        new_pattern = pattern
        offset_terms: list[str] = []
        if len(pattern) == 1:
            step, count = pattern[0]
            clamped = f_tile_size if f_tile_size and count > f_tile_size else count
            new_pattern = [[step, clamped]]
            if step and f_tile_start != "0":
                offset_terms.append(f"({step}) * ({f_tile_start})")
        return new_pattern, offset_terms

    @staticmethod
    def _build_offset_expr(base_offset: int, ch_mul: int, p_tile_start: str, extra_terms: list[str]) -> str:
        """Sum the base offset, partition-axis contribution, and per-axis free-axis contributions."""
        terms: list[str] = [str(base_offset)]
        if ch_mul and p_tile_start != "0":
            terms.append(f"({ch_mul}) * ({p_tile_start})")
        terms.extend(extra_terms)
        return " + ".join(terms)
