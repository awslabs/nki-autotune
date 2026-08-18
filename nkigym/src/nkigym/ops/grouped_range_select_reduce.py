"""Grouped dynamic range selection with a partial maximum."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import AxisRole, NKIOp, _operand_role
from nkigym.ops.range_select import _COMPARISONS


class NKIGroupedRangeSelectReduce(NKIOp):
    """Mask grouped score tiles and emit their partition-wise maxima."""

    NAME: ClassVar[str] = "range_select"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "on_true_tile": ("G", "Q", "P", "T", "F"),
        "bound0": ("P", "G", "Q"),
        "bound1": ("P", "G", "Q"),
        "dst": ("G", "Q", "P", "T", "F"),
        "reduce_res": ("P", "G", "Q", "T"),
    }
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "on_true_tile": (("G", "Q", "P"), ("T", "F")),
        "bound0": (("P",), ("G", "Q")),
        "bound1": (("P",), ("G", "Q")),
        "dst": (("G", "Q", "P"), ("T", "F")),
        "reduce_res": (("P",), ("G", "Q", "T")),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"on_true_tile", "bound0", "bound1"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {
        "G": "groups",
        "Q": "queries",
        "T": "tiles",
        "P": "partitions",
        "F": "width",
    }
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "GQTPF"}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "Q": 1, "T": 1, "P": 128, "F": 512}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "queries", "tiles", "partitions", "width"})
    SPLIT_OFFSET_KWARGS: ClassVar[dict[str, tuple[str, str]]] = {"F": ("range_start", "dst")}
    OUTPUT_DTYPES: ClassVar[dict[str, str]] = {"dst": "float32", "reduce_res": "float32"}
    OUTPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"dst": "float32", "reduce_res": "float32"}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, groups: int, queries: int, tiles: int, partitions: int, width: int) -> None:
        """Configure grouped score and bound extents."""
        super().__init__(
            groups=groups,
            queries=queries,
            tiles=tiles,
            partitions=partitions,
            width=width,
            comp_op0="greater_equal",
            comp_op1="less",
            reduce_op="maximum",
            reduce_cmd="reset_reduce",
            on_false_value=float(np.finfo(np.float32).min),
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require on-chip score and bound tensors."""
        for slot in ("on_true_tile", "bound0", "bound1"):
            if (role := _operand_role(kwargs[slot])) is not None and role not in {"sbuf", "psum"}:
                raise TypeError(f"NKIGroupedRangeSelectReduce({slot}=<role={role}>) expects on-chip data")

    def _run(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return masked scores and one maximum per score tile."""
        g, q, t, p, f = (int(kwargs[key]) for key in ("groups", "queries", "tiles", "partitions", "width"))
        data = np.asarray(kwargs["on_true_tile"]).reshape(g, q, p, t, f)
        lower = np.asarray(kwargs["bound0"]).reshape(p, g, q).transpose(1, 2, 0)[..., None, None]
        upper = np.asarray(kwargs["bound1"]).reshape(p, g, q).transpose(1, 2, 0)[..., None, None]
        indices = np.arange(t * f, dtype=np.float32).reshape(1, 1, 1, t, f)
        valid = _COMPARISONS["greater_equal"](indices, lower) & _COMPARISONS["less"](indices, upper)
        masked = np.where(valid, data, np.finfo(np.float32).min)
        partial = np.max(masked, axis=4).transpose(2, 0, 1, 3)
        return masked.reshape(g * q * p, t * f), partial.reshape(p, g * q * t)


__all__ = ["NKIGroupedRangeSelectReduce"]
