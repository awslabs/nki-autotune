"""Grouped range selection and row reduction."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import AxisRole, NKIOp, ReductionContract, _operand_role, reduction_combinator
from nkigym.ops.range_select import _COMPARISONS


class NKIGroupedRangeSelectReduceRow(NKIOp):
    """Select dynamic indices across chunks and emit per-chunk maxima."""

    NAME: ClassVar[str] = "range_select"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "on_true_tile": ("P", "G", "T", "F"),
        "bound0": ("P", "G"),
        "bound1": ("P", "G"),
        "dst": ("P", "G", "T", "F"),
        "reduce_res": ("P", "G", "T"),
    }
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "on_true_tile": (("P",), ("G", "T", "F")),
        "bound0": (("P",), ("G",)),
        "bound1": (("P",), ("G",)),
        "dst": (("P",), ("G", "T", "F")),
        "reduce_res": (("P",), ("G", "T")),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"on_true_tile", "bound0", "bound1"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "P": "partitions", "T": "chunks", "F": "width"}
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "GPTF"}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "P": 128, "T": 1, "F": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "partitions", "chunks", "width"})
    SPLIT_OFFSET_KWARGS: ClassVar[dict[str, tuple[str, str]]] = {"F": ("range_start", "on_true_tile")}
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"bound0": "float32", "bound1": "float32"}
    OUTPUT_DTYPES: ClassVar[dict[str, str]] = {"dst": "float32", "reduce_res": "float32"}
    OUTPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"dst": "float32", "reduce_res": "float32"}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, groups: int, partitions: int, chunks: int, width: int, comparison: str) -> None:
        """Configure one grouped dynamic selection."""
        if comparison not in {"exact", "range"}:
            raise ValueError(f"unsupported grouped range comparison {comparison!r}")
        comp_op0, comp_op1 = ("equal", "equal") if comparison == "exact" else ("greater_equal", "less")
        super().__init__(
            groups=groups,
            partitions=partitions,
            chunks=chunks,
            width=width,
            comp_op0=comp_op0,
            comp_op1=comp_op1,
            reduce_op="maximum",
            reduce_cmd="reset_reduce",
            on_false_value=float(np.finfo(np.float32).min),
        )

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> ReductionContract:
        """Return the masked row-reduction contract."""
        return ReductionContract(
            input_operand="on_true_tile",
            output_operand="reduce_res",
            reduction_axis="F",
            combinator=reduction_combinator(str(kwargs["reduce_op"])),
            map_operator="range_select",
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require on-chip data and bounds."""
        if any(
            _operand_role(kwargs[name]) not in {None, "sbuf", "psum"} for name in ("on_true_tile", "bound0", "bound1")
        ):
            raise TypeError("NKIGroupedRangeSelectReduceRow expects on-chip operands")

    def _run(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return selected values and one maximum per row and chunk."""
        g, p, t, f = (int(kwargs[name]) for name in ("groups", "partitions", "chunks", "width"))
        data = np.asarray(kwargs["on_true_tile"]).reshape(p, g, t, f)
        lower = np.asarray(kwargs["bound0"]).reshape(p, g, 1, 1)
        upper = np.asarray(kwargs["bound1"]).reshape(p, g, 1, 1)
        indices = np.arange(g * t * f, dtype=np.float32).reshape(1, g, t, f)
        selected = np.where(
            _COMPARISONS[str(kwargs["comp_op0"])](indices, lower)
            & _COMPARISONS[str(kwargs["comp_op1"])](indices, upper),
            data,
            np.finfo(np.float32).min,
        )
        return selected.reshape(p, g * t * f), np.max(selected, axis=3).reshape(p, g * t)


__all__ = ["NKIGroupedRangeSelectReduceRow"]
