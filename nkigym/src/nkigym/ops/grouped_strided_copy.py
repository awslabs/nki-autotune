"""Bit-preserving strided copies within contiguous free-axis groups."""

from collections.abc import Mapping
from math import prod
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, SliceContract, _operand_role


class NKIGroupedStridedCopy(NKIOp):
    """Select one regular strided view independently in each contiguous group."""

    NAME: ClassVar[str] = "tensor_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "G", "F"), "dst": ("P", "G", "N")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("P",), ("G", "F")),
        "dst": (("P",), ("G", "N")),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf"})}
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"float32", "uint32"})}
    STRIDED_COPY_INPUTS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "F": "source_width", "N": "width"}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"F", "N"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "PGFN"}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "G": 1, "F": None, "N": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "source_width", "pattern", "offset", "width"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, groups: int, source_width: int, pattern: tuple[tuple[int, int], ...], offset: int) -> None:
        """Configure complete groups and a bounded strided selection."""
        if groups < 1 or source_width < 1 or not pattern or len(pattern) > 4 or offset < 0:
            raise ValueError("grouped strided copy requires positive groups and a valid pattern")
        if any(stride < 0 or count < 1 for stride, count in pattern):
            raise ValueError("grouped strided copy requires nonnegative strides and positive counts")
        if offset + sum(stride * (count - 1) for stride, count in pattern) >= source_width:
            raise ValueError("grouped strided copy exceeds a source group")
        super().__init__(
            groups=groups,
            source_width=source_width,
            pattern=pattern,
            offset=offset,
            width=prod(count for _, count in pattern),
            engine="vector",
        )

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> SliceContract:
        """Declare the exact global ordered selection from the source."""
        groups, width = int(kwargs["groups"]), int(kwargs["width"])
        pattern = ((int(kwargs["source_width"]), groups), *kwargs["pattern"])
        return SliceContract("src", "dst", 1, int(kwargs["offset"]), groups * width, pattern)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require rank-two 32-bit SBUF storage with complete source groups."""
        source = np.asarray(kwargs["src"])
        if _operand_role(kwargs["src"]) not in {None, "sbuf"} or source.ndim != 2:
            raise TypeError("NKIGroupedStridedCopy requires rank-two SBUF input")
        if source.dtype not in (np.dtype("float32"), np.dtype("uint32")):
            raise TypeError("NKIGroupedStridedCopy requires 32-bit storage")
        if source.shape[1] != int(kwargs["groups"]) * int(kwargs["source_width"]):
            raise ValueError("NKIGroupedStridedCopy input must contain complete groups")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Select source positions without arithmetic or dtype conversion."""
        source = np.asarray(kwargs["src"])
        grouped = source.reshape(source.shape[0], int(kwargs["groups"]), int(kwargs["source_width"]))
        positions = np.array([int(kwargs["offset"])], dtype=np.int64)
        for stride, count in kwargs["pattern"]:
            positions = (positions[:, None] + stride * np.arange(count)).reshape(-1)
        return grouped[:, :, positions].reshape(source.shape[0], -1).copy()


__all__ = ["NKIGroupedStridedCopy"]
