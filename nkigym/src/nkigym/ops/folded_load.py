"""Folded batch/tile DMA load."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKIFoldedLoad(NKIOp):
    """Load independent two-dimensional tiles from one packed HBM tensor."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "G", "T", "F"), "dst": ("P", "G", "T", "F")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        slot: (("P",), ("G", "T", "F")) for slot in OPERAND_AXES
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "T": "tiles"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "PGTF"}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "G": 1, "T": 1, "F": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "tiles"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, groups: int, tiles: int) -> None:
        """Configure packed batch and tile extents."""
        if groups < 1 or tiles < 1:
            raise ValueError("folded load extents must be positive")
        super().__init__(groups=groups, tiles=tiles)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one HBM parameter source."""
        if (role := _operand_role(kwargs["src"])) is not None and role != "param":
            raise TypeError(f"NKIFoldedLoad(src=<role={role}>) expects param")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return a copy of the packed source."""
        return np.array(kwargs["src"], copy=True)


def grouped_context_input(array: np.ndarray, shape: tuple[int, ...], transform: tuple[object, ...]) -> np.ndarray:
    """Pack one batched context-attention input by group and query tile."""
    kind = str(transform[1])
    raw_dimensions = transform[2:]
    if len(raw_dimensions) != 7 or not all(isinstance(value, int) for value in raw_dimensions):
        raise ValueError(f"grouped context-attention layout requires seven integer dimensions, got {raw_dimensions}")
    dimensions = tuple(value for value in raw_dimensions if isinstance(value, int))
    groups, queries, tiles, _reduction, partitions, width, _output_width = dimensions
    if kind in {"q", "k"}:
        result = array.transpose(1, 0, 2)
    elif kind == "v":
        result = array.reshape(groups, tiles, width // 128, 128, -1).transpose(3, 0, 1, 2, 4)
    elif kind in {"lower", "upper"}:
        result = array.reshape(groups, queries, partitions).transpose(2, 0, 1)
    else:
        raise ValueError(f"unknown grouped context-attention input layout {kind!r}")
    return result.reshape(shape)


__all__ = ["NKIFoldedLoad", "grouped_context_input"]
