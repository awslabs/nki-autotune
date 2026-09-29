"""Fold partition tiles into a packed free axis with an exact tensor copy."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKIPartitionPackCopy(NKIOp):
    """Copy equal physical partitions while folding their tile axis."""

    NAME: ClassVar[str] = "tensor_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("K", "P", "F"), "dst": ("P", "K", "F")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("K", "P"), ("F",)),
        "dst": (("P",), ("K", "F")),
    }
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        slot: (("P",), ("K",), ("F",)) for slot in OPERAND_AXES
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf"})}
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"float32", "bfloat16", "float16"})}
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"P": 128, "K": "tiles"}
    TILABLE_FIXED_AXES: ClassVar[frozenset[str]] = frozenset({"K"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "K": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "K": None, "F": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"tiles"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, tiles: int) -> None:
        """Configure a positive number of physical partition tiles."""
        if tiles < 1:
            raise ValueError("partition pack copy requires a positive tile count")
        super().__init__(tiles=tiles, engine="vector")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require complete supported SBUF partition tiles."""
        source = np.asarray(kwargs["src"])
        if _operand_role(kwargs["src"]) not in {None, "sbuf"}:
            raise TypeError("partition pack copy requires SBUF")
        if source.ndim != 2 or source.shape[0] != 128 * int(kwargs["tiles"]):
            raise ValueError("partition pack copy requires 128 times tiles rows")
        if str(source.dtype) not in self.INPUT_STORAGE_DTYPES["src"]:
            raise TypeError("partition pack copy requires a supported floating-point dtype")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Preserve every element while moving its tile index to the free axis."""
        source = np.asarray(kwargs["src"])
        tiles = int(kwargs["tiles"])
        return source.reshape(tiles, 128, source.shape[1]).transpose(1, 0, 2).reshape(128, -1).copy()


__all__ = ["NKIPartitionPackCopy"]
