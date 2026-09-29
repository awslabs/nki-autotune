"""Load a matrix with the outer contraction factor in SBUF partitions."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKIInterleavedRightLoad(NKIOp):
    """Reshape HBM contraction rows into partition and free-axis factors."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "K", "N"), "dst": ("P", "K", "N")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("P", "K"), ("N",)),
        "dst": (("P",), ("K", "N")),
    }
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        slot: (("P",), ("K",), ("N",)) for slot in OPERAND_AXES
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"shared_hbm"})}
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"P": 128, "K": "tiles"}
    TILABLE_FIXED_AXES: ClassVar[frozenset[str]] = frozenset({"K"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "K": 1, "N": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "K": None, "N": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"tiles"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, tiles: int) -> None:
        """Configure a positive contraction factor."""
        if tiles < 1:
            raise ValueError("interleaved load requires a positive tile count")
        super().__init__(tiles=tiles)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require a complete HBM matrix with 128 partitions of rows."""
        source = np.asarray(kwargs["src"])
        if _operand_role(kwargs["src"]) not in {None, "param", "shared_hbm", "stored"}:
            raise TypeError("interleaved load requires HBM input")
        if source.ndim != 2 or source.shape[0] != 128 * int(kwargs["tiles"]):
            raise ValueError("interleaved load requires 128 times tiles input rows")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Preserve matrix values in the packed partition-major representation."""
        source = np.asarray(kwargs["src"])
        return source.reshape(128, -1).copy()


__all__ = ["NKIInterleavedRightLoad"]
