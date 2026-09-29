"""Load a stationary matrix with interleaved contraction factors."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKIInterleavedStationaryLoad(NKIOp):
    """Split HBM contraction rows into partition and tile axes."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "K", "M"), "dst": ("P", "K", "M")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("P", "K"), ("M",)),
        "dst": (("P",), ("K", "M")),
    }
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        slot: (("P",), ("K",), ("M",)) for slot in OPERAND_AXES
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"shared_hbm"})}
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"P": 128, "K": "tiles"}
    TILABLE_FIXED_AXES: ClassVar[frozenset[str]] = frozenset({"K"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "M": 1, "K": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "M": None, "K": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"tiles"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, tiles: int) -> None:
        """Configure a positive contraction factor."""
        if tiles < 1:
            raise ValueError("interleaved stationary load requires a positive tile count")
        super().__init__(tiles=tiles)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require a complete stationary HBM matrix."""
        source = np.asarray(kwargs["src"])
        if _operand_role(kwargs["src"]) not in {None, "param", "shared_hbm", "stored"}:
            raise TypeError("interleaved stationary load requires HBM input")
        if source.ndim != 2 or source.shape[0] != 128 * int(kwargs["tiles"]):
            raise ValueError("interleaved stationary load requires 128 times tiles rows")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the exact partition-major stationary layout."""
        source = np.asarray(kwargs["src"])
        return source.reshape(128, -1).copy()


__all__ = ["NKIInterleavedStationaryLoad"]
