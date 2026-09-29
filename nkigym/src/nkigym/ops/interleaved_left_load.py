"""Load a transposed matrix with contraction factors split across SBUF axes."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKIInterleavedLeftLoad(NKIOp):
    """Pack HBM rows as partition, contraction-tile, and output-row axes."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("M", "P", "K"), "dst": ("P", "K", "M")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("M",), ("P", "K")),
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
            raise ValueError("interleaved load requires a positive tile count")
        super().__init__(tiles=tiles)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an HBM matrix with a complete factored contraction axis."""
        source = np.asarray(kwargs["src"])
        if _operand_role(kwargs["src"]) not in {None, "param", "shared_hbm", "stored"}:
            raise TypeError("interleaved load requires HBM input")
        if source.ndim != 2 or source.shape[1] != 128 * int(kwargs["tiles"]):
            raise ValueError("interleaved load requires 128 times tiles input columns")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the exact partition-major permutation."""
        source = np.asarray(kwargs["src"])
        data = source.reshape(source.shape[0], 128, int(kwargs["tiles"]))
        return data.transpose(1, 2, 0).reshape(128, -1).copy()


__all__ = ["NKIInterleavedLeftLoad"]
