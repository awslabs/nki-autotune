"""Flattened tile broadcast through ``nisa.activation``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKITileBroadcast(NKIOp):
    """Repeat packed group vectors across chunk and tile axes."""

    NAME: ClassVar[str] = "activation"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("M", "G", "N"), "dst": ("M", "C", "T", "G", "N")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "data": (("M",), ("G", "N")),
        "dst": (("M",), ("C", "T", "G", "N")),
    }
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "data": (("M",), ("C",), ("T",), ("G",), ("N",)),
        "dst": (("M",), ("C",), ("T",), ("G",), ("N",)),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"C": "chunks", "T": "tiles", "G": "groups", "N": "queries"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "MCTGN"}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {axis: None for axis in "MCTGN"}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset("MCTGN")
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"chunks", "tiles", "groups", "queries"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, chunks: int, tiles: int, groups: int, queries: int) -> None:
        """Configure the repeated tile layout."""
        super().__init__(chunks=chunks, tiles=tiles, groups=groups, queries=queries, op="copy")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one SBUF source."""
        if (role := _operand_role(kwargs["data"])) is not None and role != "sbuf":
            raise TypeError(f"NKITileBroadcast(data=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the repeated packed vectors for CPU validation."""
        c, t, g, n = (int(kwargs[key]) for key in ("chunks", "tiles", "groups", "queries"))
        data = np.asarray(kwargs["data"]).reshape(-1, g, n)
        return np.broadcast_to(data[:, None, None], (data.shape[0], c, t, g, n)).reshape(data.shape[0], -1)


__all__ = ["NKITileBroadcast"]
