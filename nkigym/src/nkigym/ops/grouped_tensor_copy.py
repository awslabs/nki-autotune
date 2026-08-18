"""Strided packed-group copy through ``nisa.tensor_copy``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKIGroupedTensorCopy(NKIOp):
    """Interleave chunked group tiles into one packed SBUF tensor."""

    NAME: ClassVar[str] = "tensor_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "src": ("C", "G", "M", "T", "N"),
        "dst": ("M", "C", "T", "G", "N"),
    }
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("C", "G", "M"), ("T", "N")),
        "dst": (("M",), ("C", "T", "G", "N")),
    }
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("M",), ("T",), ("N",)),
        "dst": (("M",), ("T",), ("N",)),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {
        "C": "chunks",
        "G": "groups",
        "T": "tiles",
        "M": "partition",
        "N": "queries",
    }
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "CGTN"}
    MIN_TILE_SIZE.update({"M": 128})
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"C": 1, "G": 1, "T": None, "M": 128, "N": None}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"T", "M", "N"})
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"chunks", "groups", "tiles", "partition", "queries"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, chunks: int, groups: int, tiles: int, partition: int, queries: int) -> None:
        """Configure the source and destination group layouts."""
        super().__init__(chunks=chunks, groups=groups, tiles=tiles, partition=partition, queries=queries)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require a PSUM or SBUF source."""
        if (role := _operand_role(kwargs["src"])) is not None and role not in {"psum", "sbuf"}:
            raise TypeError(f"NKIGroupedTensorCopy(src=<role={role}>) expects psum or sbuf")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the packed interleaving for CPU validation."""
        c, g, t, m, n = (int(kwargs[key]) for key in ("chunks", "groups", "tiles", "partition", "queries"))
        source = np.asarray(kwargs["src"]).reshape(c, g, m, t, n)
        return source.transpose(2, 0, 3, 1, 4).reshape(m, c * t * g * n)


def grouped_attention(
    q: object, k: object, v: object, mask: object, chunks: int, groups: int, tiles: int, width: int, queries: int
) -> object:
    """Mark grouped token-generation attention in a synthetic FX graph."""
    raise RuntimeError("grouped_attention is a trace-only marker")


__all__ = ["NKIGroupedTensorCopy"]
