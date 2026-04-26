"""SBUF → HBM store op: ``nisa.dma_copy`` + ``store_block`` gadget."""

from typing import Any, ClassVar

import nki.isa as nisa
import numpy as np

from nkigym.ops.base import NKIOp


class NKIStore(NKIOp):
    """Copy an SBUF buffer back to HBM with identical logical layout."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, str]]] = {"output": ("P", "F")}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()
    """Same story as ``NKILoad``: ``nisa.dma_copy`` only caps the
    partition axis (128) — the free axis is unbounded."""
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": 128}

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: identity pass-through."""
        data: np.ndarray = kwargs["data"]
        return data


def store_block(mem_slice: Any, sbuf: Any) -> None:
    """SBUF → HBM: write every leaf of ``sbuf`` into ``mem_slice``.

    Required: ``mem_slice.shape == (num_p_tiles * p_tile, f_tile)``.
    """
    num_p_tiles = len(sbuf)
    if num_p_tiles == 0:
        raise ValueError("store_block got empty sbuf")
    p_tile, f_tile = sbuf[0].shape
    expected = (num_p_tiles * p_tile, f_tile)
    if tuple(mem_slice.shape) != expected:
        raise ValueError(
            f"store_block extent mismatch: sbuf covers ({num_p_tiles * p_tile}, {f_tile}) "
            f"via {num_p_tiles} P-slots of ({p_tile}, {f_tile}), expected mem_slice {expected}, "
            f"got {tuple(mem_slice.shape)}"
        )
    for pt in range(num_p_tiles):
        nisa.dma_copy(mem_slice[pt * p_tile : (pt + 1) * p_tile, 0:f_tile], sbuf[pt][0:p_tile, 0:f_tile])
