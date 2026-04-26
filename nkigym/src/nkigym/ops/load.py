"""HBM → SBUF load op: ``nisa.dma_copy`` + ``load_block`` gadget."""

from typing import Any, ClassVar

import nki.isa as nisa
import numpy as np

from nkigym.ops.base import NKIOp


class NKILoad(NKIOp):
    """Copy an HBM tensor into an SBUF buffer with identical logical layout."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, str]]] = {"output": ("P", "F")}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()
    """``nisa.dma_copy`` has no tile-size constraint beyond
    ``src.size == dst.size`` and partition-dim validation. Only the
    partition axis is capped by the NeuronCore's 128-partition SBUF
    layout; the free axis is unbounded."""
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": 128}

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: identity pass-through."""
        data: np.ndarray = kwargs["data"]
        return data


def load_block(sbuf: Any, mem_slice: Any) -> None:
    """HBM → SBUF: copy ``mem_slice`` into every leaf of ``sbuf`` via ``nisa.dma_copy``.

    ``sbuf`` is a list of ``num_p_tiles`` leaves of shape
    ``(p_tile, f_tile)``. The caller passes the HBM region as a
    Python slice; the gadget splits it across the P-slots.

    Required: ``mem_slice.shape == (num_p_tiles * p_tile, f_tile)``.

    For HBM→SBUF transposes, use :func:`dma_transpose_block` instead.
    """
    num_p_tiles = len(sbuf)
    if num_p_tiles == 0:
        raise ValueError("load_block got empty sbuf")
    p_tile, f_tile = sbuf[0].shape
    expected = (num_p_tiles * p_tile, f_tile)
    if tuple(mem_slice.shape) != expected:
        raise ValueError(
            f"load_block extent mismatch: sbuf covers ({num_p_tiles * p_tile}, {f_tile}) "
            f"via {num_p_tiles} P-slots of ({p_tile}, {f_tile}), expected mem_slice {expected}, "
            f"got {tuple(mem_slice.shape)}"
        )
    for pt in range(num_p_tiles):
        nisa.dma_copy(sbuf[pt][0:p_tile, 0:f_tile], mem_slice[pt * p_tile : (pt + 1) * p_tile, 0:f_tile])
