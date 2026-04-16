"""Tile-by-tile DMA transfer helpers for on-chip buffers.

4D buffer layout: ``(tile_size_P, num_tiles_P, num_tiles_F, tile_size_F)``.
2D buffer layout: ``(tile_size_P, num_tiles_P)``.
"""

from typing import Any

import nki.isa as nisa


def _tile_slice_4d(buf: Any, par_tid: int, free_tid: int) -> Any:
    """Index one tile in a 4D buffer."""
    tp = buf.shape[0]
    tf = buf.shape[3]
    return buf[0:tp, par_tid : par_tid + 1, free_tid : free_tid + 1, 0:tf]


def _tile_slice_2d(buf: Any, par_tid: int) -> Any:
    """Index one tile in a 2D buffer."""
    tp = buf.shape[0]
    return buf[0:tp, par_tid : par_tid + 1]


def load_tensor_block(dst: Any, src: Any, par_ofs: int, free_ofs: int) -> None:
    """Load tiles from HBM into an on-chip buffer.

    Iterates over all tile slots in ``dst`` and copies each
    from ``src`` via DMA, starting at the given element offsets.

    Args:
        dst: On-chip buffer (4D or 2D layout).
        src: HBM tensor (2D or 1D).
        par_ofs: Element offset along the partition axis.
        free_ofs: Element offset along the free axis (ignored for 1D).
    """
    tp = dst.shape[0]
    np_ = dst.shape[1]

    if len(dst.shape) == 4:
        tf = dst.shape[3]
        nf = dst.shape[2]
        for pi in range(np_):
            ps = par_ofs + pi * tp
            for fi in range(nf):
                fs = free_ofs + fi * tf
                nisa.dma_copy(dst=_tile_slice_4d(dst, pi, fi), src=src[ps : ps + tp, fs : fs + tf])
    elif len(dst.shape) == 2:
        for pi in range(np_):
            ps = par_ofs + pi * tp
            nisa.dma_copy(dst=_tile_slice_2d(dst, pi), src=src[ps : ps + tp])


def stage_tensor_block(dst: Any, src: Any) -> None:
    """Copy all tiles from a PSUM buffer to an equally-shaped SBUF buffer.

    Iterates over every tile slot and issues ``nisa.tensor_copy``
    for each.

    Args:
        dst: SBUF buffer (same shape as src).
        src: PSUM buffer.
    """
    np_ = src.shape[1]

    if len(src.shape) == 4:
        nf = src.shape[2]
        for pi in range(np_):
            for fi in range(nf):
                nisa.tensor_copy(dst=_tile_slice_4d(dst, pi, fi), src=_tile_slice_4d(src, pi, fi))
    elif len(src.shape) == 2:
        for pi in range(np_):
            nisa.tensor_copy(dst=_tile_slice_2d(dst, pi), src=_tile_slice_2d(src, pi))


def store_tensor_block(dst: Any, src: Any, par_ofs: int, free_ofs: int) -> None:
    """Store tiles from an on-chip SBUF buffer to HBM.

    The source must be in SBUF. For PSUM tensors, call
    ``stage_tensor_block`` first.

    Args:
        dst: HBM tensor (2D or 1D).
        src: SBUF buffer (4D or 2D layout).
        par_ofs: Element offset along the partition axis.
        free_ofs: Element offset along the free axis (ignored for 1D).
    """
    tp = src.shape[0]
    np_ = src.shape[1]

    if len(src.shape) == 4:
        tf = src.shape[3]
        nf = src.shape[2]
        for pi in range(np_):
            ps = par_ofs + pi * tp
            for fi in range(nf):
                fs = free_ofs + fi * tf
                nisa.dma_copy(dst=dst[ps : ps + tp, fs : fs + tf], src=_tile_slice_4d(src, pi, fi))
    elif len(src.shape) == 2:
        for pi in range(np_):
            ps = par_ofs + pi * tp
            nisa.dma_copy(dst=dst[ps : ps + tp], src=_tile_slice_2d(src, pi))
