"""Block-level DMA/staging wrappers that iterate per-slot (NKI forbids a single DMA on a multi-slot partition axis).

SBUF layout: ``(phys_P, num_slots_P[, num_slots_F, phys_F])``.
``stage_block``'s PSUM source is either a single 2D
``(phys_P, phys_F)`` ndarray or a Python list of 2D tiles indexed
row-major over ``(num_slots_P, num_slots_F)``.
"""

from typing import Any

import nki.isa as nisa


def load_block(dst: Any, src: Any) -> None:
    """HBM → SBUF: copy every tile slot in ``dst`` from the matching stripe of ``src``.

    Slot ``(pi, fi)`` of ``dst`` receives
    ``src[pi*phys_P : (pi+1)*phys_P, fi*phys_F : (fi+1)*phys_F]``.
    """
    if len(dst.shape) == 4:
        tp, np_p, nf_f, tf = dst.shape
        for pi in range(np_p):
            for fi in range(nf_f):
                nisa.dma_copy(
                    dst[0:tp, pi : pi + 1, fi : fi + 1, 0:tf], src[pi * tp : (pi + 1) * tp, fi * tf : (fi + 1) * tf]
                )
    else:
        tp, np_p = dst.shape
        for pi in range(np_p):
            nisa.dma_copy(dst[0:tp, pi : pi + 1], src[pi * tp : (pi + 1) * tp])


def store_block(dst: Any, src: Any) -> None:
    """SBUF → HBM: copy every tile slot of ``src`` into the matching stripe of ``dst``."""
    if len(src.shape) == 4:
        tp, np_p, nf_f, tf = src.shape
        for pi in range(np_p):
            for fi in range(nf_f):
                nisa.dma_copy(
                    dst[pi * tp : (pi + 1) * tp, fi * tf : (fi + 1) * tf], src[0:tp, pi : pi + 1, fi : fi + 1, 0:tf]
                )
    else:
        tp, np_p = src.shape
        for pi in range(np_p):
            nisa.dma_copy(dst[pi * tp : (pi + 1) * tp], src[0:tp, pi : pi + 1])


def stage_block(dst: Any, src: Any) -> None:
    """PSUM → SBUF: copy every PSUM tile into the matching SBUF slots via ``nisa.tensor_copy``.

    ``src`` is either a single 2D PSUM tile (``(phys_P, psum_F)``)
    or a list of such tiles. ``dst`` is a 4D SBUF view
    ``(phys_P, np_p, nf_f, phys_F)`` (or 2D ``(phys_P, np_p)``).

    Layout: the PSUM list length must evenly divide the SBUF
    slot count on exactly one of the two slot axes. For the
    single-PSUM case, one tile fills every SBUF slot — when the
    tile is wider than ``phys_F`` the extra chunks fill
    consecutive F-axis slots.
    """
    if len(dst.shape) == 4:
        tp, np_p, nf_f, tf = dst.shape
        src_list = [src] if _is_single_tile(src) else list(src)
        n = len(src_list)
        if n == 1:
            _stage_single(dst, src_list[0], tp, np_p, nf_f, tf)
        elif n == np_p:
            _stage_along_partition(dst, src_list, tp, np_p, nf_f, tf)
        elif n == nf_f:
            _stage_along_free(dst, src_list, tp, np_p, nf_f, tf)
        else:
            raise ValueError(f"PSUM list length {n} doesn't match SBUF slots np_p={np_p}, nf_f={nf_f}")
    else:
        tp, np_p = dst.shape
        for pi in range(np_p):
            tile = src if np_p == 1 else src[pi]
            nisa.tensor_copy(dst[0:tp, pi : pi + 1], tile[0:tp])


def _stage_single(dst: Any, tile: Any, tp: int, np_p: int, nf_f: int, tf: int) -> None:
    """One PSUM tile (possibly wider than ``tf``) fills every ``(pi, fi)`` SBUF slot.

    When ``tile.shape[1] == nf_f * tf`` the free axis is split
    across ``nf_f`` slots; when ``tile.shape[1] == tf`` the same
    narrow tile is broadcast to every slot (np_p == nf_f == 1).
    """
    for pi in range(np_p):
        for fi in range(nf_f):
            nisa.tensor_copy(dst[0:tp, pi : pi + 1, fi : fi + 1, 0:tf], tile[0:tp, fi * tf : (fi + 1) * tf])


def _stage_along_partition(dst: Any, src_list: list, tp: int, np_p: int, nf_f: int, tf: int) -> None:
    """PSUM list of length ``np_p``: tile ``ti`` fills slots ``(ti, fi)`` for every ``fi``."""
    for ti, tile in enumerate(src_list):
        for fi in range(nf_f):
            nisa.tensor_copy(dst[0:tp, ti : ti + 1, fi : fi + 1, 0:tf], tile[0:tp, fi * tf : (fi + 1) * tf])


def _stage_along_free(dst: Any, src_list: list, tp: int, np_p: int, nf_f: int, tf: int) -> None:
    """PSUM list of length ``nf_f``: tile ``ti`` fills slots ``(pi, ti)`` for every ``pi``."""
    for pi in range(np_p):
        for ti, tile in enumerate(src_list):
            nisa.tensor_copy(dst[0:tp, pi : pi + 1, ti : ti + 1, 0:tf], tile[0:tp, 0:tf])


def _is_single_tile(src: Any) -> bool:
    """True when ``src`` is a single PSUM tile (ndarray) rather than a list of tiles."""
    return hasattr(src, "shape") and not isinstance(src, list)
