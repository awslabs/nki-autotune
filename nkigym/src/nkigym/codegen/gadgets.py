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
    """PSUM → SBUF: copy every PSUM tile into the matching SBUF slot via ``nisa.tensor_copy``.

    ``src`` is either a single 2D PSUM tile (for single-slot
    ``dst``) or a list of 2D tiles indexed row-major over the
    multi-slot axes of ``dst``. Slot ``(pi, fi)`` of ``dst``
    receives ``src[pi * num_slots_F + fi]`` (or ``src`` itself
    when ``num_slots_P * num_slots_F == 1``).
    """
    if len(dst.shape) == 4:
        tp, np_p, nf_f, tf = dst.shape
        for pi in range(np_p):
            for fi in range(nf_f):
                tile = src if np_p * nf_f == 1 else src[pi * nf_f + fi]
                nisa.tensor_copy(dst[0:tp, pi : pi + 1, fi : fi + 1, 0:tf], tile[0:tp, 0:tf])
    else:
        tp, np_p = dst.shape
        for pi in range(np_p):
            tile = src if np_p == 1 else src[pi]
            nisa.tensor_copy(dst[0:tp, pi : pi + 1], tile[0:tp])
