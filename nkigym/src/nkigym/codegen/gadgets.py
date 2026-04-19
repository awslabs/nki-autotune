"""Block-level DMA/staging wrappers that hide per-slot iteration.

Purpose: NKI forbids a single DMA / ``tensor_copy`` that spans
multiple slots on the partition axis, so every load/stage/store
site has to emit ``for pi in range(NP): for fi in range(NF): ...``
around the ISA call. The gadgets encapsulate that loop so the
generated kernel source stays readable.

Contract: ``dst`` (or ``src`` for ``store_block``) is a 4D SBUF
region of shape ``(P, NP, NF, F)``; the 2D side is the HBM or PSUM
region that fills it. Shape compatibility is strict —
``P * NP == OP`` on partition and ``NF * F == OF`` on free, else
``ValueError``. No broadcast, no list source, no fallback.
"""

from typing import Any

import nki.isa as nisa


def load_block(dst: Any, src: Any) -> None:
    """HBM → SBUF: fill every slot of a 4D SBUF region from the matching chunk of a 2D HBM region.

    Slot ``(pi, fi)`` of ``dst`` receives
    ``src[pi*P : (pi+1)*P, fi*F : (fi+1)*F]``.
    """
    p, np_p, nf_f, f = dst.shape
    op, of = src.shape
    if op != p * np_p or of != nf_f * f:
        raise ValueError(
            f"load_block shape mismatch: dst {dst.shape} flattens to ({p * np_p}, {nf_f * f}), src {src.shape}"
        )
    for pi in range(np_p):
        for fi in range(nf_f):
            nisa.dma_copy(dst[0:p, pi : pi + 1, fi : fi + 1, 0:f], src[pi * p : (pi + 1) * p, fi * f : (fi + 1) * f])


def store_block(dst: Any, src: Any) -> None:
    """SBUF → HBM: write every slot of a 4D SBUF region to the matching chunk of a 2D HBM region.

    Slot ``(pi, fi)`` of ``src`` writes to
    ``dst[pi*P : (pi+1)*P, fi*F : (fi+1)*F]``.
    """
    p, np_p, nf_f, f = src.shape
    op, of = dst.shape
    if op != p * np_p or of != nf_f * f:
        raise ValueError(
            f"store_block shape mismatch: src {src.shape} flattens to ({p * np_p}, {nf_f * f}), dst {dst.shape}"
        )
    for pi in range(np_p):
        for fi in range(nf_f):
            nisa.dma_copy(dst[pi * p : (pi + 1) * p, fi * f : (fi + 1) * f], src[0:p, pi : pi + 1, fi : fi + 1, 0:f])


def stage_block(dst: Any, src: Any) -> None:
    """PSUM → SBUF: fill every slot of a 4D SBUF region from the matching chunk of a 2D PSUM region.

    Slot ``(pi, fi)`` of ``dst`` receives
    ``src[pi*P : (pi+1)*P, fi*F : (fi+1)*F]``.
    """
    p, np_p, nf_f, f = dst.shape
    op, of = src.shape
    if op != p * np_p or of != nf_f * f:
        raise ValueError(
            f"stage_block shape mismatch: dst {dst.shape} flattens to ({p * np_p}, {nf_f * f}), src {src.shape}"
        )
    for pi in range(np_p):
        for fi in range(nf_f):
            nisa.tensor_copy(dst[0:p, pi : pi + 1, fi : fi + 1, 0:f], src[pi * p : (pi + 1) * p, fi * f : (fi + 1) * f])
