"""Block-level DMA/staging wrappers for the list-of-2D-tiles SBUF model.

SBUF buffers are nested Python lists ``sbuf_X[NP_list][NF_list]``
where each leaf is a 2D ``nl.ndarray(P, F)``. NKI forbids a
single DMA that spans multiple partition slots, so the gadgets
Python-iterate per leaf and emit one ISA call per tile. Each
inner call is a genuine 2D memref access — no 4D reshape, no
partition striding in the slice.

Contract:
* ``sbuf`` (dst for load/stage, src for store): nested Python
  list ``[NP_list][NF_list]`` of 2D ``nl.ndarray`` leaves of
  shape ``(P, F)``.
* ``mem`` (src for load/stage, dst for store): 2D ``nl.ndarray``
  of shape ``(p_count * P, f_count * F)`` — the matching chunk of
  HBM or PSUM.
* ``p_start``, ``f_start``: starting slot indices into ``sbuf``.
* ``p_count``, ``f_count``: number of slots to transfer on each
  axis. ``mem`` must cover exactly this region, else ``ValueError``.

``load_block`` accepts ``transpose=True`` to fold a 2D transpose
into the HBM→SBUF DMA via ``nisa.dma_transpose``. The ``mem``
region is then the pre-transpose HBM tile with shape
``(f_count * F, p_count * P)``; each leaf is filled with
``mem[fi*F:(fi+1)*F, pi*P:(pi+1)*P]`` so the destination's
partition axis takes values from the source's free axis.
"""

from typing import Any

import nki.isa as nisa


def load_block(
    sbuf: Any, mem: Any, p_start: int, p_count: int, f_start: int, f_count: int, transpose: bool = False
) -> None:
    """HBM → SBUF: copy ``mem`` into the ``[p_start : p_start + p_count][f_start : f_start + f_count]`` sub-block of ``sbuf``.

    When ``transpose=True``, ``mem`` is the pre-transpose HBM tile
    of shape ``(f_count * F, p_count * P)`` and each leaf is filled
    via ``nisa.dma_transpose`` so the destination's partition axis
    takes values from the source's free axis.
    """
    p, f = sbuf[0][0].shape
    op, of = mem.shape
    expected = (f_count * f, p_count * p) if transpose else (p_count * p, f_count * f)
    if (op, of) != expected:
        raise ValueError(
            f"load_block shape mismatch: sbuf sub-block ({p_count}, {f_count})x({p}, {f}) "
            f"expects mem {expected}, got {mem.shape} (transpose={transpose})"
        )
    for pi in range(p_count):
        for fi in range(f_count):
            dst = sbuf[p_start + pi][f_start + fi][0:p, 0:f]
            if transpose:
                nisa.dma_transpose(dst, mem[fi * f : (fi + 1) * f, pi * p : (pi + 1) * p])
            else:
                nisa.dma_copy(dst, mem[pi * p : (pi + 1) * p, fi * f : (fi + 1) * f])


def stage_block(sbuf: Any, mem: Any, p_start: int, p_count: int, f_start: int, f_count: int) -> None:
    """PSUM → SBUF: copy ``mem`` into the ``[p_start : p_start + p_count][f_start : f_start + f_count]`` sub-block of ``sbuf``."""
    p, f = sbuf[0][0].shape
    op, of = mem.shape
    if op != p_count * p or of != f_count * f:
        raise ValueError(
            f"stage_block shape mismatch: sbuf sub-block ({p_count}, {f_count})x({p}, {f}) "
            f"covers ({p_count * p}, {f_count * f}), mem {mem.shape}"
        )
    for pi in range(p_count):
        for fi in range(f_count):
            nisa.tensor_copy(
                sbuf[p_start + pi][f_start + fi][0:p, 0:f], mem[pi * p : (pi + 1) * p, fi * f : (fi + 1) * f]
            )


def store_block(mem: Any, sbuf: Any, p_start: int, p_count: int, f_start: int, f_count: int) -> None:
    """SBUF → HBM: write the ``[p_start : p_start + p_count][f_start : f_start + f_count]`` sub-block of ``sbuf`` into ``mem``."""
    p, f = sbuf[0][0].shape
    op, of = mem.shape
    if op != p_count * p or of != f_count * f:
        raise ValueError(
            f"store_block shape mismatch: sbuf sub-block ({p_count}, {f_count})x({p}, {f}) "
            f"covers ({p_count * p}, {f_count * f}), mem {mem.shape}"
        )
    for pi in range(p_count):
        for fi in range(f_count):
            nisa.dma_copy(mem[pi * p : (pi + 1) * p, fi * f : (fi + 1) * f], sbuf[p_start + pi][f_start + fi][0:p, 0:f])
