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
import nki.language as nl


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


def memset_block(buffer: Any, value: float) -> None:
    """Fill every leaf of a nested SBUF / PSUM list buffer with ``value``.

    ``buffer`` is the ``[NP_list][NF_list]`` nested list used
    throughout codegen; each leaf is a 2D ``nl.ndarray(P, F)``.
    Iterates every list slot and emits one ``nisa.memset`` per
    leaf covering the full ``(0:P, 0:F)`` region.
    """
    np_slots = len(buffer)
    nf_slots = len(buffer[0]) if np_slots > 0 else 0
    if np_slots == 0 or nf_slots == 0:
        raise ValueError(f"memset_block got empty buffer with shape ({np_slots}, {nf_slots})")
    p, f = buffer[0][0].shape
    for pi in range(np_slots):
        for fi in range(nf_slots):
            nisa.memset(buffer[pi][fi][0:p, 0:f], value)


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


def matmul_block(
    sbuf_out: Any,
    p_start: int,
    p_count: int,
    f_start: int,
    f_count: int,
    sbuf_stationary: Any,
    s_p_slot: int,
    sbuf_moving: Any,
    m_f_slot: int,
    k_start: int,
    k_count: int,
    tile_m: int,
    tile_n: int,
) -> None:
    """Two-level matmul block: absorbs M-ltile / N-ltile / K-ltile iteration.

    ``sbuf_stationary[k][s_p_slot]`` is one SBUF leaf of shape
    ``(TILE_K, p_count * tile_m)``; ``sbuf_moving[k][m_f_slot]``
    is one leaf of shape ``(TILE_K, f_count * tile_n)``. The
    gadget slices inside each leaf to reach the ``(pi, fi)``
    output tile.

    For each ``(pi, fi)`` in
    ``[p_start, p_start+p_count) x [f_start, f_start+f_count)``:

      1. Zero a reused PSUM scratch tile.
      2. Reduce ``k_count`` inner-K ``nc_matmul`` calls into PSUM.
      3. ``tensor_copy`` PSUM → an SBUF scratch, then
         ``tensor_tensor`` add into the running accumulator
         ``sbuf_out[p_start + pi][f_start + fi]``.

    Caller must pre-memset every ``sbuf_out`` leaf before the
    first outer-K invocation.
    """
    tile_k = sbuf_stationary[k_start][s_p_slot].shape[0]
    psum_tile = nl.ndarray((tile_m, tile_n), dtype=nl.float32, buffer=nl.psum)
    acc_tile = nl.ndarray((tile_m, tile_n), dtype=sbuf_out[0][0].dtype, buffer=nl.sbuf)
    for pi in range(p_count):
        for fi in range(f_count):
            nisa.memset(psum_tile[0:tile_m, 0:tile_n], 0.0)
            for ki in range(k_count):
                nisa.nc_matmul(
                    dst=psum_tile[0:tile_m, 0:tile_n],
                    stationary=sbuf_stationary[k_start + ki][s_p_slot][0:tile_k, pi * tile_m : pi * tile_m + tile_m],
                    moving=sbuf_moving[k_start + ki][m_f_slot][0:tile_k, fi * tile_n : fi * tile_n + tile_n],
                )
            nisa.tensor_copy(acc_tile[0:tile_m, 0:tile_n], psum_tile[0:tile_m, 0:tile_n])
            nisa.tensor_tensor(
                sbuf_out[p_start + pi][f_start + fi][0:tile_m, 0:tile_n],
                sbuf_out[p_start + pi][f_start + fi][0:tile_m, 0:tile_n],
                acc_tile[0:tile_m, 0:tile_n],
                op=nl.add,
            )
