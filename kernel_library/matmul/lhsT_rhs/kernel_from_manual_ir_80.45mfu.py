from typing import Any

import nki
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
    sbuf_moving: Any,
    k_start: int,
    k_count: int,
    tile_m: int,
    tile_n: int,
) -> None:
    """Two-level matmul block: absorbs M-ltile / N-ltile / K-ltile iteration.

    ``sbuf_stationary`` and ``sbuf_moving`` use wide leaves of
    ``(TILE_K, p_count * tile_m)`` and ``(TILE_K, f_count * tile_n)``
    respectively (one leaf per inner-K slot); the gadget slices
    inside each leaf to reach the ``(pi, fi)`` output tile.

    For each output tile ``(pi, fi)`` in
    ``[p_start, p_start+p_count) x [f_start, f_start+f_count)``:

      1. Zero a reused PSUM scratch tile.
      2. Reduce ``k_count`` inner-K ``nc_matmul`` calls into PSUM.
      3. ``tensor_copy`` PSUM → an SBUF scratch, then
         ``tensor_tensor`` add into the running accumulator
         ``sbuf_out[p_start + pi][f_start + fi]``.

    Caller must pre-memset every ``sbuf_out`` leaf before the
    first outer-K invocation.
    """
    tile_k = sbuf_stationary[k_start][0].shape[0]
    psum_tile = nl.ndarray((tile_m, tile_n), dtype=nl.float32, buffer=nl.psum)
    acc_tile = nl.ndarray((tile_m, tile_n), dtype=sbuf_out[0][0].dtype, buffer=nl.sbuf)
    for pi in range(p_count):
        for fi in range(f_count):
            nisa.memset(psum_tile[0:tile_m, 0:tile_n], 0.0)
            for ki in range(k_count):
                nisa.nc_matmul(
                    dst=psum_tile[0:tile_m, 0:tile_n],
                    stationary=sbuf_stationary[k_start + ki][0][0:tile_k, pi * tile_m : pi * tile_m + tile_m],
                    moving=sbuf_moving[k_start + ki][0][0:tile_k, fi * tile_n : fi * tile_n + tile_n],
                )
            nisa.tensor_copy(acc_tile[0:tile_m, 0:tile_n], psum_tile[0:tile_m, 0:tile_n])
            nisa.tensor_tensor(
                sbuf_out[p_start + pi][f_start + fi][0:tile_m, 0:tile_n],
                sbuf_out[p_start + pi][f_start + fi][0:tile_m, 0:tile_n],
                acc_tile[0:tile_m, 0:tile_n],
                op=nl.add,
            )


@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    # Group 0: dma_load, dma_load, nc_matmul, dma_store [dims: d2, d0, d1]
    for i_block_d2 in range(4):
        sbuf_output = [[nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)] for _ in range(16)]
        for i_pmr in range(16):
            for i_fmr in range(1):
                nisa.memset(sbuf_output[i_pmr][i_fmr][0:128, 0:512], 0.0)
        for i_block_d0 in range(2):
            sbuf_rhs = [[nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)] for _ in range(8)]
            load_block(
                sbuf_rhs,
                rhs[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d2 * 512 : i_block_d2 * 512 + 512],
                0,
                8,
                0,
                1,
            )
            for i_block_d1 in range(4):
                sbuf_lhs_T = [
                    [nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)] for _ in range(8)
                ]
                load_block(
                    sbuf_lhs_T,
                    lhs_T[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d1 * 512 : i_block_d1 * 512 + 512],
                    0,
                    8,
                    0,
                    1,
                )
                matmul_block(sbuf_output, i_block_d1 * 4, 4, 0, 1, sbuf_lhs_T, sbuf_rhs, 0, 8, 128, 512)
        store_block(output[0:2048, i_block_d2 * 512 : i_block_d2 * 512 + 512], sbuf_output, 0, 16, 0, 1)

    return output
