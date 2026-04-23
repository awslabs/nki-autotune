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


@nki.jit
def double_matmul_nkigym(Q, K, V):
    assert Q.shape == (2048, 2048)
    assert K.shape == (2048, 2048)
    assert V.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    # Group 0: dma_transpose, dma_transpose, nc_matmul, nc_transpose [dims: d2, d0, d1]
    sbuf_S_t = [[nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)] for _ in range(16)]
    sbuf_Q_t = [[nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)] for _ in range(8)]
    sbuf_K_t = [[nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)] for _ in range(16)]
    sbuf_S = [[nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)] for _ in range(1)]
    psum_S = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    psum_S_t = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
    for i_block_d2 in range(1):
        for i_ltile_d2 in range(4):
            load_block(
                sbuf_K_t,
                K[i_block_d2 * 2048 + i_ltile_d2 * 512 : i_block_d2 * 2048 + i_ltile_d2 * 512 + 512, 0:2048],
                0,
                16,
                0,
                1,
                transpose=True,
            )
            for i_block_d0 in range(16):
                for i_ltile_d0 in range(1):
                    nisa.memset(psum_S[0:128, 0:512], 0.0)
                    for i_block_d1 in range(2):
                        load_block(
                            sbuf_Q_t,
                            Q[
                                i_block_d0 * 128 + i_ltile_d0 * 128 : i_block_d0 * 128 + i_ltile_d0 * 128 + 128,
                                i_block_d1 * 1024 : i_block_d1 * 1024 + 1024,
                            ],
                            0,
                            8,
                            0,
                            1,
                            transpose=True,
                        )
                        for i_ltile_d1 in range(8):
                            nisa.nc_matmul(
                                dst=psum_S[0:128, 0:512],
                                stationary=sbuf_Q_t[i_ltile_d1][0][0:128, 0:128],
                                moving=sbuf_K_t[i_block_d1 * 8 + i_ltile_d1][0][0:128, 0:512],
                            )
                    stage_block(sbuf_S, psum_S, 0, 1, 0, 1)
                    for i_ptile_d2 in range(4):
                        nisa.nc_transpose(
                            psum_S_t[0:128, 0:128], sbuf_S[0][0][0:128, i_ptile_d2 * 128 : i_ptile_d2 * 128 + 128]
                        )
                        nisa.tensor_copy(
                            sbuf_S_t[i_ltile_d2 * 4 + i_ptile_d2][i_block_d0][0:128, 0:128], psum_S_t[0:128, 0:128]
                        )

    # Group 1: dma_load, nc_matmul [dims: d4, d0, d2]
    sbuf_output = [[nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(4)] for _ in range(16)]
    sbuf_V = [[nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(2)] for _ in range(16)]
    psum_output = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    for i_block_d4 in range(2):
        load_block(sbuf_V, V[0:2048, i_block_d4 * 1024 : i_block_d4 * 1024 + 1024], 0, 16, 0, 2)
        for i_ltile_d4 in range(2):
            for i_block_d0 in range(16):
                for i_ltile_d0 in range(1):
                    nisa.memset(psum_output[0:128, 0:512], 0.0)
                    for i_block_d2 in range(1):
                        for i_ltile_d2 in range(4):
                            for i_ptile_d2 in range(4):
                                nisa.nc_matmul(
                                    dst=psum_output[0:128, 0:512],
                                    stationary=sbuf_S_t[i_ltile_d2 * 4 + i_ptile_d2][i_block_d0][0:128, 0:128],
                                    moving=sbuf_V[i_ltile_d2 * 4 + i_ptile_d2][i_ltile_d4][0:128, 0:512],
                                )
                    stage_block(sbuf_output, psum_output, i_block_d0, 1, i_block_d4 * 2 + i_ltile_d4, 1)

    # Group 2: dma_store [dims: d0, d4]
    store_block(output[0:2048, 0:2048], sbuf_output, 0, 16, 0, 4)
    for i_block_d0 in range(16):
        for i_ltile_d0 in range(1):
            for i_block_d4 in range(2):
                for i_ltile_d4 in range(2):
                    pass

    return output
