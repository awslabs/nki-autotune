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
def matmul_lhsT_rhsT_nkigym(lhs_T, rhs_T):
    assert lhs_T.shape == (2048, 2048)
    assert rhs_T.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    # Group 0: dma_transpose [dims: d0, d2]
    sbuf_rhs = [[nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(4)] for _ in range(16)]
    load_block(sbuf_rhs, rhs_T[0:2048, 0:2048], 0, 16, 0, 4, transpose=True)
    for i_block_d0 in range(1):
        for i_ltile_d0 in range(4):
            for i_block_d2 in range(16):
                for i_ltile_d2 in range(1):
                    pass

    # Group 1: dma_load [dims: d2, d3]
    sbuf_lhs_T = [[nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)] for _ in range(16)]
    load_block(sbuf_lhs_T, lhs_T[0:2048, 0:2048], 0, 16, 0, 16)
    for i_block_d2 in range(16):
        for i_ltile_d2 in range(1):
            for i_block_d3 in range(4):
                for i_ltile_d3 in range(4):
                    pass

    # Group 2: nc_matmul [dims: d3, d0, d2]
    sbuf_output = [[nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(4)] for _ in range(16)]
    psum_output = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    for i_block_d3 in range(4):
        for i_ltile_d3 in range(4):
            for i_block_d0 in range(1):
                for i_ltile_d0 in range(4):
                    nisa.memset(psum_output[0:128, 0:512], 0.0)
                    for i_block_d2 in range(16):
                        for i_ltile_d2 in range(1):
                            nisa.nc_matmul(
                                dst=psum_output[0:128, 0:512],
                                stationary=sbuf_lhs_T[i_block_d2][i_block_d3 * 4 + i_ltile_d3][0:128, 0:128],
                                moving=sbuf_rhs[i_block_d2][i_ltile_d0][0:128, 0:512],
                            )
                    stage_block(sbuf_output, psum_output, i_block_d3 * 4 + i_ltile_d3, 1, i_ltile_d0, 1)

    # Group 3: dma_store [dims: d3, d0]
    store_block(output[0:2048, 0:2048], sbuf_output, 0, 16, 0, 4)
    for i_block_d3 in range(4):
        for i_ltile_d3 in range(4):
            for i_block_d0 in range(1):
                for i_ltile_d0 in range(4):
                    pass

    return output
