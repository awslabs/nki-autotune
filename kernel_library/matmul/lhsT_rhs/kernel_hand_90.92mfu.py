import nki
import nki.isa as nisa
import nki.language as nl


def matmul_block_single(sbuf_out, sbuf_lhs_T, sbuf_rhs):
    """Single-level matmul: all K in one PSUM accum, one tensor_copy per output tile."""
    _TILE_N_MAX = 512
    num_m_tiles = len(sbuf_out)
    num_k_tiles = len(sbuf_lhs_T)
    tile_k = sbuf_lhs_T[0].shape[0]
    tile_m = sbuf_out[0].shape[0]
    free_width = sbuf_rhs[0].shape[1]
    tile_n = free_width if free_width <= _TILE_N_MAX else _TILE_N_MAX
    num_n_tiles = free_width // tile_n
    for m_idx in range(num_m_tiles):
        for n_idx in range(num_n_tiles):
            psum_tile = nl.ndarray((tile_m, tile_n), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(psum_tile[0:tile_m, 0:tile_n], 0.0)
            for k_idx in range(num_k_tiles):
                nisa.nc_matmul(
                    dst=psum_tile[0:tile_m, 0:tile_n],
                    stationary=sbuf_lhs_T[k_idx][0:tile_k, m_idx * tile_m : (m_idx + 1) * tile_m],
                    moving=sbuf_rhs[k_idx][0:tile_k, n_idx * tile_n : (n_idx + 1) * tile_n],
                )
            nisa.tensor_copy(
                sbuf_out[m_idx][0:tile_m, n_idx * tile_n : (n_idx + 1) * tile_n], psum_tile[0:tile_m, 0:tile_n]
            )


def allocate_buffers(p_tile_size, num_p_tiles, f_tile_size, num_f_tiles, loc, dtype, num_p_buffers, num_f_buffers):
    """Nested lists of 2D leaves with per-axis multi-buffer counts."""
    leaf_shape = (p_tile_size, f_tile_size * num_f_tiles)
    p_count = 1 if num_p_buffers is None else num_p_buffers
    f_count = 1 if num_f_buffers is None else num_f_buffers
    nested = [
        [[nl.ndarray(leaf_shape, dtype=dtype, buffer=loc) for _ in range(num_p_tiles)] for _ in range(f_count)]
        for _ in range(p_count)
    ]
    result = nested
    if num_f_buffers is None:
        result = [row[0] for row in result]
    if num_p_buffers is None:
        result = result[0]
    return result


def memset_buffers(sbuf, value):
    p_tile, f_tile = sbuf[0].shape
    for leaf in sbuf:
        nisa.memset(leaf[0:p_tile, 0:f_tile], value)


def load_block(sbuf, mem_slice, transpose):
    num_p_tiles = len(sbuf)
    p_tile, f_tile = sbuf[0].shape
    for pt in range(num_p_tiles):
        dst = sbuf[pt][0:p_tile, 0:f_tile]
        if transpose:
            nisa.dma_transpose(dst, mem_slice[0:f_tile, pt * p_tile : (pt + 1) * p_tile])
        else:
            nisa.dma_copy(dst, mem_slice[pt * p_tile : (pt + 1) * p_tile, 0:f_tile])


def store_block(mem_slice, sbuf):
    num_p_tiles = len(sbuf)
    p_tile, f_tile = sbuf[0].shape
    for pt in range(num_p_tiles):
        nisa.dma_copy(mem_slice[pt * p_tile : (pt + 1) * p_tile, 0:f_tile], sbuf[pt][0:p_tile, 0:f_tile])


def matmul_block(sbuf_out, sbuf_lhs_T, sbuf_rhs):
    _TILE_N_MAX = 512
    num_m_tiles = len(sbuf_out)
    num_k_tiles = len(sbuf_lhs_T)
    tile_k = sbuf_lhs_T[0].shape[0]
    tile_m = sbuf_out[0].shape[0]
    free_width = sbuf_rhs[0].shape[1]
    tile_n = free_width if free_width <= _TILE_N_MAX else _TILE_N_MAX
    num_n_tiles = free_width // tile_n
    for m_idx in range(num_m_tiles):
        for n_idx in range(num_n_tiles):
            psum_tile = nl.ndarray((tile_m, tile_n), dtype=nl.float32, buffer=nl.psum)
            acc_tile = nl.ndarray((tile_m, tile_n), dtype=sbuf_out[0].dtype, buffer=nl.sbuf)
            nisa.memset(psum_tile[0:tile_m, 0:tile_n], 0.0)
            for k_idx in range(num_k_tiles):
                nisa.nc_matmul(
                    dst=psum_tile[0:tile_m, 0:tile_n],
                    stationary=sbuf_lhs_T[k_idx][0:tile_k, m_idx * tile_m : (m_idx + 1) * tile_m],
                    moving=sbuf_rhs[k_idx][0:tile_k, n_idx * tile_n : (n_idx + 1) * tile_n],
                )
            nisa.tensor_copy(acc_tile[0:tile_m, 0:tile_n], psum_tile[0:tile_m, 0:tile_n])
            nisa.tensor_tensor(
                sbuf_out[m_idx][0:tile_m, n_idx * tile_n : (n_idx + 1) * tile_n],
                sbuf_out[m_idx][0:tile_m, n_idx * tile_n : (n_idx + 1) * tile_n],
                acc_tile[0:tile_m, 0:tile_n],
                op=nl.add,
            )


@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    sbuf_output = allocate_buffers(128, 16, 512, 1, nl.sbuf, nl.bfloat16, num_p_buffers=None, num_f_buffers=4)

    for i_block_d2 in range(4):
        sbuf_lhs_T = allocate_buffers(128, 8, 128, 4, nl.sbuf, nl.bfloat16, num_p_buffers=2, num_f_buffers=4)
        sbuf_rhs = allocate_buffers(128, 8, 512, 1, nl.sbuf, nl.bfloat16, num_p_buffers=2, num_f_buffers=None)
        cur_sbuf_output = sbuf_output[(i_block_d2) % 4]
        memset_buffers(cur_sbuf_output, 0.0)
        for i_block_d0 in range(2):
            cur_sbuf_rhs = sbuf_rhs[(i_block_d0) % 2]
            load_block(
                cur_sbuf_rhs,
                rhs[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d2 * 512 : i_block_d2 * 512 + 512],
                transpose=False,
            )
            for i_block_d1 in range(4):
                cur_sbuf_lhs_T = sbuf_lhs_T[(i_block_d0) % 2][(i_block_d1) % 4]
                load_block(
                    cur_sbuf_lhs_T,
                    lhs_T[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d1 * 512 : i_block_d1 * 512 + 512],
                    transpose=False,
                )
                matmul_block(cur_sbuf_output[i_block_d1 * 4 : i_block_d1 * 4 + 4], cur_sbuf_lhs_T, cur_sbuf_rhs)
        store_block(output[0:2048, i_block_d2 * 512 : i_block_d2 * 512 + 512], cur_sbuf_output)

    return output
