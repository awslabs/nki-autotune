from typing import Tuple

import neuronxcc.nki.language as nl


def load_tensor_block(input_tensor, ofs: Tuple[int, int], load_shape: Tuple[int, nl.par_dim, int]):
    """
    Load a 2D rectangle region from the input HBM tensor to SBUF.
    The location of the 2D region is offset by (ofs[0], ofs[1]) at its upper left corner.
    The size of the 2D region to load into SBUF is (block_size * par_size, free_size).
    Load the input HBM tensor by (par_size, free_size) tiles in parallel in the block dimension.
    Output SBUF tensor has a shape of (block_size, par_size, free_size).

    +------------------+
    |                  |
    |    +--------+    |  ← Starting at (ofs[0], ofs[1])
    |    |Tile 0  |    |
    |    |Tile 1  |    |  Each tile is (par_size * free_size)
    |    |  ...   |    |
    |    |Tile N-1|    |  N = block_size
    |    +--------+    |
    |                  |
    +------------------+

    Args:
        input_tensor: the input 2D HBM tensor
        ofs: location offsets in the 2D HBM tensor dimensions
        load_shape: (block_size, par_dim(par_size), free_size)

    Returns:
        Loaded tiles in SBUF in the shape of load_shape
    """
    assert len(ofs) == 2, f"'ofs' expects (ofs_0, ofs_1). Received {ofs}."
    assert len(load_shape) == 3, f"'load_shape' expects (block, par, free). Received {load_shape}."
    max_rows, max_cols = input_tensor.shape
    load_block_size, load_par_size, load_free_size = load_shape
    tile_index = nl.mgrid[0:load_par_size, 0:load_free_size]
    loaded_tensor = nl.ndarray(
        (load_block_size, nl.par_dim(load_par_size), load_free_size), dtype=input_tensor.dtype, buffer=nl.sbuf
    )
    for block_id in nl.affine_range(load_block_size):
        row_indices = ofs[0] + block_id * load_par_size + tile_index.p
        col_indices = ofs[1] + tile_index.x
        loaded_tensor[block_id, tile_index.p, tile_index.x] = nl.load(
            input_tensor[row_indices, col_indices], mask=(row_indices < max_rows) & (col_indices < max_cols)
        )
    return loaded_tensor


def save_result_dma(result, result_tiles, block_id, m_ofs, n_ofs, TILE_K):
    _, num_m_tiles, num_n_tiles, _, TILE_N = result_tiles.shape

    for tile_id_M in nl.affine_range(num_m_tiles):
        idx_res = nl.mgrid[0:TILE_K, 0:TILE_N]
        idx_res_packed = nl.mgrid[0:TILE_K, 0 : TILE_N * num_n_tiles]

        result_packed = nl.ndarray((TILE_K, TILE_N * num_n_tiles), dtype=result_tiles.dtype, buffer=nl.sbuf)

        # coalesce result tiles for better DMA performance
        for tile_id_N in nl.affine_range(num_n_tiles):
            result_packed[idx_res.p, tile_id_N * TILE_N + idx_res.x] = nl.copy(
                result_tiles[block_id, tile_id_M, tile_id_N, idx_res.p, idx_res.x]
            )

        nl.store(
            result[(m_ofs + tile_id_M) * TILE_K + idx_res_packed.p, n_ofs + idx_res_packed.x],
            value=result_packed[idx_res_packed.p, idx_res_packed.x],
        )


def save_result_acc(result, result_tiles, BLOCK_M, BLOCK_N):
    NUM_BLOCK_K, NUM_BLOCK_M, NUM_BLOCK_N, num_m_tiles, num_n_tiles, TILE_M, TILE_N = result_tiles.shape

    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    for block_id_M in nl.affine_range(NUM_BLOCK_M):
        m_ofs = block_id_M * BLOCK_M
        for block_id_N in nl.affine_range(NUM_BLOCK_N):
            n_ofs = block_id_N * BLOCK_N
            for tile_id_M in nl.affine_range(num_m_tiles):
                for tile_id_N in nl.affine_range(num_n_tiles):
                    result_acc = nl.zeros(
                        (num_m_tiles, num_n_tiles, nl.par_dim(TILE_M), TILE_N), dtype=result_tiles.dtype, buffer=nl.sbuf
                    )
                    for block_id_K in nl.affine_range(NUM_BLOCK_K):
                        result_acc[tile_id_M, tile_id_N] += result_tiles[
                            block_id_K, block_id_M, block_id_N, tile_id_M, tile_id_N
                        ]

                    nl.store(
                        result[m_ofs + tile_id_M * TILE_M + idx_res.p, n_ofs + tile_id_N * TILE_N + idx_res.x],
                        value=result_acc[tile_id_M, tile_id_N],
                    )


def save_result_block(result, result_block, m_ofs: int, n_ofs: int):
    """
    Store result_block into result
    Args:
    result: M, N
    result_block: TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, nl.par_dim(TILE_M), TILE_N
    """
    TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILE_M, TILE_N = result_block.shape

    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    for tile_id_M in nl.affine_range(TILES_IN_BLOCK_M):
        m_start = m_ofs + tile_id_M * TILE_M
        for tile_id_N in nl.affine_range(TILES_IN_BLOCK_N):
            n_start = n_ofs + tile_id_N * TILE_N
            nl.store(result[m_start + idx_res.p, n_start + idx_res.x], value=result_block[tile_id_M, tile_id_N])
