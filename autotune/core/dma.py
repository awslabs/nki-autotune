from typing import Tuple

import neuronxcc.nki.language as nl


def load_tensor_block(input_tensor, ofs: Tuple[int, int], load_shape: Tuple[int, int, int, int]):
    """
    Load a 2D rectangle region from the input HBM tensor to SBUF.
    The location of the 2D region is offset by (row_offset, column_offset) at its upper left corner.
    The size of the 2D region to load into SBUF is (row_tile_size * row_num_tiles, column_tile_size * column_num_tiles).
    Load the input HBM tensor by (row_tile_size, column_tile_size) tiles in parallel.
    Output SBUF tensor has a shape of load_shape.

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
        load_shape: (row_tile_size, row_num_tiles, column_num_tiles, column_tile_size)

    Returns:
        Loaded tiles in SBUF in the shape of load_shape
    """
    assert len(ofs) == 2, f"'ofs' expects (row_offset, column_offset). Received {ofs}."
    assert (
        len(load_shape) == 4
    ), f"'load_shape' expects (row_tile_size, row_num_tiles, column_num_tiles, column_tile_size). Received {load_shape}."
    max_rows, max_cols = input_tensor.shape
    row_tile_size, row_num_tiles, column_num_tiles, column_tile_size = load_shape
    tile_index = nl.mgrid[0:row_tile_size, 0:column_tile_size]
    block = nl.ndarray(
        (nl.par_dim(row_tile_size), row_num_tiles, column_num_tiles, column_tile_size),
        dtype=input_tensor.dtype,
        buffer=nl.sbuf,
    )
    for row_tile_id in nl.affine_range(row_num_tiles):
        for column_tile_id in nl.affine_range(column_num_tiles):
            row_indices = ofs[0] + row_tile_id * row_tile_size + tile_index.p
            col_indices = ofs[1] + column_tile_id * column_tile_size + tile_index.x
            block[tile_index.p, row_tile_id, column_tile_id, tile_index.x] = nl.load(
                input_tensor[row_indices, col_indices], mask=(row_indices < max_rows) & (col_indices < max_cols)
            )
    return block


def save_result_dma(result, result_blocks, block_id_M: int):
    M, N = result.shape
    TILE_M, NUM_BLOCK_N, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILE_N = result_blocks.shape
    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    idx_res_packed = nl.mgrid[0:TILE_M, 0:N]
    for tile_id_M in nl.affine_range(TILES_IN_BLOCK_M):
        m_ofs = (block_id_M * TILES_IN_BLOCK_M + tile_id_M) * TILE_M
        result_blocks_packed = nl.ndarray((TILE_M, N), dtype=result_blocks.dtype, buffer=nl.sbuf)
        for block_id_N in nl.affine_range(NUM_BLOCK_N):
            for tile_id_N in nl.affine_range(TILES_IN_BLOCK_N):
                n_ofs = (block_id_N * TILES_IN_BLOCK_N + tile_id_N) * TILE_N
                result_blocks_packed[idx_res.p, n_ofs + idx_res.x] = nl.copy(
                    result_blocks[idx_res.p, block_id_N, tile_id_M, tile_id_N, idx_res.x]
                )
        nl.store(result[m_ofs + idx_res_packed.p, idx_res_packed.x], value=result_blocks_packed)


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
    M, N = result.shape
    TILE_M, TILE_N, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N = result_block.shape

    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    for tile_id_M in nl.affine_range(TILES_IN_BLOCK_M):
        m_start = m_ofs + tile_id_M * TILE_M
        for tile_id_N in nl.affine_range(TILES_IN_BLOCK_N):
            n_start = n_ofs + tile_id_N * TILE_N
            nl.store(
                result[m_start + idx_res.p, n_start + idx_res.x],
                value=result_block[idx_res.p, idx_res.x, tile_id_M, tile_id_N],
            )
