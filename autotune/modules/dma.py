import math
from typing import Tuple

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np


def load_tensor_block(input_tensor, dim_0: Tuple[int, int, int], dim_1: Tuple[int, int, int], transpose: bool):
    """
    Args:
        input_tensor: the input 2D HBM tensor
        dim_0, dim_1: (tile size, start tile index, number of tiles)
        transpose: whether to transpose each loaded tile

    Returns:
        Loaded tiles in SBUF in the shape of (tile_size_0, num_tiles_0, num_tiles_1, tile_size_1)
    """
    assert len(dim_0) == 3, f"'dim_0' expects (tile size, start tile index, number of tiles). Received {dim_0}."
    assert len(dim_1) == 3, f"'dim_1' expects (tile size, start tile index, number of tiles). Received {dim_1}."
    assert len(input_tensor.shape) == 2, f"Expects 2D input tensor. Received {input_tensor.shape}."
    max_rows, max_cols = input_tensor.shape
    row_tile_size, row_tile_id_start, row_num_tiles = dim_0
    column_tile_size, column_tile_id_start, column_num_tiles = dim_1
    pmax = nl.tile_size.pmax
    row_transp_num_tiles = math.ceil(row_tile_size / pmax)
    column_transp_num_tiles = math.ceil(column_tile_size / pmax)
    transp_index = nl.mgrid[0:pmax, 0:pmax]
    if nisa.get_nc_version() == nisa.nc_version.gen3:
        tileT_dtype = input_tensor.dtype
    else:
        tileT_dtype = np.float32
    tile_index = nl.mgrid[0:row_tile_size, 0:column_tile_size]
    block = nl.ndarray(
        (nl.par_dim(row_tile_size), row_num_tiles, column_num_tiles, column_tile_size),
        dtype=input_tensor.dtype,
        buffer=nl.sbuf,
    )
    for row_tile_id in nl.affine_range(row_num_tiles):
        row_start = (row_tile_id_start + row_tile_id) * row_tile_size
        row_indices = row_start + tile_index.p
        row_mask = row_indices < max_rows
        for column_tile_id in nl.affine_range(column_num_tiles):
            column_start = (column_tile_id_start + column_tile_id) * column_tile_size
            column_indices = column_start + tile_index.x
            column_mask = column_indices < max_cols
            block[tile_index.p, row_tile_id, column_tile_id, tile_index.x] = nl.load(
                input_tensor[row_indices, column_indices], mask=row_mask & column_mask
            )
            if transpose:
                for row_transp_tile_id in nl.affine_range(row_transp_num_tiles):
                    row_ofs = row_transp_tile_id * pmax
                    transp_row_indices = row_ofs + transp_index.p
                    transp_row_mask = (row_start + transp_row_indices) < max_rows
                    for column_transp_tile_id in nl.affine_range(column_transp_num_tiles):
                        column_ofs = column_transp_tile_id * pmax
                        transp_column_indices = column_ofs + transp_index.x
                        transp_column_mask = (column_start + transp_column_indices) < max_cols
                        transp_mask = transp_row_mask & transp_column_mask
                        tileT = nl.ndarray((nl.par_dim(pmax), pmax), dtype=tileT_dtype, buffer=nl.psum)
                        tileT[transp_index.p, transp_index.x] = nisa.nc_transpose(
                            block[transp_row_indices, row_tile_id, column_tile_id, transp_column_indices],
                            mask=transp_mask,
                        )
                        block[transp_row_indices, row_tile_id, column_tile_id, transp_column_indices] = nl.copy(
                            tileT, dtype=block.dtype, mask=transp_mask
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


def save_result_block(result, result_block, dst_ofs: Tuple[int, int]):
    """
    Store a 4D SBUF block of tiled results back into a 2D HBM tensor.
    The result_block tensor contains data organized in tiles that need to be stored in the
    appropriate locations in the result tensor. The starting position for storing tiles is
    determined by tile_index_ofs, which specifies the tile offset in both dimensions.

    +--------------------------------------------------+
    |                                                  |
    |    +--------+--------+----+--------+             |
    |    |Tile 0,0|Tile 0,1|... |Tile 0,n|             |  ← Starting at position
    |    +--------+--------+----+--------+             |    (tile_index_ofs[0] * TILE_M, tile_index_ofs[1] * TILE_N)
    |    |Tile 1,0|Tile 1,1|... |Tile 1,n|             |
    |    +--------+--------+----+--------+             |  Each tile is (TILE_M, TILE_N) in size
    |    |   ...  |   ...  |... |   ...  |             |
    |    +--------+--------+----+--------+             |
    |    |Tile m,0|Tile m,1|... |Tile m,n|             |  m=TILES_IN_M-1, n=TILES_IN_N-1
    |    +--------+--------+----+--------+             |
    |                                                  |
    +--------------------------------------------------+

    Args:
        result: The destination 2D tensor with shape (M, N) where tiles will be stored
        result_block: The source 4D tensor containing tiled results with shape
                     (TILE_M, TILES_IN_M, TILES_IN_N, TILE_N), where:
                     - TILE_M: Height of each tile
                     - TILES_IN_M: Number of tiles in the M dimension (rows)
                     - TILES_IN_N: Number of tiles in the N dimension (columns)
                     - TILE_N: Width of each tile
        dst_ofs: A tuple of (start row tile, start column tile) specifying the target location in the result tensor where tiles should be stored

    Returns:
        None. The result tensor is modified in-place.
    """
    start_row_tile, start_column_tile = dst_ofs
    row_tile_size, num_row_tiles, num_column_tiles, column_tile_size = result_block.shape
    max_rows, max_cols = result.shape
    idx_res = nl.mgrid[0:row_tile_size, 0:column_tile_size]
    for row_tile_id in nl.affine_range(num_row_tiles):
        row_start = (start_row_tile + row_tile_id) * row_tile_size
        row_indices = row_start + idx_res.p
        for column_tile_id in nl.affine_range(num_column_tiles):
            column_start = (start_column_tile + column_tile_id) * column_tile_size
            column_indices = column_start + idx_res.x
            nl.store(
                result[row_indices, column_indices],
                value=result_block[idx_res.p, row_tile_id, column_tile_id, idx_res.x],
                mask=(row_indices < max_rows) & (column_indices < max_cols),
            )


def copy_block(src_block, dest_block):
    block_size, par_size, free_size = src_block.shape
    _block_size, _par_size, _free_size = dest_block.shape
    assert block_size == _block_size and par_size == _par_size and free_size == _free_size

    for block_id in nl.affine_range(block_size):
        dest_block[block_id] = nl.copy(src_block[block_id], dtype=src_block.dtype)
