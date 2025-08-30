import math
from typing import Tuple

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np

from autotune.core.tensor import SBUFTensor


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


def save_result_tiles(result, result_tiles: SBUFTensor):
    """
    Store SBUF tiles back into a 2D HBM tensor using global tile indexing.

    The function uses the SBUFTensor's internal tile coordinates to determine
    where each tile should be stored in the result tensor. Each tile is read
    using the read_tile method with global indices and stored at the appropriate
    position in the HBM result tensor.

    +--------------------------------------------------+
    |                                                  |
    |    +--------+--------+----+--------+             |
    |    |Tile 0,0|Tile 0,1|... |Tile 0,n|             |  ← Global tile positions
    |    +--------+--------+----+--------+             |    determined by tile_coordinates
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
        result_tiles: SBUFTensor containing tiled results with internal tile coordinates
                     specifying global tile positions

    Returns:
        None. The result tensor is modified in-place.
    """
    # Get tile sizes and global coordinates from the SBUFTensor
    par_axis = result_tiles.par_axis
    free_axis = result_tiles.free_axis
    row_tile_size = result_tiles.tile_sizes[par_axis]
    column_tile_size = result_tiles.tile_sizes[free_axis]

    # Get global tile coordinates
    par_tile_coords = result_tiles.tile_coordinates[par_axis]
    free_tile_coords = result_tiles.tile_coordinates[free_axis]

    start_row_tile = par_tile_coords["start_tile_index"]
    num_row_tiles = par_tile_coords["num_tiles"]
    start_column_tile = free_tile_coords["start_tile_index"]
    num_column_tiles = free_tile_coords["num_tiles"]

    max_rows, max_cols = result.shape
    idx_res = nl.mgrid[0:row_tile_size, 0:column_tile_size]

    for row_tile_offset in nl.affine_range(num_row_tiles):
        # Calculate global row tile index
        global_row_tile = start_row_tile + row_tile_offset
        row_start = global_row_tile * row_tile_size
        row_indices = row_start + idx_res.p

        for column_tile_offset in nl.affine_range(num_column_tiles):
            # Calculate global column tile index
            global_column_tile = start_column_tile + column_tile_offset
            column_start = global_column_tile * column_tile_size
            column_indices = column_start + idx_res.x

            # Read tile using global indices
            tile_indices = {par_axis: global_row_tile, free_axis: global_column_tile}
            tile_data = result_tiles.read_tile(tile_indices)

            # Store tile to result tensor
            nl.store(
                result[row_indices, column_indices],
                value=tile_data,
                mask=(row_indices < max_rows) & (column_indices < max_cols),
            )


def copy_block(src_block, dest_block):
    block_size, par_size, free_size = src_block.shape
    _block_size, _par_size, _free_size = dest_block.shape
    assert block_size == _block_size and par_size == _par_size and free_size == _free_size

    for block_id in nl.affine_range(block_size):
        dest_block[block_id] = nl.copy(src_block[block_id], dtype=src_block.dtype)
