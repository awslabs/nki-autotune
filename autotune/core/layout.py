import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
from neuronxcc.nki.typing import tensor


def transpose_tiles_in_block(block: tensor):
    """
    Transpose the (pmax, pmax) tiles in a block in place
    all PE array ops must output to FP32 on trn1 but must match input dtype in trn2
    """
    if nisa.get_nc_version() == nisa.nc_version.gen3:
        blockT_dtype = block.dtype
    else:
        blockT_dtype = np.float32
    assert (
        len(block.shape) == 4
    ), f"Expect (row_tile_size, column_tile_size, row_num_tiles, column_num_tiles). Received {block.shape}."
    row_tile_size, column_tile_size, row_num_tiles, column_num_tiles = block.shape
    pmax = nl.tile_size.pmax
    index = nl.mgrid[0:pmax, 0:pmax]
    for row_tile_id in nl.affine_range(row_num_tiles):
        for column_tile_id in nl.affine_range(column_num_tiles):
            for row_transp_tile_id in nl.affine_range(row_tile_size // pmax):
                row_ofs = row_transp_tile_id * pmax
                for column_transp_tile_id in nl.affine_range(column_tile_size // pmax):
                    column_ofs = column_transp_tile_id * pmax
                    tileT = nl.ndarray((nl.par_dim(pmax), pmax), dtype=blockT_dtype, buffer=nl.psum)
                    tileT[index.p, index.x] = nisa.nc_transpose(
                        block[row_ofs + index.p, column_ofs + index.x, row_tile_id, column_tile_id]
                    )
                    block[row_ofs + index.p, column_ofs + index.x, row_tile_id, column_tile_id] = nl.copy(
                        tileT, dtype=block.dtype
                    )


def transpose_tile(tile: tensor):
    """Transpose a (pmax, free_dim) tile

    Args:
        tile (tensor): _description_
    """
    if nisa.get_nc_version() == nisa.nc_version.gen3:
        transpose_dtype = tile.dtype
    else:
        transpose_dtype = np.float32
    pmax = nl.tile_size.pmax
    assert (
        len(tile.shape) == 2 and tile.shape[0] == pmax
    ), f"Only supports transposing (pmax, free_dim) tiles. Received {tile.shape}."
    _, free_size = tile.shape
    index = nl.mgrid[0:pmax, 0:pmax]
    for free_id in nl.affine_range(free_size // pmax):
        free_ofs = free_id * pmax
        tileT = nl.ndarray((nl.par_dim(pmax), pmax), dtype=transpose_dtype, buffer=nl.psum)
        tileT[...] = nisa.nc_transpose(tile[index.p, free_ofs + index.x])
        tile[index.p, free_ofs + index.x] = nl.copy(tileT, dtype=tile.dtype)
