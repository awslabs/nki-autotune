import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
from neuronxcc.nki.typing import tensor


def transpose_tiles_in_block(block):
    """
    Transpose the (pmax, pmax) tiles in a block in place
    all PE array ops must output to FP32 on trn1 but must match input dtype in trn2
    """
    if nisa.get_nc_version() == nisa.nc_version.gen3:
        blockT_dtype = block.dtype
    else:
        blockT_dtype = np.float32
    tiles_in_block, par_size, free_size = block.shape
    pmax = nl.tile_size.pmax
    index = nl.mgrid[0:pmax, 0:pmax]

    for tile_id in nl.affine_range(tiles_in_block):
        for par_id in nl.affine_range(par_size // pmax):
            par_ofs = par_id * pmax
            for free_id in nl.affine_range(free_size // pmax):
                free_ofs = free_id * pmax
                tileT = nl.ndarray((nl.par_dim(pmax), pmax), dtype=blockT_dtype, buffer=nl.psum)
                tileT[index.p, index.x] = nisa.nc_transpose(block[tile_id, par_ofs + index.p, free_ofs + index.x])
                block[tile_id, par_ofs + index.p, free_ofs + index.x] = nl.copy(tileT, dtype=block.dtype)


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
