from typing import Dict, Tuple

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
from neuronxcc.nki.typing import tensor

from autotune.modules.matmul import GEMMCompatibility


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
    ), f"Expect (row_tile_size, row_num_tiles, column_num_tiles, column_tile_size). Received {block.shape}."
    row_tile_size, row_num_tiles, column_num_tiles, column_tile_size = block.shape
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
                        block[row_ofs + index.p, row_tile_id, column_tile_id, column_ofs + index.x]
                    )
                    block[row_ofs + index.p, row_tile_id, column_tile_id, column_ofs + index.x] = nl.copy(
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


def get_block_shape(
    mm: GEMMCompatibility, loop_order: Dict[str, int], dims: Tuple[str, str], curr_position: int
) -> Tuple[int, int, int, int]:
    """
    Calculate the shape of tensor blocks in a GEMM operation.

    This function computes the shape of blocks based on the current position in nested loops
    and the specified dimensions. It determines how many blocks should be processed together
    for each dimension based on loop nesting.

    Args:
        mm (GEMMCompatibility): Object containing GEMM configuration parameters and block dimensions
        loop_order (Dict[str,int]): Str representing the order of loops (e.g., "MNK")
        dims (Tuple[str, str]): Tuple of dimension names to calculate shape for (e.g., ("M", "N"))
        curr_position (int): Current position in the nested loops (0, 1, 2, or 3)

    Returns:
        Tuple[int, int, int, int]: A 4-tuple representing the block shape:
            - First element: Tile size for the first dimension
            - Second element: Number of blocks * tiles in block for the first dimension
            - Third element: Number of blocks * tiles in block for the second dimension
            - Fourth element: Tile size for the second dimension

    Note:
        For dimensions with loop position less than curr_position, num_block is set to 1.
        For other dimensions, num_block is set to the corresponding NUM_BLOCK_{dim} value.
    """
    num_blocks = []
    for dim in dims:
        dim_position = loop_order[dim]
        if dim_position <= curr_position:
            num_block = 1
        else:
            num_block = getattr(mm, f"NUM_BLOCK_{dim}")
        num_blocks.append(num_block)
    block_shape = (
        getattr(mm, f"TILE_{dims[0]}"),
        num_blocks[0] * getattr(mm, f"TILES_IN_BLOCK_{dims[0]}"),
        num_blocks[1] * getattr(mm, f"TILES_IN_BLOCK_{dims[1]}"),
        getattr(mm, f"TILE_{dims[1]}"),
    )
    # print(
    #     f"get_block_shape: dependent dims {dims}. curr loop position {curr_position}. loop_order {loop_order}.\n--> block_shape{block_shape}."
    # )
    return block_shape


def get_block_ofs(
    mm: GEMMCompatibility, loop_order: Dict[str, int], dims: Tuple[str, str], curr_position: int, curr_block_ids
) -> Tuple[int, int]:
    """
    This function computes the starting offsets for blocks along specified dimensions based on
    the current position in nested loops and the corresponding block indices.

    Args:
        mm (GEMMCompatibility): Object containing GEMM configuration parameters and block dimensions
        loop_order (Dict[str,int]): Dict representing the order of loops
        dims (Tuple[str, str]): Tuple of dimension names to calculate offsets for (e.g., ("K", "M"))
        curr_position (int): Current integer position in the nested loops
        curr_block_ids (List[int]): List of current block indices from the outer loops

    Returns:
        Tuple[int, int]: Tuple containing the offsets for the specified dimensions

    Note:
        For dimensions with loop position <= curr_position, offset is calculated as
        block_id * block_size. For other dimensions, offset is block_id * 0 = 0.
    """
    block_ofs = []
    for dim in dims:
        dim_position = loop_order[dim]
        if dim_position <= curr_position:
            block_size = getattr(mm, f"BLOCK_{dim}")
            ofs = curr_block_ids[dim_position] * block_size
        else:
            ofs = 0
        block_ofs.append(ofs)
    block_ofs = tuple(block_ofs)
    print(
        f"get_block_ofs: dependent dims {dims}. curr loop position {curr_position}. loop_order {loop_order}.\n-->{block_ofs}."
    )
    return block_ofs
