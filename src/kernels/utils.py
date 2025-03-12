import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from typing import Tuple


def load_tensor(input_tensor, par_ofs: int, free_ofs: int, load_shape: Tuple[int, ...]):
    """
    Load a rectangular 2D tile of shape (num_par_tiles * par_tile_size, free_dim_size) from the input tensor.
    The location of the 2D tile from the input is offset by (par_ofs, free_ofs) at its upper left corner.
    Load the input tile by tile in parallel in the par dimension.

    Input tensor (par_size, free_size):
    +------------------+
    |                  |
    |    +--------+    |  ← Starting at (par_ofs, free_ofs)
    |    |Tile 0  |    |
    |    |Tile 1  |    |  Each tile is (par_tile_size * free_dim_size)
    |    |  ...   |    |
    |    |Tile N-1|    |  N = num_par_tiles
    |    +--------+    |
    |                  |
    +------------------+

    Args:
        input_tensor: the input tensor to load from, arranged in (par_size, free_size)
        par_ofs: offset in the partition dimension
        free_ofs: offset in the free dimension
        load_shape:
        2D: (nl.par_dim(par_size), free_size)
        3D: (block_size, par_size, free_size)

    Returns:
        Loaded tiles in SBUF in the shape of load_shape
    """

    par_size, free_size = input_tensor.shape
    if len(load_shape) == 2:
        load_par_size, load_free_size = load_shape
        load_block_size = None
    elif len(load_shape) == 3:
        load_block_size, load_par_size, load_free_size = load_shape
    else:
        raise ValueError(f"'load_shape' expects (par, free) or (block, par, free). Received {load_shape}.")
    idx = nl.mgrid[0:load_par_size, 0:load_free_size]
    if load_block_size:
        loaded_tensor = nl.ndarray(
            (load_block_size, nl.par_dim(load_par_size), load_free_size), dtype=input_tensor.dtype, buffer=nl.sbuf
        )
        for block_id in nl.affine_range(load_block_size):
            par_indices = par_ofs + block_id * load_par_size + idx.p
            free_indices = free_ofs + idx.x
            loaded_tensor[block_id, idx.p, idx.x] = nl.load(
                input_tensor[par_indices, free_indices], mask=(par_indices < par_size) & (free_indices < free_size)
            )
    else:
        loaded_tensor = nl.ndarray(
            (nl.par_dim(load_par_size), load_free_size), dtype=input_tensor.dtype, buffer=nl.sbuf
        )
        par_indices = par_ofs + idx.p
        free_indices = free_ofs + idx.x
        loaded_tensor[idx.p, idx.x] = nl.load(
            input_tensor[par_indices, free_indices], mask=(par_indices < par_size) & (free_indices < free_size)
        )
    return loaded_tensor


def matmul_tiles(lhsT_tiles, rhs_tiles, result_tiles):
    """
    Accumulate matmul result tiles between lhsT and rhs into result_tiles

    Args:
    lhsT_tiles: num_k_tiles, TILE_K, m
    rhs_tiles: _num_k_tiles, _TILE_K, n
    result_tiles : num_m_tiles, num_n_tiles, TILE_M, TILE_N
    """
    num_k_tiles, TILE_K, m = lhsT_tiles.shape
    _num_k_tiles, _TILE_K, n = rhs_tiles.shape
    num_m_tiles, num_n_tiles, TILE_M, TILE_N = result_tiles.shape
    _n = num_n_tiles * TILE_N

    # Data checks
    assert (
        num_k_tiles == _num_k_tiles and TILE_K == _TILE_K
    ), f"lhsT_tiles {lhsT_tiles.shape} does not match with rhs_tiles {rhs_tiles.shape}"
    assert (
        m == num_m_tiles * TILE_M
    ), f"lhsT_tiles {lhsT_tiles.shape} does not match with result_tiles {result_tiles.shape}"
    assert n == _n, f"rhs_tiles {rhs_tiles.shape} does not match with result_tiles {result_tiles.shape}"

    idx_lhsT = nl.mgrid[0:TILE_K, 0:TILE_M]
    idx_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    for tile_id_M in nl.affine_range(num_m_tiles):
        for tile_id_N in nl.affine_range(num_n_tiles):
            res_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

            # Use PSUM buffer to accumulate into a single hardware tile
            # to minimize the number of calls to nl.loop_reduce
            for tile_id_K in nl.affine_range(num_k_tiles):
                res_tile += nisa.nc_matmul(
                    lhsT_tiles[tile_id_K, idx_lhsT.p, tile_id_M * TILE_M + idx_lhsT.x],
                    rhs_tiles[tile_id_K, idx_rhs.p, tile_id_N * TILE_N + idx_rhs.x],
                )

            result_tiles[tile_id_M, tile_id_N, idx_res.p, idx_res.x] += res_tile[idx_res.p, idx_res.x]
