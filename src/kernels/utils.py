import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from typing import Tuple
import numpy as np


class MatMulCompatibility:
    """
    Inputs compatibility checks for GEMM
    """

    def __init__(
        self,
        lhs_shape: Tuple,
        rhs_shape: Tuple,
        NUM_BLOCK_M: int,
        NUM_BLOCK_N: int,
        NUM_BLOCK_K: int,
        BUFFER_M: int,
        BUFFER_N: int,
        BUFFER_K: int,
    ) -> None:
        # Input sizes
        self.M, self.K = lhs_shape
        K_, self.N = rhs_shape

        # Single tile sizes
        self.TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
        self.TILE_N = nl.tile_size.gemm_moving_fmax  # 512
        self.TILE_K = nl.tile_size.pmax  # 128

        # Number of blocks
        self.NUM_BLOCK_M = NUM_BLOCK_M
        self.NUM_BLOCK_N = NUM_BLOCK_N
        self.NUM_BLOCK_K = NUM_BLOCK_K

        # Tiles in a block
        self.TILES_IN_BLOCK_M = self.M // self.NUM_BLOCK_M // self.TILE_M
        self.TILES_IN_BLOCK_N = self.N // self.NUM_BLOCK_N // self.TILE_N
        self.TILES_IN_BLOCK_K = self.K // self.NUM_BLOCK_K // self.TILE_K

        # Total number of tiles
        self.TILES_IN_M = self.NUM_BLOCK_M * self.TILES_IN_BLOCK_M
        self.TILES_IN_N = self.NUM_BLOCK_N * self.TILES_IN_BLOCK_N
        self.TILES_IN_K = self.NUM_BLOCK_K * self.TILES_IN_BLOCK_K

        # Block sizes
        self.BLOCK_M = self.TILE_M * self.TILES_IN_BLOCK_M
        self.BLOCK_N = self.TILE_N * self.TILES_IN_BLOCK_N
        self.BLOCK_K = self.TILE_K * self.TILES_IN_BLOCK_K

        # Buffer degrees
        self.BUFFER_K = BUFFER_K
        self.BUFFER_M = BUFFER_M
        self.BUFFER_N = BUFFER_N

        self._check(K_)
        self._show()

    def _check(self, K_):
        pmax, fmax = nl.tile_size.pmax, nl.tile_size.psum_fmax  # 128, 512
        assert K_ == self.K, "Reduction dimension must match"

        assert self.K == K_, f"lhs and rhs contraction dimension mismatch, got {self.K} and {K_}"
        assert (
            self.NUM_BLOCK_M * self.TILES_IN_BLOCK_M * self.TILE_M == self.M
        ), f"NUM_BLOCK_M {self.NUM_BLOCK_M} * TILES_IN_BLOCK_M {self.TILES_IN_BLOCK_M} * TILE_M {self.TILE_M} != M {self.M}"
        assert (
            self.NUM_BLOCK_N * self.TILES_IN_BLOCK_N * self.TILE_N == self.N
        ), f"NUM_BLOCK_N {self.NUM_BLOCK_N} * TILES_IN_BLOCK_N {self.TILES_IN_BLOCK_N} * TILE_N {self.TILE_N} != N {self.N}"
        assert (
            self.NUM_BLOCK_K * self.TILES_IN_BLOCK_K * self.TILE_K == self.K
        ), f"NUM_BLOCK_K {self.NUM_BLOCK_K} * TILES_IN_BLOCK_K {self.TILES_IN_BLOCK_K} * TILE_K {self.TILE_K} != K {self.K}"

        assert (
            self.BUFFER_M <= self.NUM_BLOCK_M
        ), f"M buffer degree {self.BUFFER_M} cannot be larger than number of M blocks {self.NUM_BLOCK_M}"
        assert (
            self.BUFFER_N <= self.NUM_BLOCK_N
        ), f"N buffer degree {self.BUFFER_N} cannot be larger than number of N blocks {self.NUM_BLOCK_N}"
        assert (
            self.BUFFER_K <= self.NUM_BLOCK_K
        ), f"K buffer degree {self.BUFFER_K} cannot be larger than number of K blocks {self.NUM_BLOCK_K}"

    def _show(self):
        print(
            f"NUM_BLOCK_M {self.NUM_BLOCK_M} * TILES_IN_BLOCK_M {self.TILES_IN_BLOCK_M} * TILE_M {self.TILE_M} = M {self.M}"
        )
        print(
            f"NUM_BLOCK_N {self.NUM_BLOCK_N} * TILES_IN_BLOCK_N {self.TILES_IN_BLOCK_N} * TILE_N {self.TILE_N} = N {self.N}"
        )
        print(
            f"NUM_BLOCK_K {self.NUM_BLOCK_K} * TILES_IN_BLOCK_K {self.TILES_IN_BLOCK_K} * TILE_K {self.TILE_K} = K {self.K}"
        )


def load_tensor_block(input_tensor, par_ofs: int, free_ofs: int, load_shape: Tuple[int, ...]):
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
        3D: (block_size, par_size, free_size)

    Returns:
        Loaded tiles in SBUF in the shape of load_shape
    """

    par_size, free_size = input_tensor.shape
    assert len(load_shape) == 3, f"'load_shape' expects (block, par, free). Received {load_shape}."
    load_block_size, load_par_size, load_free_size = load_shape
    idx = nl.mgrid[0:load_par_size, 0:load_free_size]
    loaded_tensor = nl.ndarray(
        (load_block_size, nl.par_dim(load_par_size), load_free_size), dtype=input_tensor.dtype, buffer=nl.sbuf
    )
    for block_id in nl.affine_range(load_block_size):
        par_indices = par_ofs + block_id * load_par_size + idx.p
        free_indices = free_ofs + idx.x
        loaded_tensor[block_id, idx.p, idx.x] = nl.load(
            input_tensor[par_indices, free_indices], mask=(par_indices < par_size) & (free_indices < free_size)
        )
    return loaded_tensor


def matmul_block(lhsT_block, rhs_block, result_block):
    """
    Accumulate matmul result tiles between lhsT and rhs into resulresult_blockt_tiles

    Args:
    lhsT_block: TILES_IN_BLOCK_K, TILE_K, BLOCK_M
    rhs_block: TILES_IN_BLOCK_K, TILE_K, BLOCK_N
    result_block : TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILE_M, TILE_N
    """
    TILES_IN_BLOCK_K, TILE_K, BLOCK_M = lhsT_block.shape
    _TILES_IN_BLOCK_K, _TILE_K, BLOCK_N = rhs_block.shape
    TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILE_M, TILE_N = result_block.shape

    # Data checks
    assert (
        TILES_IN_BLOCK_K == _TILES_IN_BLOCK_K and TILE_K == _TILE_K
    ), f"lhsT_block {lhsT_block.shape} does not match with rhs_block {rhs_block.shape}"
    assert (
        BLOCK_M == TILES_IN_BLOCK_M * TILE_M and BLOCK_N == TILES_IN_BLOCK_N * TILE_N
    ), f"lhsT_block {lhsT_block.shape} does not match with result_block {result_block.shape}"

    idx_lhsT = nl.mgrid[0:TILE_K, 0:TILE_M]
    idx_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    for tile_id_M in nl.affine_range(TILES_IN_BLOCK_M):
        for tile_id_N in nl.affine_range(TILES_IN_BLOCK_N):
            result_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            """
            Use PSUM buffer to accumulate into a single hardware tile
            to minimize the number of calls to nl.loop_reduce
            """
            for tile_id_K in nl.affine_range(TILES_IN_BLOCK_K):
                result_tile += nisa.nc_matmul(
                    lhsT_block[tile_id_K, idx_lhsT.p, tile_id_M * TILE_M + idx_lhsT.x],
                    rhs_block[tile_id_K, idx_rhs.p, tile_id_N * TILE_N + idx_rhs.x],
                )

            result_block[tile_id_M, tile_id_N, idx_res.p, idx_res.x] += result_tile[idx_res.p, idx_res.x]


def save_result_block(result, result_tiles, m_ofs, n_ofs):
    num_m_tiles, num_n_tiles, TILE_M, TILE_N = result_tiles.shape

    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    for tile_id_M in nl.affine_range(num_m_tiles):
        for tile_id_N in nl.affine_range(num_n_tiles):
            nl.store(
                result[m_ofs + tile_id_M * TILE_M + idx_res.p, n_ofs + tile_id_N * TILE_N + idx_res.x],
                value=result_tiles[tile_id_M, tile_id_N],
            )


def transpose_block(block):
    """
    all PE array ops must output to FP32 on trn1 but must match input dtype in trn2
    """
    if nisa.get_nc_version() == nisa.nc_version.gen3:
        blockT_dtype = block.dtype
    else:
        blockT_dtype = np.float32
    block_size, par_size, free_size = block.shape
    pmax = nl.tile_size.pmax
    index = nl.mgrid[0:pmax, 0:pmax]

    for block_id in nl.affine_range(block_size):
        for par_id in nl.affine_range(par_size // pmax):
            par_ofs = par_id * pmax
            for free_id in nl.affine_range(free_size // pmax):
                free_ofs = free_id * pmax
                blockT = nl.ndarray((nl.par_dim(pmax), pmax), dtype=blockT_dtype, buffer=nl.psum)
                blockT[index.p, index.x] = nisa.nc_transpose(block[block_id, par_ofs + index.p, free_ofs + index.x])
                block[block_id, par_ofs + index.p, free_ofs + index.x] = nl.copy(blockT, dtype=block.dtype)
    return block


def load_non_transposed_lhs_block(input_tensor, par_ofs: int, free_ofs: int, load_shape: Tuple[int, ...]):
    """
    Load a rectangular 2D tile of shape (num_m_tiles * TILE_M, K) from the input tensor.
    The location of the 2D tile from the input is offset by (free_ofs, par_ofs) at its upper left corner.

    Args:
        input_tensor: the input tensor to load from, arranged in (M, K)
        free_ofs: offset in the M dimension
        par_ofs: offset in the K dimension
        load_shape: 3D: (TILES_IN_BLOCK_M, TILE_M, K_size)

    Returns:
        Loaded tiles in SBUF in the shape of load_shape
    """
    m_size, k_size = input_tensor.shape
    assert len(load_shape) == 3, f"'load_shape' expects (tiles_m, tile_m, k_size). Received {load_shape}."
    tiles_in_block_m, tile_m, load_k_size = load_shape

    idx = nl.mgrid[0:tile_m, 0:load_k_size]
    loaded_tensor = nl.ndarray(
        (tiles_in_block_m, nl.par_dim(tile_m), load_k_size), dtype=input_tensor.dtype, buffer=nl.sbuf
    )

    for tile_id_m in nl.affine_range(tiles_in_block_m):
        m_indices = free_ofs + tile_id_m * tile_m + idx.p
        k_indices = par_ofs + idx.x
        loaded_tensor[tile_id_m, idx.p, idx.x] = nl.load(
            input_tensor[m_indices, k_indices], mask=(m_indices < m_size) & (k_indices < k_size)
        )

    return loaded_tensor


def matmul_non_transposed_blocks(lhs_block, rhs_block, result_tiles):
    """
    Accumulate matmul result tiles between non-transposed lhs and rhs into result_tiles

    Args:
        lhs_block: BLOCK_M, TILES_IN_BLOCK_K, par_dim(TILE_K)
        rhs_block: TILES_IN_BLOCK_K, par_dim(TILE_K), BLOCK_N
        result_tiles: TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILE_M, TILE_N
    """
    BLOCK_M, TILES_IN_BLOCK_K, TILE_K = lhs_block.shape
    _TILES_IN_BLOCK_K, _TILE_K, BLOCK_N = rhs_block.shape
    TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILE_M, TILE_N = result_tiles.shape

    # Data checks
    assert (
        TILES_IN_BLOCK_K == _TILES_IN_BLOCK_K and TILE_K == _TILE_K
    ), f"lhs_block {lhs_block.shape} shape mismatch with rhs_block {rhs_block.shape}"
    assert (
        TILES_IN_BLOCK_M * TILE_M == BLOCK_M
    ), f"lhs_block {lhs_block.shape} shape mismatch with result_tiles {result_tiles.shape}"
    assert (
        TILES_IN_BLOCK_N * TILE_N == BLOCK_N
    ), f"rhs_block {rhs_block.shape} shape mismatch with result_tiles {result_tiles.shape}"

    idx_lhsT = nl.mgrid[0:TILE_K, 0:TILE_M]
    idx_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]

    for tile_id_M in nl.affine_range(TILES_IN_BLOCK_M):
        for tile_id_N in nl.affine_range(TILES_IN_BLOCK_N):
            res_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            for tile_id_K in nl.affine_range(TILES_IN_BLOCK_K):
                # Need to transpose LHS on the fly for matrix multiplication
                lhs_tile = lhs_block[tile_id_M * TILE_M + idx_lhsT.x, tile_id_K, idx_lhsT.p]
                print(f"lhs_tile = {lhs_tile.shape}")

                """
                TILES_IN_BLOCK_K, TILE_K, BLOCK_M = lhsT_block.shape
                res_tile += nisa.nc_matmul(
                    lhsT_block[tile_id_K, idx_lhsT.p, tile_id_M * TILE_M + idx_lhsT.x],
                    rhs_block[tile_id_K, idx_rhs.p, tile_id_N * TILE_N + idx_rhs.x],
                )
                """

                # Perform the matrix multiplication
                res_tile += nisa.nc_matmul(
                    lhs_tile[idx_lhsT.p, idx_lhsT.x], rhs_block[tile_id_K, idx_rhs.p, tile_id_N * TILE_N + idx_rhs.x]
                )

            # Store the result
            result_tiles[tile_id_M, tile_id_N, idx_res.p, idx_res.x] += res_tile[idx_res.p, idx_res.x]
