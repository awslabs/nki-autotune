import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from typing import Tuple, List
import numpy as np


class MatMulCompatibility:
    """
    Inputs compatibility checks for GEMM
    """

    def __init__(
        self,
        lhs_shape: Tuple,
        rhs_shape: Tuple,
        NUM_BLOCK_M: int = 1,
        NUM_BLOCK_N: int = 1,
        NUM_BLOCK_K: int = -1,
        BUFFER_M: int = 1,
        BUFFER_N: int = 1,
        BUFFER_K: int = -1,
    ) -> None:
        # Input sizes
        assert len(rhs_shape) == 2, f"Expecting (K, N) in RHS. Received {rhs_shape}."
        if len(lhs_shape) == 2:
            self.M, self.K = lhs_shape
        elif len(lhs_shape) == 3:
            batch, self.M, self.K = lhs_shape
        else:
            raise ValueError(f"lhs_shape must be either (M, K) or (batch, M, K). Received {lhs_shape}.")
        K_, self.N = rhs_shape

        # Single tile sizes
        self.TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
        self.TILE_N = nl.tile_size.gemm_moving_fmax  # 512
        self.TILE_K = nl.tile_size.pmax  # 128

        # Number of blocks
        self.NUM_BLOCK_M = NUM_BLOCK_M
        self.NUM_BLOCK_N = NUM_BLOCK_N
        self.NUM_BLOCK_K = NUM_BLOCK_K

        # Buffer degrees
        self.BUFFER_K = BUFFER_K
        self.BUFFER_M = BUFFER_M
        self.BUFFER_N = BUFFER_N

        # Calculate other sizes
        if self.NUM_BLOCK_K == -1 and self.BUFFER_K == -1:
            dimensions = ["M", "N"]
        else:
            dimensions = ["M", "N", "K"]
        self._calculate_sizes(dimensions)
        self._check(K_, dimensions)
        # self._show(dimensions)

    def _calculate_sizes(self, dimensions: List[str]):
        for dimension in dimensions:
            size = getattr(self, f"{dimension}")
            num_block = getattr(self, f"NUM_BLOCK_{dimension}")
            tile_size = getattr(self, f"TILE_{dimension}")
            tiles_in_block = size // num_block // tile_size
            tiles_in_dim = size // tile_size
            block_size = size // num_block

            setattr(self, f"TILES_IN_BLOCK_{dimension}", tiles_in_block)
            setattr(self, f"TILES_IN_{dimension}", tiles_in_dim)
            setattr(self, f"BLOCK_{dimension}", block_size)

    def _check(self, K_, dimensions: List[str]):
        assert self.K == K_, f"lhs and rhs contraction dimension mismatch, got {self.K} and {K_}"
        for dimension in dimensions:
            size = getattr(self, f"{dimension}")
            num_block = getattr(self, f"NUM_BLOCK_{dimension}")
            tiles_in_block = getattr(self, f"TILES_IN_BLOCK_{dimension}")
            tile_size = getattr(self, f"TILE_{dimension}")
            buffer_size = getattr(self, f"BUFFER_{dimension}")
            assert (
                num_block * tiles_in_block * tile_size == size
            ), f"{dimension} size {size} cannot be divided into {num_block} blocks * {tiles_in_block} tiles * {tile_size}"
            assert (
                buffer_size <= num_block
            ), f"{dimension} buffer degree {buffer_size} cannot be larger than number of blocks {num_block}"

    def _show(self, dimensions: List[str]):
        for dimension in dimensions:
            size = getattr(self, f"{dimension}")
            num_block = getattr(self, f"NUM_BLOCK_{dimension}")
            tiles_in_block = getattr(self, f"TILES_IN_BLOCK_{dimension}")
            tile_size = getattr(self, f"TILE_{dimension}")
            block_size = getattr(self, f"BLOCK_{dimension}")
            print(
                f"NUM_BLOCK_{dimension} {num_block} * TILES_IN_BLOCK_{dimension} {tiles_in_block} * TILE_{dimension} {tile_size} = {dimension} {size}"
            )
            print(f"BLOCK_{dimension} {block_size}")


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


def matmul_block(lhsT_block, rhs_block, result_block):
    """
    Accumulate matmul result tiles between lhsT and rhs into result_block

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
    Transpose the (pmax, pmax) tiles in a block in place
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


def matmul_blocks_tile_transposed_lhs(lhs_block, rhs_block, result_block):
    """
    Accumulate matmul result tiles between lhs and rhs into result_block
    LHS is transposed at the tile level.
    Note that this is not the same as lhsT.
    'matmul_block' module computes lhsT @ rhs.

    Args:
    lhs_block: TILES_IN_M, TILE_M, K
    rhs_block: TILES_IN_K, TILE_K, N
    result_block : TILES_IN_M, TILES_IN_N, TILE_M, TILE_N
    """
    TILES_IN_M, TILE_M, K = lhs_block.shape
    TILES_IN_K, TILE_K, N = rhs_block.shape
    _TILES_IN_M, TILES_IN_N, _TILE_M, TILE_N = result_block.shape
    assert TILES_IN_K * TILE_K == K, f"K dimension mismatch: lhs_block {lhs_block.shape}. rhs_block {rhs_block.shape}."
    assert (
        TILES_IN_M == _TILES_IN_M and TILE_M == _TILE_M
    ), f"M dimension mismatch: lhs_block {lhs_block.shape}. result_block {result_block.shape}."
    assert (
        N == TILES_IN_N * TILE_N
    ), f"N dimension mismatch: rhs_block {rhs_block.shape}. result_block {result_block.shape}."

    idx_lhs = nl.mgrid[0:TILE_M, 0:TILE_K]
    idx_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    for tile_id_M in nl.affine_range(TILES_IN_M):
        for tile_id_N in nl.affine_range(TILES_IN_N):
            result_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            """
            Use PSUM buffer to accumulate into a single hardware tile
            """
            for tile_id_K in nl.affine_range(TILES_IN_K):
                result_tile += nisa.nc_matmul(
                    lhs_block[tile_id_M, idx_lhs.p, tile_id_K * TILE_K + idx_lhs.x],
                    rhs_block[tile_id_K, idx_rhs.p, tile_id_N * TILE_N + idx_rhs.x],
                )

            result_block[tile_id_M, tile_id_N, idx_res.p, idx_res.x] += result_tile[idx_res.p, idx_res.x]
