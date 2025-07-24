from typing import Tuple

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np

from autotune.typing import INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE, OUTPUT_TENSORS_DTYPE


class GEMMCompatibility:
    """
    Validates compatibility of input shapes and parameters for GEMM operations.

    This class checks whether the given matrix dimensions can be properly
    divided into blocks and tiles according to hardware constraints.
    """

    def __init__(self, transposed_lhs: bool) -> None:
        """
        Initialize GEMM compatibility checker.

        Args:
            transposed_lhs: Whether the LHS matrix should be interpreted as transposed.
        """
        self.transposed_lhs = transposed_lhs

    def __call__(self, input_tensors: INPUT_TENSORS_DTYPE, kernel_kwargs: KERNEL_KWARGS_DTYPE):
        """
        Check compatibility of input tensors for GEMM operation with specified parameters.

        Args:
            input_tensors: Tuple containing (lhs, rhs) matrices for GEMM operation.
                lhs_shape: Shape of left-hand side matrix, either (M,K) or (batch,M,K)
                           If transposed_lhs is True, dimensions are interpreted as (K,M) or (batch,K,M)
                rhs_shape: Shape of right-hand side matrix, expected to be (K,N)
            kernel_kwargs: Dictionary containing kernel parameters:
                NUM_BLOCK_M: Number of blocks in M dimension
                NUM_BLOCK_N: Number of blocks in N dimension
                NUM_BLOCK_K: Number of blocks in K dimension, or None to skip K blocking

        Raises:
            ValueError: If matrix shapes are incompatible or cannot be properly divided
                       into the specified number of blocks and tiles.
        """
        lhs, rhs = input_tensors
        lhs_shape = lhs.shape
        rhs_shape = rhs.shape
        NUM_BLOCK_M: int = kernel_kwargs.get("NUM_BLOCK_M", 1)
        NUM_BLOCK_N: int = kernel_kwargs.get("NUM_BLOCK_N", 1)
        NUM_BLOCK_K: int = kernel_kwargs.get("NUM_BLOCK_K", 1)
        if len(rhs_shape) != 2:
            raise ValueError(f"Expecting (K, N) in RHS. Received {rhs_shape}.")

        self.batch = None
        if len(lhs_shape) == 2:
            if self.transposed_lhs:
                self.K, self.M = lhs_shape
            else:
                self.M, self.K = lhs_shape
        elif len(lhs_shape) == 3:
            if self.transposed_lhs:
                self.batch, self.K, self.M = lhs_shape
            else:
                self.batch, self.M, self.K = lhs_shape
        else:
            raise ValueError(f"lhs_shape must be either 2D or (batch, 2D). Received {lhs_shape}.")

        K_, self.N = rhs_shape

        # Validate dimensions > 0
        for dim_name, dim_value in [("M", self.M), ("K", self.K), ("N", self.N)]:
            if dim_value <= 0:
                raise ValueError(f"Dimension {dim_name} must be positive, got {dim_value}")

        # Single tile sizes (hardware constants)
        self.TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
        self.TILE_N = nl.tile_size.gemm_moving_fmax  # 512
        self.TILE_K = nl.tile_size.pmax  # 128

        # Number of blocks (None means no blocking in that dimension)
        self.NUM_BLOCK_M = NUM_BLOCK_M
        self.NUM_BLOCK_N = NUM_BLOCK_N
        self.NUM_BLOCK_K = NUM_BLOCK_K

        # Calculate derived sizes
        self._calculate_sizes()

        # Validate contraction dimension matches
        if self.K != K_:
            raise ValueError(f"Contraction dimension mismatch: LHS has K={self.K}, RHS has K={K_}")

        # Validate dimensions
        self._check()

    def _calculate_sizes(self) -> None:
        """
        Calculate derived sizes for each dimension (M, N, K).

        For each dimension, calculates:
        - Block size (dimension divided by number of blocks)
        - Number of tiles in each block (block size divided by tile size)
        - Total number of tiles in the dimension (dimension size divided by tile size)

        Results are stored as attributes on the class instance.
        """
        # Calculate sizes for M dimension
        num_block_m = 1 if self.NUM_BLOCK_M is None else self.NUM_BLOCK_M
        self.BLOCK_M: int = self.M // num_block_m
        self.TILES_IN_BLOCK_M = self.BLOCK_M // self.TILE_M
        self.TILES_IN_M = self.M // self.TILE_M

        # Calculate sizes for N dimension
        num_block_n = 1 if self.NUM_BLOCK_N is None else self.NUM_BLOCK_N
        self.BLOCK_N: int = self.N // num_block_n
        self.TILES_IN_BLOCK_N = self.BLOCK_N // self.TILE_N
        self.TILES_IN_N = self.N // self.TILE_N

        # Calculate sizes for K dimension
        num_block_k = 1 if self.NUM_BLOCK_K is None else self.NUM_BLOCK_K
        self.BLOCK_K: int = self.K // num_block_k
        self.TILES_IN_BLOCK_K = self.BLOCK_K // self.TILE_K
        self.TILES_IN_K = self.K // self.TILE_K

    def _check(self) -> None:
        """
        Validate that dimensions can be evenly divided into blocks and tiles.

        Verifies that for each dimension (M, N, K):
        1. The dimension size can be evenly divided into the specified number of blocks
        2. Each block can be evenly divided into tiles

        Raises:
            ValueError: If any dimension cannot be evenly divided as required.
        """
        for dimension in ["M", "N", "K"]:
            size = getattr(self, f"{dimension}")
            num_block = getattr(self, f"NUM_BLOCK_{dimension}")

            # Handle dimensions with no blocking
            if num_block is None:
                num_block = 1

            tiles_in_block = getattr(self, f"TILES_IN_BLOCK_{dimension}")
            tile_size = getattr(self, f"TILE_{dimension}")

            # Check even division
            if num_block * tiles_in_block * tile_size != size:
                raise ValueError(
                    f"{dimension} size {size} cannot be divided evenly into "
                    f"{num_block} blocks * {tiles_in_block} tiles * {tile_size}"
                )


def matmul_blocks_lhsT(lhsT_block, rhs_block, result_block, ofs: Tuple[int, int]):
    """
    Perform tiled matrix multiplication between transposed left-hand side and right-hand side matrices.

    This function computes the matrix multiplication lhsT @ rhs, where lhsT is already in transposed format.
    Results are accumulated into the result_block at the specified offset position. The multiplication
    is performed by iterating over tiles in each dimension and accumulating results along the K dimension.
    This implementation optimizes performance by operating on blocked matrices for hardware efficiency.

    Args:
        lhsT_block: Left-hand side matrix block in transposed format.
                   Shape: (TILE_K, TILES_IN_K, TILES_IN_M, TILE_M)
                   Where TILE_K/M are tile sizes and TILES_IN_K/M are counts of tiles in those dimensions.
        rhs_block: Right-hand side matrix block.
                  Shape: (TILE_K, TILES_IN_K, TILES_IN_N, TILE_N)
                  Where TILE_K/N are tile sizes and TILES_IN_K/N are counts of tiles in those dimensions.
        result_block: Output matrix block where results are accumulated.
                     Shape: (TILE_M, >=TILES_IN_M, >=TILES_IN_N, TILE_N)
                     Must have sufficient space to store all tiles in M and N dimensions.
        ofs: Tuple of (M_offset, N_offset) specifying where in the result_block to start accumulating.
             These offsets are in #elements (not tiles or blocks).

    Notes:
        - This function accumulates results, so the result_block is both input and output
        - Intermediate calculations use hardware-specific buffer allocation (nl.psum)
        - The K dimension is fully accumulated over during the computation
    """
    # print(f"lhsT_block {lhsT_block.shape} @ rhs_block {rhs_block.shape} = result_block {result_block.shape}.")
    TILE_K, TILES_IN_K, TILES_IN_M, TILE_M = lhsT_block.shape
    _, _, TILES_IN_N, TILE_N = rhs_block.shape
    assert rhs_block.shape == (
        TILE_K,
        TILES_IN_K,
        TILES_IN_N,
        TILE_N,
    ), f"lhsT_block {lhsT_block.shape} shape mismatch with rhs_block {rhs_block.shape}"
    assert (
        result_block.shape[0] == TILE_M
        and result_block.shape[1] >= TILES_IN_M
        and result_block.shape[2] >= TILES_IN_N
        and result_block.shape[3] == TILE_N
    ), f"result_block {result_block.shape} shape mismatch with lhsT_block {lhsT_block.shape} @ rhs_block {rhs_block.shape}"

    idx_lhsT = nl.mgrid[0:TILE_K, 0:TILE_M]
    idx_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    M_ofs, N_ofs = ofs
    for tile_id_M in nl.affine_range(TILES_IN_M):
        for tile_id_N in nl.affine_range(TILES_IN_N):
            result_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            """
            Use PSUM buffer to accumulate into a single hardware tile
            """
            for tile_id_K in nl.affine_range(TILES_IN_K):
                result_tile += nisa.nc_matmul(
                    lhsT_block[idx_lhsT.p, tile_id_K, tile_id_M, idx_lhsT.x],
                    rhs_block[idx_rhs.p, tile_id_K, tile_id_N, idx_rhs.x],
                )
            result_block[idx_res.p, M_ofs // TILE_M + tile_id_M, N_ofs // TILE_N + tile_id_N, idx_res.x] += result_tile[
                idx_res.p, idx_res.x
            ]


def matmul_blocks_lhs(lhs_block, rhs_block, result_block):
    """
    Accumulate matmul result tiles between lhs and rhs into result_block
    LHS is not transposed.
    Note that this is not the same as lhsT.
    'matmul_block' module computes lhsT @ rhs.

    Args:
    lhs_block: TILES_IN_M, TILE_M, K
    rhs_block: TILES_IN_K, TILE_K, N
    result_block : TILES_IN_M, TILES_IN_N, TILE_M, TILE_N
    """
    TILE_M, TILES_IN_M, TILES_IN_K, TILE_K = lhs_block.shape
    _TILE_K, _TILES_IN_K, TILES_IN_N, TILE_N = rhs_block.shape
    _TILE_M, _TILES_IN_M, _TILES_IN_N, _TILE_N = result_block.shape
    assert (
        TILE_K == _TILE_K and TILES_IN_K == _TILES_IN_K
    ), f"lhs_block and rhs_block shape mismatch: lhs_block {lhs_block.shape}. rhs_block {rhs_block.shape}."
    assert (
        TILE_M == _TILE_M and TILES_IN_M == _TILES_IN_M
    ), f"lhs_block and result_block shape mismatch: lhs_block {lhs_block.shape}. result_block {result_block.shape}."
    assert (
        TILE_N == _TILE_N and TILES_IN_N == _TILES_IN_N
    ), f"rhs_block and result_block shape mismatch: rhs_block {rhs_block.shape}. result_block {result_block.shape}."

    if nisa.get_nc_version() == nisa.nc_version.gen3:
        tileT_dtype = lhs_block.dtype
    else:
        tileT_dtype = np.float32

    idx_lhs = nl.mgrid[0:TILE_M, 0:TILE_K]
    idx_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    # TODO: do M-K-N loop order, transpose M, K lhs_block then use it in N
    for tile_id_M in nl.affine_range(TILES_IN_M):
        for tile_id_N in nl.affine_range(TILES_IN_N):
            result_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            """
            Use PSUM buffer to accumulate into a single hardware tile
            """
            for tile_id_K in nl.affine_range(TILES_IN_K):
                # FIXME: in-place transposition repeated across tile_id_N
                # TODO: use a temp tile to hold tileT
                tileT_psum = nl.ndarray((nl.par_dim(TILE_K), TILE_M), dtype=tileT_dtype, buffer=nl.psum)
                tileT_psum[...] = nisa.nc_transpose(lhs_block[idx_lhs.p, tile_id_M, tile_id_K, idx_lhs.x])
                tileT_sbuf = nl.ndarray((nl.par_dim(TILE_K), TILE_M), dtype=tileT_dtype, buffer=nl.sbuf)
                tileT_sbuf[...] = nl.copy(tileT_psum, dtype=lhs_block.dtype)
                result_tile += nisa.nc_matmul(tileT_sbuf, rhs_block[idx_rhs.p, tile_id_K, tile_id_N, idx_rhs.x])

            result_block[idx_res.p, tile_id_M, tile_id_N, idx_res.x] += result_tile[idx_res.p, idx_res.x]


def matmul_blocks_tile_transposed_lhs(tileT_lhs_block, rhs_block, result_blocks, block_id_N):
    """
    Accumulate matmul result tiles between lhs and rhs into result_block
    LHS is transposed at the tile level.
    Note that this is not the same as lhsT.
    'matmul_block' module computes lhsT @ rhs.

    Args:
    tileT_lhs_block: TILE_M, TILE_K, TILES_IN_M, TILES_IN_K
    rhs_block: TILE_K, TILE_N, TILES_IN_K, TILES_IN_N
    result_block : TILE_M, TILE_N, TILES_IN_M, TILES_IN_N
    """
    TILE_M, TILES_IN_M, TILES_IN_K, TILE_K = tileT_lhs_block.shape
    _TILE_K, _TILES_IN_K, TILES_IN_N, TILE_N = rhs_block.shape
    _TILE_M, NUM_BLOCK_N, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, _TILE_N = result_blocks.shape
    assert (
        TILE_K == _TILE_K and TILES_IN_K == _TILES_IN_K
    ), f"K dimension mismatch: tileT_lhs_block {tileT_lhs_block.shape}. rhs_block {rhs_block.shape}."

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
                    tileT_lhs_block[idx_lhs.p, tile_id_M, tile_id_K, idx_lhs.x],
                    rhs_block[idx_rhs.p, tile_id_K, tile_id_N, idx_rhs.x],
                )
            result_blocks[idx_res.p, block_id_N, tile_id_M, tile_id_N, idx_res.x] += result_tile[idx_res.p, idx_res.x]


def matmul_blocks_tile_transposed_lhs(tileT_lhs_block, rhs_block, result_blocks, block_id_N):
    """
    Accumulate matmul result tiles between lhs and rhs into result_block
    LHS is transposed at the tile level.
    Note that this is not the same as lhsT.
    'matmul_block' module computes lhsT @ rhs.

    Args:
    tileT_lhs_block: TILE_M, TILE_K, TILES_IN_M, TILES_IN_K
    rhs_block: TILE_K, TILE_N, TILES_IN_K, TILES_IN_N
    result_block : TILE_M, TILE_N, TILES_IN_M, TILES_IN_N
    """
    TILE_M, TILES_IN_M, TILES_IN_K, TILE_K = tileT_lhs_block.shape
    _TILE_K, _TILES_IN_K, TILES_IN_N, TILE_N = rhs_block.shape
    _TILE_M, NUM_BLOCK_N, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, _TILE_N = result_blocks.shape
    assert (
        TILE_K == _TILE_K and TILES_IN_K == _TILES_IN_K
    ), f"K dimension mismatch: tileT_lhs_block {tileT_lhs_block.shape}. rhs_block {rhs_block.shape}."

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
                    tileT_lhs_block[idx_lhs.p, tile_id_M, tile_id_K, idx_lhs.x],
                    rhs_block[idx_rhs.p, tile_id_K, tile_id_N, idx_rhs.x],
                )
            result_blocks[idx_res.p, block_id_N, tile_id_M, tile_id_N, idx_res.x] += result_tile[idx_res.p, idx_res.x]


class GEMMCorrectness:
    def __init__(self, transposed_lhs: bool) -> None:
        self.transposed_lhs = transposed_lhs

    def __call__(
        self,
        input_tensors: INPUT_TENSORS_DTYPE,
        kernel_kwargs: KERNEL_KWARGS_DTYPE,
        nki_out_tensors: OUTPUT_TENSORS_DTYPE,
    ):
        data_type = np.float32
        atol, rtol = 1e-2, 1e-2
        lhs, rhs = input_tensors
        if self.transposed_lhs:
            golden = nl.static_cast(lhsT_rhs_gemm_np(lhs, rhs), data_type)
        else:
            golden = nl.static_cast(lhs_rhs_gemm_np(lhs, rhs), data_type)
        nki_out_tensor = nl.static_cast(nki_out_tensors[0], data_type)
        np.testing.assert_allclose(
            actual=nki_out_tensor, desired=golden, atol=atol, rtol=rtol, err_msg="", verbose=True
        )


def lhs_rhs_gemm_np(lhs, rhs):
    """
    Calculate the general matrix multiplication (GEMM) between lhs and rhs.

    Parameters:
    -----------
    lhs : numpy.ndarray
        Left-hand side matrix or tensor. Can have an extra batch dimension.
    rhs : numpy.ndarray
        Right-hand side matrix.

    Returns:
    --------
    numpy.ndarray
        Result of the matrix multiplication.
    """
    return np.matmul(lhs, rhs)


def lhsT_rhs_gemm_np(lhsT, rhs):
    """
    Calculate the general matrix multiplication (GEMM) between lhsT and rhs.

    Parameters:
    -----------
    lhs : numpy.ndarray
        Left-hand side matrix or tensor. Can have an extra batch dimension.
    rhs : numpy.ndarray
        Right-hand side matrix.

    Returns:
    --------
    numpy.ndarray
        Result of the matrix multiplication.
    """
    if len(lhsT.shape) == 3:  # Batch dimension exists
        lhs = np.transpose(lhsT, (0, 2, 1))
    else:
        lhs = lhsT.T
    return np.matmul(lhs, rhs)


def blocked_gemm_np_mkn(lhs, rhs, NUM_BLOCK_M: int, NUM_BLOCK_N: int, NUM_BLOCK_K: int):
    """
    GEMM algorithm with vectorized block processing using MKN loop ordering.
    """
    batch, M, K = lhs.shape
    K_check, N = rhs.shape

    if K != K_check:
        raise ValueError(f"Incompatible dimensions: lhs K dimension is {K} and rhs K dimension is {K_check}")

    # Calculate block sizes in each dimension
    BLOCK_M = M // NUM_BLOCK_M
    BLOCK_N = N // NUM_BLOCK_N
    BLOCK_K = K // NUM_BLOCK_K

    # Initialize output matrix with the input data type
    output = np.zeros((batch, M, N), dtype=lhs.dtype)

    # Process each batch
    for b in range(batch):
        # Process blocks of M dimension
        for block_m in range(NUM_BLOCK_M):
            m_start = block_m * BLOCK_M
            m_end = min(M, (block_m + 1) * BLOCK_M)
            m_size = m_end - m_start

            # Initialize accumulator for the entire M block with higher precision (float64)
            outputs = np.zeros((m_size, N), dtype=np.float64)

            # Process K dimension in blocks
            for block_k in range(NUM_BLOCK_K):
                k_start = block_k * BLOCK_K
                k_end = min(K, (block_k + 1) * BLOCK_K)

                # Extract current K block for all M rows in this block
                lhs_block = lhs[b, m_start:m_end, k_start:k_end]

                # Process each N block
                for block_n in range(NUM_BLOCK_N):
                    n_start = block_n * BLOCK_N
                    n_end = min(N, (block_n + 1) * BLOCK_N)

                    # Extract current N block for this K block
                    rhs_block = rhs[k_start:k_end, n_start:n_end]

                    # Calculate contribution and add to the higher precision accumulator
                    contribution = np.matmul(lhs_block, rhs_block)
                    outputs[:, n_start:n_end] += contribution

            # Store final results for this M block, casting back to the original data type
            output[b, m_start:m_end, :] = outputs.astype(lhs.dtype)

    return output
