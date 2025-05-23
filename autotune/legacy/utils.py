from typing import Optional, Tuple

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np


class GEMMCompatibility:
    """
    Validates compatibility of input shapes and parameters for GEMM operations.

    This class checks whether the given matrix dimensions can be properly
    divided into blocks and tiles according to hardware constraints.
    """

    def __init__(self, transposed_lhs: bool) -> None:
        self.transposed_lhs = transposed_lhs

    def __call__(
        self,
        lhs_shape: Tuple[int, ...],
        rhs_shape: Tuple[int, ...],
        NUM_BLOCK_M: int = 1,
        NUM_BLOCK_N: int = 1,
        NUM_BLOCK_K: Optional[int] = None,  # Changed from -1 to None
        BUFFER_M: int = 1,
        BUFFER_N: int = 1,
        BUFFER_K: Optional[int] = None,  # Changed from -1 to None
        **kwargs,
    ) -> bool:
        """
        Initialize GEMM compatibility checker.

        Args:
            lhs_shape: Shape of left-hand side matrix, either (M,K) or (batch,M,K)
                       If transposed_lhs is True, dimensions are interpreted as (K,M) or (batch,K,M)
            rhs_shape: Shape of right-hand side matrix, expected to be (K,N)
            transposed_lhs: Whether the LHS matrix is transposed
            NUM_BLOCK_M: Number of blocks in M dimension
            NUM_BLOCK_N: Number of blocks in N dimension
            NUM_BLOCK_K: Number of blocks in K dimension, or None to skip K blocking
            BUFFER_M: Buffer degree for M dimension, must be <= NUM_BLOCK_M
            BUFFER_N: Buffer degree for N dimension, must be <= NUM_BLOCK_N
            BUFFER_K: Buffer degree for K dimension, or None to skip K buffering
        """
        # Input sizes
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

        # Buffer degrees (None means no buffering in that dimension)
        self.BUFFER_K = BUFFER_K
        self.BUFFER_M = BUFFER_M
        self.BUFFER_N = BUFFER_N

        # Calculate derived sizes
        self._calculate_sizes()

        # Validate contraction dimension matches
        if self.K != K_:
            raise ValueError(f"Contraction dimension mismatch: LHS has K={self.K}, RHS has K={K_}")

        # Validate dimensions
        self._check()
        return True

    def _calculate_sizes(self) -> None:
        """Calculate derived sizes for each dimension."""
        for dimension in ["M", "N", "K"]:
            size = getattr(self, f"{dimension}")
            num_block = getattr(self, f"NUM_BLOCK_{dimension}")

            # Handle dimensions with no blocking
            if num_block is None:
                num_block = 1

            tile_size = getattr(self, f"TILE_{dimension}")
            block_size = size // num_block
            tiles_in_block = block_size // tile_size
            tiles_in_dim = size // tile_size

            setattr(self, f"TILES_IN_BLOCK_{dimension}", tiles_in_block)
            setattr(self, f"TILES_IN_{dimension}", tiles_in_dim)
            setattr(self, f"BLOCK_{dimension}", block_size)

    def _check(self) -> None:
        """Validate that dimensions can be evenly divided into blocks and tiles."""
        for dimension in ["M", "N", "K"]:
            size = getattr(self, f"{dimension}")
            num_block = getattr(self, f"NUM_BLOCK_{dimension}")

            # Handle dimensions with no blocking
            if num_block is None:
                num_block = 1

            tiles_in_block = getattr(self, f"TILES_IN_BLOCK_{dimension}")
            tile_size = getattr(self, f"TILE_{dimension}")
            buffer_size = getattr(self, f"BUFFER_{dimension}")

            # Check even division
            if num_block * tiles_in_block * tile_size != size:
                raise ValueError(
                    f"{dimension} size {size} cannot be divided evenly into "
                    f"{num_block} blocks * {tiles_in_block} tiles * {tile_size}"
                )

            # Check buffer size if specified
            if buffer_size is not None:
                if buffer_size <= 0:
                    raise ValueError(f"{dimension} buffer degree must be positive, got {buffer_size}")
                if buffer_size > num_block:
                    raise ValueError(
                        f"{dimension} buffer degree {buffer_size} cannot be larger "
                        f"than number of blocks {num_block}"
                    )

    def __repr__(self) -> str:
        """String representation showing the division of dimensions into blocks and tiles."""
        # Determine which dimensions to include
        if self.NUM_BLOCK_K is None and self.BUFFER_K is None:
            dimensions = ["M", "N"]
        else:
            dimensions = ["M", "N", "K"]

        lines = [f"GEMM Compatibility Check:"]
        if self.batch is not None:
            lines.append(f"Batch size: {self.batch}")

        # Overall shape
        lines.append(f"Matrix dimensions: ({self.M}, {self.K}) × ({self.K}, {self.N})")

        # Add detailed information for each dimension
        for dimension in dimensions:
            size = getattr(self, f"{dimension}")
            num_block = getattr(self, f"NUM_BLOCK_{dimension}")
            if num_block is None:
                num_block = 1
            tiles_in_block = getattr(self, f"TILES_IN_BLOCK_{dimension}")
            tile_size = getattr(self, f"TILE_{dimension}")
            block_size = getattr(self, f"BLOCK_{dimension}")
            buffer_size = getattr(self, f"BUFFER_{dimension}")

            lines.append(
                f"{dimension}: {num_block} blocks × {tiles_in_block} tiles × " f"{tile_size} elements = {size} total"
            )
            lines.append(f"  Block size: {block_size}, Buffer degree: {buffer_size}")

        return "\n".join(lines)


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
                tileT_psum[...] = nisa.nc_transpose(lhs_block[tile_id_M, idx_lhs.p, tile_id_K * TILE_K + idx_lhs.x])
                tileT_sbuf = nl.ndarray((nl.par_dim(TILE_K), TILE_M), dtype=tileT_dtype, buffer=nl.sbuf)
                tileT_sbuf[...] = nl.copy(tileT_psum, dtype=lhs_block.dtype)
                result_tile += nisa.nc_matmul(
                    tileT_sbuf, rhs_block[tile_id_K, idx_rhs.p, tile_id_N * TILE_N + idx_rhs.x]
                )

            result_block[tile_id_M, tile_id_N, idx_res.p, idx_res.x] += result_tile[idx_res.p, idx_res.x]


def matmul_blocks_tile_transposed_lhs(tileT_lhs_block, rhs_block, result_block):
    """
    Accumulate matmul result tiles between lhs and rhs into result_block
    LHS is transposed at the tile level.
    Note that this is not the same as lhsT.
    'matmul_block' module computes lhsT @ rhs.

    Args:
    tileT_lhs_block: TILES_IN_M, TILE_M, K
    rhs_block: TILES_IN_K, TILE_K, N
    result_block : TILES_IN_M, TILES_IN_N, TILE_M, TILE_N
    """
    TILES_IN_M, TILE_M, K = tileT_lhs_block.shape
    TILES_IN_K, TILE_K, N = rhs_block.shape
    _TILES_IN_M, TILES_IN_N, _TILE_M, TILE_N = result_block.shape
    assert (
        TILES_IN_K * TILE_K == K
    ), f"K dimension mismatch: tileT_lhs_block {tileT_lhs_block.shape}. rhs_block {rhs_block.shape}."
    assert (
        TILES_IN_M == _TILES_IN_M and TILE_M == _TILE_M
    ), f"M dimension mismatch: tileT_lhs_block {tileT_lhs_block.shape}. result_block {result_block.shape}."
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
                    tileT_lhs_block[tile_id_M, idx_lhs.p, tile_id_K * TILE_K + idx_lhs.x],
                    rhs_block[tile_id_K, idx_rhs.p, tile_id_N * TILE_N + idx_rhs.x],
                )

            result_block[tile_id_M, tile_id_N, idx_res.p, idx_res.x] += result_tile[idx_res.p, idx_res.x]
