import os
import tempfile

# Set tempdir to a subdirectory of /tmp so klir_artifacts_directory starts with "/tmp/"
tempfile.tempdir = "/tmp/nki_artifacts"
os.makedirs(tempfile.tempdir, exist_ok=True)

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
from neuronxcc.nki_standalone import NKI_IR_VERSION, compile_nki_ir_kernel_to_neff


@nki.jit
def nki_matmul_block_free_dimension_(lhsT, rhs):
    """NKI kernel to compute a matrix multiplication operation while blocking the
       free dimensions of the LHS and RHS to improve memory access pattern.

    Args:
        lhsT: an input tensor of shape [K,M], where both K and M are multiples for
          128.  It is the left-hand-side argument of the matrix multiplication,
          delivered transposed for optimal performance.
        rhs: an input tensor of shape [K,N], where K is a multiple of 128, and N
          is a multiple of 512.  It is the right-hand-side argument of the matrix
          multiplication.
    Returns:
        result: the resulting output tensor of shape [M,N]
    """

    # Verify that the lhsT and rhs have the same contraction dimension.
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, "lhsT and rhs must have the same contraction dimension"

    # Lookup the device matrix multiply dimensions.
    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_K = nl.tile_size.pmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    # Configuring the blocking size for the free dimensions
    TILES_IN_BLOCK_M = 2
    TILES_IN_BLOCK_N = 2

    BLOCK_M = TILE_M * TILES_IN_BLOCK_M  # 256
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N  # 1024

    # the size has to be multiple of block size
    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0

    # Create a space for the result in HBM (not initialized)
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    # Loop over blocks over the M dimension
    for m in nl.affine_range(M // BLOCK_M):
        # Load TILES_IN_BLOCK_M columns tiles by TILES_K rows from lhsT
        lhsT_tiles = []
        for bm in nl.affine_range(TILES_IN_BLOCK_M):
            # Inner tile array.
            lhsT_tiles_internal = []
            for k in nl.affine_range(K // TILE_K):
                # Allocate space in SBUF for the tile (uninitialized)
                lhsT_tile = nl.ndarray(shape=(TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                # Copy the tile from HBM to SBUF
                nisa.dma_copy(
                    dst=lhsT_tile,
                    src=lhsT[
                        k * TILE_K : (k + 1) * TILE_K,
                        (m * TILES_IN_BLOCK_M + bm) * TILE_M : ((m * TILES_IN_BLOCK_M + bm) + 1) * TILE_M,
                    ],
                )
                # Append the tile to the inner list of tiles.
                lhsT_tiles_internal.append(lhsT_tile)
            # Append the inner list of tiles into the outer list of tiles.
            lhsT_tiles.append(lhsT_tiles_internal)

        for n in nl.affine_range(N // BLOCK_N):
            # Load TILES_IN_BLOCK_N columns from rhs by TILES_K rows from rhs
            rhs_tiles = []
            for bn in nl.affine_range(TILES_IN_BLOCK_N):
                # Inner tile array.
                rhs_tiles_internal = []
                for k in nl.affine_range(K // TILE_K):
                    # Allocate space in SBUF for the tile (uninitialized)
                    rhs_tile = nl.ndarray(shape=(TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
                    # Copy the tile from HBM to SBUF
                    nisa.dma_copy(
                        dst=rhs_tile,
                        src=rhs[
                            k * TILE_K : (k + 1) * TILE_K,
                            (n * TILES_IN_BLOCK_N + bn) * TILE_N : ((n * TILES_IN_BLOCK_N + bn) + 1) * TILE_N,
                        ],
                    )
                    # Append the tile to the inner list of tiles.
                    rhs_tiles_internal.append(rhs_tile)
                # Append the inner list of tiles into the outer list of tiles.
                rhs_tiles.append(rhs_tiles_internal)

            for bm in nl.affine_range(TILES_IN_BLOCK_M):
                for bn in nl.affine_range(TILES_IN_BLOCK_N):
                    # Allocate a tensor in PSUM
                    result_tile = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                    for k in nl.affine_range(K // TILE_K):
                        # Accumulate partial-sums into PSUM
                        nisa.nc_matmul(dst=result_tile, stationary=lhsT_tiles[bm][k], moving=rhs_tiles[bn][k])

                    # Copy the result from PSUM back to SBUF, and cast to expected
                    # output data-type
                    result_tmp = nl.ndarray(shape=result_tile.shape, dtype=result.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=result_tmp, src=result_tile)

                    # Copy the result from SBUF to HBM.
                    nisa.dma_copy(
                        dst=result[
                            (m * TILES_IN_BLOCK_M + bm) * TILE_M : ((m * TILES_IN_BLOCK_M + bm) + 1) * TILE_M,
                            (n * TILES_IN_BLOCK_N + bn) * TILE_N : ((n * TILES_IN_BLOCK_N + bn) + 1) * TILE_N,
                        ],
                        src=result_tmp,
                    )

    return result


# Define inputs as a dict with tensor name -> tensor-like object
# Objects just need shape and dtype attributes (duck typing)
M = 2048
N = 1024
K = 1024
inputs = {"lhsT": np.zeros((K, M), dtype=np.float32), "rhs": np.zeros((K, N), dtype=np.float32)}

# Define outputs - need objects with shape, dtype, and name attributes
from dataclasses import dataclass


@dataclass
class TensorStub:
    shape: tuple
    dtype: np.dtype
    name: str


outputs = [TensorStub(shape=(M, N), dtype=np.float32, name="C")]

compile_nki_ir_kernel_to_neff(
    kernel_func=nki_matmul_block_free_dimension_,
    kernel_inputs_dict=inputs,
    kernel_outputs=outputs,
    platform_target="trn2",
    logical_nc_config=2,
    output_directory="/tmp/my_kernel_output",
    version=NKI_IR_VERSION.beta2,
    additional_compiler_args="--auto-cast=none --internal-compiler-debug-mode=penguin",
)
