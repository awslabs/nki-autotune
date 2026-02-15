"""Golden data for autotune backend tests."""

import subprocess

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np


def _neuron_devices_available() -> bool:
    """Check if Neuron devices are available by running neuron-ls."""
    try:
        result = subprocess.run(["neuron-ls"], capture_output=True, timeout=10)
        return result.returncode == 0
    except FileNotFoundError:
        return False


NEURON_DEVICES_AVAILABLE = _neuron_devices_available()

SHAPES = [(128, 128), (128, 256), (128, 512)]
SCALAR_VALUES = [0.0, 1.5, -2.0]
ATOL, RTOL = 1e-5, 1e-5
WARMUP, ITERS = 2, 5


@nki.jit
def nki_tensor_add_scalar_(a_input, b_input, c):
    """NKI kernel that computes a_input + b_input + c.

    Args:
        a_input: First input tensor of shape [P, F] where P <= 128.
        b_input: Second input tensor of shape [P, F] where P <= 128.
        c: Scalar value to add.

    Returns:
        result: Output tensor of shape [P, F].
    """
    P, F = a_input.shape
    result = nl.ndarray((P, F), dtype=a_input.dtype, buffer=nl.shared_hbm)

    a_tile = nl.ndarray((P, F), dtype=a_input.dtype, buffer=nl.sbuf)
    b_tile = nl.ndarray((P, F), dtype=b_input.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_tile, src=a_input)
    nisa.dma_copy(dst=b_tile, src=b_input)

    sum_tile = nl.ndarray((P, F), dtype=a_input.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sum_tile, data1=a_tile, data2=b_tile, op=nl.add)

    result_tile = nl.ndarray((P, F), dtype=a_input.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(result_tile, sum_tile, nl.add, c)

    nisa.dma_copy(dst=result, src=result_tile)
    return result


def golden_add_scalar(a_input: np.ndarray, b_input: np.ndarray, c: float) -> np.ndarray:
    """Golden reference for nki_tensor_add_scalar_.

    Args:
        a_input: First input array.
        b_input: Second input array.
        c: Scalar value to add.

    Returns:
        Result of a_input + b_input + c, cast to a_input's dtype.
    """
    return (a_input + b_input + c).astype(a_input.dtype)


@nki.jit
def nki_matmul_block_free_dimension_(lhsT, rhs):
    """NKI kernel to compute matrix multiplication with blocked free dimensions.

    Computes C = lhsT.T @ rhs where lhsT is the transposed left-hand-side matrix.
    Blocking the free dimensions improves memory access patterns.

    Args:
        lhsT: Input tensor of shape [K, M], where K and M are multiples of 128.
            Left-hand-side argument delivered transposed for optimal performance.
        rhs: Input tensor of shape [K, N], where K is a multiple of 128 and N
            is a multiple of 512.

    Returns:
        result: Output tensor of shape [M, N].
    """
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, "lhsT and rhs must have the same contraction dimension"

    TILE_M = nl.tile_size.gemm_stationary_fmax
    TILE_K = nl.tile_size.pmax
    TILE_N = nl.tile_size.gemm_moving_fmax

    TILES_IN_BLOCK_M = 2
    TILES_IN_BLOCK_N = 2

    BLOCK_M = TILE_M * TILES_IN_BLOCK_M
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N

    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // BLOCK_M):
        lhsT_tiles = []
        for bm in nl.affine_range(TILES_IN_BLOCK_M):
            lhsT_tiles_internal = []
            for k in nl.affine_range(K // TILE_K):
                lhsT_tile = nl.ndarray(shape=(TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=lhsT_tile,
                    src=lhsT[
                        k * TILE_K : (k + 1) * TILE_K,
                        (m * TILES_IN_BLOCK_M + bm) * TILE_M : ((m * TILES_IN_BLOCK_M + bm) + 1) * TILE_M,
                    ],
                )
                lhsT_tiles_internal.append(lhsT_tile)
            lhsT_tiles.append(lhsT_tiles_internal)

        for n in nl.affine_range(N // BLOCK_N):
            rhs_tiles = []
            for bn in nl.affine_range(TILES_IN_BLOCK_N):
                rhs_tiles_internal = []
                for k in nl.affine_range(K // TILE_K):
                    rhs_tile = nl.ndarray(shape=(TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=rhs_tile,
                        src=rhs[
                            k * TILE_K : (k + 1) * TILE_K,
                            (n * TILES_IN_BLOCK_N + bn) * TILE_N : ((n * TILES_IN_BLOCK_N + bn) + 1) * TILE_N,
                        ],
                    )
                    rhs_tiles_internal.append(rhs_tile)
                rhs_tiles.append(rhs_tiles_internal)

            for bm in nl.affine_range(TILES_IN_BLOCK_M):
                for bn in nl.affine_range(TILES_IN_BLOCK_N):
                    result_tile = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                    for k in nl.affine_range(K // TILE_K):
                        nisa.nc_matmul(dst=result_tile, stationary=lhsT_tiles[bm][k], moving=rhs_tiles[bn][k])

                    result_tmp = nl.ndarray(shape=result_tile.shape, dtype=result.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=result_tmp, src=result_tile)

                    nisa.dma_copy(
                        dst=result[
                            (m * TILES_IN_BLOCK_M + bm) * TILE_M : ((m * TILES_IN_BLOCK_M + bm) + 1) * TILE_M,
                            (n * TILES_IN_BLOCK_N + bn) * TILE_N : ((n * TILES_IN_BLOCK_N + bn) + 1) * TILE_N,
                        ],
                        src=result_tmp,
                    )

    return result


def matmul_transposed_lhs_golden(lhsT: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Compute golden reference for transposed-LHS matmul.

    Args:
        lhsT: Transposed left-hand side matrix of shape [K, M].
        rhs: Right-hand side matrix of shape [K, N].

    Returns:
        Expected result of lhsT.T @ rhs.
    """
    return lhsT.T @ rhs
