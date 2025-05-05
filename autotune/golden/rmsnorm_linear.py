import numpy as np


def golden_fun(hidden, gate, gamma, qkv_weights, eps: float):
    def _silu(x):
        return x / (1 + np.exp(-x))

    if gate is not None:
        hidden = hidden * _silu(gate.astype(np.float32))
    rms = np.sqrt(np.mean(np.square(hidden), axis=-1, keepdims=True) + eps)
    output = hidden * np.reciprocal(rms)
    if gamma is not None:
        output *= gamma
    if qkv_weights is not None:
        output = output @ qkv_weights
    return output


def rmsnorm_gemm(x, y, eps: float = 1e-6):
    """
    Applies RMSNorm to x and then performs matrix multiplication with y.

    Args:
        x: Input tensor to normalize
        y: Weight matrix for the linear operation

    Returns:
        Result of normalized x multiplied by y
    """
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    x_normalized = x / rms
    result = np.matmul(x_normalized, y)
    return result


import math

import numpy as np


def fused_rmsnorm_gemm(lhs: np.ndarray, rhs: np.ndarray, epsilon: float = 1e-5):
    """
    Implements Fused RMSNorm + GEMM algorithm with blocked processing in all dimensions

    Parameters:
    -----------
    lhs : 3D numpy array
        Input tensor of shape (batch, m, k)
    rhs : 2D numpy array
        Weight matrix of shape (k, n)
    epsilon : float
        Small constant for numerical stability

    Returns:
    --------
    O : 3D numpy array
        Output tensor of shape (batch, m, n)
    """
    batch, m, k = lhs.shape
    k_r, n = rhs.shape

    assert k == k_r, f"Matrix dimensions mismatch: lhs has {k} columns but rhs has {k_r} rows"

    # Hard-coded block sizes
    TILES_IN_BLOCK_M = 1024
    TILES_IN_BLOCK_N = 1024
    TILES_IN_BLOCK_K = 1024

    # Calculate number of blocks in each dimension
    NUM_BLOCK_M = math.ceil(m / TILES_IN_BLOCK_M)
    NUM_BLOCK_N = math.ceil(n / TILES_IN_BLOCK_N)
    NUM_BLOCK_K = math.ceil(k / TILES_IN_BLOCK_K)

    # Initialize output tensor
    O = np.zeros((batch, m, n))

    # Process in batches and blocks of rows
    for batch_id in range(batch):
        for block_id_M in range(NUM_BLOCK_M):
            block_start_M = block_id_M * TILES_IN_BLOCK_M
            block_end_M = min((block_id_M + 1) * TILES_IN_BLOCK_M, m)
            block_size_M = block_end_M - block_start_M

            for block_id_N in range(NUM_BLOCK_N):
                block_start_N = block_id_N * TILES_IN_BLOCK_N
                block_end_N = min((block_id_N + 1) * TILES_IN_BLOCK_N, n)
                block_size_N = block_end_N - block_start_N

                # Initialize accumulators for this M-N block
                # FIXME: for the same block M, the square sums are the same.
                square_sums = np.zeros(block_size_M)
                prev_square_sums = np.zeros(block_size_M)
                result_block = np.zeros((block_size_M, block_size_N))

                for block_id_K in range(NUM_BLOCK_K):
                    block_start_K = block_id_K * TILES_IN_BLOCK_K
                    block_end_K = min((block_id_K + 1) * TILES_IN_BLOCK_K, k)

                    # Get current blocks
                    lhs_block = lhs[batch_id, block_start_M:block_end_M, block_start_K:block_end_K]
                    rhs_block = rhs[block_start_K:block_end_K, block_start_N:block_end_N]

                    # Update previous sum of squares
                    prev_square_sums[:] = square_sums

                    # Calculate sum of squares for this K block
                    square_sums_block = np.sum(lhs_block**2, axis=1)
                    square_sums += square_sums_block

                    # Compute normalization factors
                    norm_factor_prevs = np.sqrt(prev_square_sums / k + epsilon)
                    norm_factors = np.sqrt(square_sums / k + epsilon)

                    # Adjust previous output based on new normalization factors
                    if block_id_K > 0:
                        rescale = (norm_factor_prevs / norm_factors)[:, np.newaxis]
                        result_block *= rescale

                    # Calculate contribution from this block
                    # (normalized inputs * weights) for all elements in the block at once
                    norm_block = lhs_block / norm_factors[:, np.newaxis]
                    block_contribution = np.matmul(norm_block, rhs_block)
                    result_block += block_contribution

                # Store results for this block
                O[batch_id, block_start_M:block_end_M, block_start_N:block_end_N] = result_block

    return O
