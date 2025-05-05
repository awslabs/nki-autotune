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
    Implements Fused RMSNorm + GEMM algorithm with blocked processing in both dimensions

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
    m_block_size = 1024
    k_block_size = 1024

    # Calculate number of blocks in each dimension
    NUM_BLOCK_M = math.ceil(m / m_block_size)
    NUM_BLOCK_K = math.ceil(k / k_block_size)

    # Initialize output tensor
    O = np.zeros((batch, m, n))

    # Process in batches and blocks of rows
    for batch_id in range(batch):
        for block_id_M in range(NUM_BLOCK_M):
            # Calculate start and end indices for M dimension
            block_start_M = block_id_M * m_block_size
            block_end_M = min((block_id_M + 1) * m_block_size, m)
            block_rows = block_end_M - block_start_M

            # Initialize accumulators for this M block
            m_sums = np.zeros(block_rows)
            m_prevs = np.zeros(block_rows)  # Initialize m_prevs here
            o_primes = np.zeros((block_rows, n))

            # Process k dimension in blocks
            for block_id_K in range(NUM_BLOCK_K):
                # Calculate start and end indices for K dimension
                block_start_K = block_id_K * k_block_size
                block_end_K = min((block_id_K + 1) * k_block_size, k)

                # Update previous sum of squares in-place
                m_prevs[:] = m_sums

                # Calculate sum of squares for this K block
                block = lhs[batch_id, block_start_M:block_end_M, block_start_K:block_end_K]
                m_sums_block = np.sum(block**2, axis=1)  # Sum along k dimension
                m_sums += m_sums_block

                # Compute normalization factors
                norm_factor_prevs = np.sqrt(m_prevs / k + epsilon)
                norm_factors = np.sqrt(m_sums / k + epsilon)

                # Adjust previous output based on new normalization factors
                if block_id_K > 0:
                    rescale = (norm_factor_prevs / norm_factors)[:, np.newaxis]
                    o_primes *= rescale

                # Calculate contribution from this block
                # (normalized inputs * weights) for all elements in the block at once
                norm_block = block / norm_factors[:, np.newaxis]
                block_contribution = np.matmul(norm_block, rhs[block_start_K:block_end_K])
                o_primes += block_contribution

            # Store results for this block
            O[batch_id, block_start_M:block_end_M] = o_primes

    return O
