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


def fused_rmsnorm_gemm(lhs: np.ndarray, rhs: np.ndarray, epsilon: float = 1e-5):
    """
    Implements Fused RMSNorm + GEMM algorithm with chunked processing in both dimensions

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

    # Hard-coded chunk sizes
    m_chunk_size = 1024
    k_chunk_size = 1024

    # Initialize output tensor
    O = np.zeros((batch, m, n))

    # Process in batches and chunks of rows
    for b in range(batch):
        for i_start in range(0, m, m_chunk_size):
            i_end = min(i_start + m_chunk_size, m)

            # Initialize accumulators for this m chunk
            m_sums = np.zeros(i_end - i_start)
            o_primes = np.zeros((i_end - i_start, n))

            # Process k dimension in chunks
            for j_start in range(0, k, k_chunk_size):
                j_end = min(j_start + k_chunk_size, k)

                # Store previous sum of squares
                m_prevs = m_sums.copy()

                # Calculate sum of squares for this k chunk
                chunk = lhs[b, i_start:i_end, j_start:j_end]  # Shape: (chunk_m, chunk_k)
                m_sums_chunk = np.sum(chunk**2, axis=1)  # Sum along k dimension
                m_sums += m_sums_chunk

                # Compute normalization factors
                norm_factor_prevs = np.sqrt(m_prevs / k + epsilon)
                norm_factors = np.sqrt(m_sums / k + epsilon)

                # Adjust previous output based on new normalization factors
                if j_start > 0:
                    rescale = (norm_factor_prevs / norm_factors)[:, np.newaxis]
                    o_primes *= rescale

                # Calculate contribution from this chunk
                # (normalized inputs * weights) for all elements in the chunk at once
                norm_chunk = chunk / norm_factors[:, np.newaxis]
                chunk_contribution = np.matmul(norm_chunk, rhs[j_start:j_end])
                o_primes += chunk_contribution

            # Store results for this chunk
            O[b, i_start:i_end] = o_primes

    return O
