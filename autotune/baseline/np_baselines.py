import numpy as np


def matmul_xt_op(x_t, y):
    """Matrix multiplication with transposed first operand"""
    x = np.transpose(x_t, (1, 0))
    return np.matmul(x, y)


def matmul_op(x, y):
    """Matrix multiplication with non-transposed first operand"""
    return np.matmul(x, y)


def rmsnorm_linear_op(x, y, eps: float = 1e-6):
    """
    Applies RMSNorm to x and then performs matrix multiplication with y.

    Args:
        x: Input tensor to normalize
        y: Weight matrix for the linear operation

    Returns:
        Result of normalized x multiplied by y
    """
    # Apply RMSNorm
    # Calculate root mean square along the last dimension
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)

    # Normalize x by dividing by its RMS
    x_normalized = x / rms

    # Perform matrix multiplication (GEMM) of normalized x with y
    result = np.matmul(x_normalized, y)

    return result
