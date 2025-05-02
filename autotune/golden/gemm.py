import numpy as np


def gemm_core(lhs, rhs, lhs_is_transposed: bool):
    assert (
        len(lhs.shape) == 2 and len(rhs.shape) == 2
    ), f"gemm_core only computes 2D @ 2D GEMM. Received {lhs.shape} @ {rhs.shape}"
    if lhs_is_transposed:
        lhs = lhs.T
    M, K = lhs.shape
    _K, N = rhs.shape
    assert K == _K, f"lhs and rhs shape mismatch: {lhs.shape}, {rhs.shape}"
    result = np.matmul(lhs, rhs)
    return result


def gemm_cpu_golden(lhs, rhs, lhs_is_transposed: bool):
    if len(lhs.shape) == 2:
        output = gemm_core(lhs, rhs, lhs_is_transposed)
    elif len(lhs.shape) == 3:
        if lhs_is_transposed:
            batch_size, K, M = lhs.shape
        else:
            batch_size, M, K = lhs.shape
        _K, N = rhs.shape
        output = np.zeros((batch_size, M, N))
        for batch_id in range(batch_size):
            output[batch_id] = gemm_core(lhs[batch_id], rhs, lhs_is_transposed)
        if batch_size == 1:
            output = output[0]
    else:
        raise ValueError(f"lhs shape {lhs.shape} is not supported")
    return output


def matmul_xt_op(x_t, y):
    """Matrix multiplication with transposed first operand"""
    x = np.transpose(x_t, (1, 0))
    return np.matmul(x, y)


def matmul_op(x, y):
    """Matrix multiplication with non-transposed first operand"""
    return np.matmul(x, y)
