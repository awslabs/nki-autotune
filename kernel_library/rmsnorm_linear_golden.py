# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np


def rmsnorm_golden(x: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
    """Golden reference for RMSNorm + matmul.

    Args:
        x: Input tensor.
        y: Weight matrix.
        eps: Epsilon for numerical stability.

    Returns:
        Expected RMSNorm-matmul result as float32.
    """
    return rmsnorm_matmul_golden(x, y, eps).astype(np.float32)


def rmsnorm_matmul_golden(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """Compute RMSNorm followed by matmul.

    Normalizes x by its root-mean-square, then multiplies by weight.
    Uses sum-of-squares divided by feature size as a workaround for
    the lack of mean reduction on partition dimension.

    Args:
        x: Input tensor.
        weight: Weight matrix.
        eps: Epsilon for numerical stability.

    Returns:
        Result of RMSNorm(x) @ weight.
    """
    z = np.square(x)

    z = np.divide(z, x.shape[-1])
    z = np.sum(z, axis=-1, keepdims=True)

    z = z + eps
    z = x / np.sqrt(z)

    matmul_result = np.matmul(z, weight)

    return matmul_result


def fused_rmsnorm_gemm_golden(lhs: np.ndarray, rhs: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """Fused RMSNorm + GEMM algorithm with blocked processing in all dimensions.

    Args:
        lhs: Input tensor of shape (batch, m, k).
        rhs: Weight matrix of shape (k, n).
        epsilon: Small constant for numerical stability.

    Returns:
        Output tensor of shape (batch, m, n).
    """
    batch, m, k = lhs.shape
    k_r, n = rhs.shape

    assert k == k_r, f"Matrix dimensions mismatch: lhs has {k} columns but rhs has {k_r} rows"

    TILES_IN_BLOCK_M = 1024
    TILES_IN_BLOCK_N = 1024
    TILES_IN_BLOCK_K = 1024

    NUM_BLOCK_M = math.ceil(m / TILES_IN_BLOCK_M)
    NUM_BLOCK_N = math.ceil(n / TILES_IN_BLOCK_N)
    NUM_BLOCK_K = math.ceil(k / TILES_IN_BLOCK_K)

    O = np.zeros((batch, m, n))

    for batch_id in range(batch):
        for block_id_M in range(NUM_BLOCK_M):
            block_start_M = block_id_M * TILES_IN_BLOCK_M
            block_end_M = min((block_id_M + 1) * TILES_IN_BLOCK_M, m)
            block_size_M = block_end_M - block_start_M

            for block_id_N in range(NUM_BLOCK_N):
                block_start_N = block_id_N * TILES_IN_BLOCK_N
                block_end_N = min((block_id_N + 1) * TILES_IN_BLOCK_N, n)
                block_size_N = block_end_N - block_start_N

                square_sums = np.zeros(block_size_M)
                prev_square_sums = np.zeros(block_size_M)
                result_block = np.zeros((block_size_M, block_size_N))

                for block_id_K in range(NUM_BLOCK_K):
                    block_start_K = block_id_K * TILES_IN_BLOCK_K
                    block_end_K = min((block_id_K + 1) * TILES_IN_BLOCK_K, k)

                    lhs_block = lhs[batch_id, block_start_M:block_end_M, block_start_K:block_end_K]
                    rhs_block = rhs[block_start_K:block_end_K, block_start_N:block_end_N]

                    prev_square_sums[:] = square_sums

                    square_sums_block = np.sum(lhs_block**2, axis=1)
                    square_sums += square_sums_block

                    norm_factor_prevs = np.sqrt(prev_square_sums / k + epsilon)
                    norm_factors = np.sqrt(square_sums / k + epsilon)

                    if block_id_K > 0:
                        rescale = (norm_factor_prevs / norm_factors)[:, np.newaxis]
                        result_block *= rescale

                    norm_block = lhs_block / norm_factors[:, np.newaxis]
                    block_contribution = np.matmul(norm_block, rhs_block)
                    result_block += block_contribution

                O[batch_id, block_start_M:block_end_M, block_start_N:block_end_N] = result_block

    return O


def fused_rmsnorm_gemm_mkn(lhs: np.ndarray, rhs: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """Fused RMSNorm + GEMM algorithm with MKN loop ordering.

    Args:
        lhs: Input tensor of shape (batch, m, k).
        rhs: Weight matrix of shape (k, n).
        epsilon: Small constant for numerical stability.

    Returns:
        Output tensor of shape (batch, m, n).
    """
    batch, m, k = lhs.shape
    k_r, n = rhs.shape

    assert k == k_r, f"Matrix dimensions mismatch: lhs has {k} columns but rhs has {k_r} rows"

    BLOCK_M = 1024
    BLOCK_N = 1024
    BLOCK_K = 1024

    NUM_BLOCK_M = math.ceil(m / BLOCK_M)
    NUM_BLOCK_N = math.ceil(n / BLOCK_N)
    NUM_BLOCK_K = math.ceil(k / BLOCK_K)

    O = np.zeros((batch, m, n))

    for batch_id in range(batch):
        for block_id_M in range(NUM_BLOCK_M):
            block_start_M = block_id_M * BLOCK_M
            block_end_M = min((block_id_M + 1) * BLOCK_M, m)
            block_size_M = block_end_M - block_start_M

            square_sums = np.zeros(block_size_M)
            prev_square_sums = np.zeros(block_size_M)

            result_blocks = np.zeros((NUM_BLOCK_N, block_size_M, BLOCK_N))

            for block_id_K in range(NUM_BLOCK_K):
                block_start_K = block_id_K * BLOCK_K
                block_end_K = min((block_id_K + 1) * BLOCK_K, k)

                lhs_block = lhs[batch_id, block_start_M:block_end_M, block_start_K:block_end_K]

                prev_square_sums[:] = square_sums

                square_sums_block = np.sum(lhs_block**2, axis=1)
                square_sums += square_sums_block

                norm_factor_prevs = np.sqrt(prev_square_sums / k + epsilon)
                norm_factors = np.sqrt(square_sums / k + epsilon)

                if block_id_K > 0:
                    rescale = (norm_factor_prevs / norm_factors)[:, np.newaxis]
                    for block_id_N in range(NUM_BLOCK_N):
                        block_end_N = min((block_id_N + 1) * BLOCK_N, n)
                        block_size_N = block_end_N - block_id_N * BLOCK_N
                        result_blocks[block_id_N, :, :block_size_N] *= rescale

                norm_block = lhs_block / norm_factors[:, np.newaxis]

                for block_id_N in range(NUM_BLOCK_N):
                    block_start_N = block_id_N * BLOCK_N
                    block_end_N = min((block_id_N + 1) * BLOCK_N, n)
                    block_size_N = block_end_N - block_start_N

                    rhs_block = rhs[block_start_K:block_end_K, block_start_N:block_end_N]

                    block_contribution = np.matmul(norm_block, rhs_block)

                    result_blocks[block_id_N, :, :block_size_N] += block_contribution

            for block_id_N in range(NUM_BLOCK_N):
                block_start_N = block_id_N * BLOCK_N
                block_end_N = min((block_id_N + 1) * BLOCK_N, n)
                block_size_N = block_end_N - block_start_N

                O[batch_id, block_start_M:block_end_M, block_start_N:block_end_N] = result_blocks[
                    block_id_N, :, :block_size_N
                ]

    return O
