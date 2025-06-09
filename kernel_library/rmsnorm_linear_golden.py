import math

import neuronxcc.nki.language as nl
import numpy as np
import torch

from autotune.typing import INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE, OUTPUT_TENSOR_DTYPE


def rmsnorm_correctness_postprocessing(
    input_tensors: INPUT_TENSORS_DTYPE, kernel_kwargs: KERNEL_KWARGS_DTYPE, kernel_output: OUTPUT_TENSOR_DTYPE
) -> None:
    kernel_output = nl.static_cast(kernel_output, np.float32)

    x, y = input_tensors
    golden = rmsnorm_matmul(x, y, kernel_kwargs["eps"])
    np.testing.assert_allclose(actual=kernel_output, desired=golden, atol=1e-3, rtol=1e-3, err_msg="", verbose=True)


def rmsnorm_matmul(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Computes RMSNorm + GEMM on CPU using input precision and returns results as float32.

    Parameters:
    -----------
    x : np.ndarray
        Input array of shape (batch_size, seq_len, hidden_dim) in float32 or bfloat16
    weight : np.ndarray
        Weight matrix of shape (hidden_dim, out_dim) in float32 or bfloat16
    eps : float
        Epsilon value for numerical stability

    Returns:
    --------
    np.ndarray
        Result of RMSNorm(x) @ weight converted to float32 NumPy array
    """
    # Determine if inputs are bfloat16
    is_bfloat16 = str(x.dtype) == "bfloat16" or str(weight.dtype) == "bfloat16"

    # Convert NumPy arrays to PyTorch tensors (on CPU)
    if is_bfloat16:
        # Convert bfloat16 to float32 then to tensor then back to bfloat16
        x_float32 = x.astype(np.float32)
        weight_float32 = weight.astype(np.float32)

        # Fix the to() syntax by using correct keyword format
        x_tensor = torch.from_numpy(x_float32).to(dtype=torch.bfloat16)
        weight_tensor = torch.from_numpy(weight_float32).to(dtype=torch.bfloat16)
    else:
        # Already float32, direct conversion
        x_tensor = torch.from_numpy(x.copy())
        weight_tensor = torch.from_numpy(weight.copy())

    # Square the input
    z = torch.square(x_tensor)

    # Compute mean across the last dimension
    z = torch.div(z, x_tensor.shape[-1])
    z_mean = torch.sum(z, dim=-1, keepdim=True)

    # Add epsilon for numerical stability
    z_mean = z_mean + eps

    # Normalize x
    x_normalized = x_tensor / torch.sqrt(z_mean)

    # Matrix multiplication with weights
    result = torch.matmul(x_normalized, weight_tensor)

    # Convert result to NumPy float32
    result_np = result.float().numpy()

    return result_np


def rmsnorm_matmul_golden(x, weight, eps: float):
    """
    z: Array["B, L or 1, 1"] = (x**2).mean(-1, keepdims=True) + self.eps
    z: Array["B, L or 1, D"] = x / np.sqrt(z)
    ret = z * self.weight
    """
    z = np.square(x)

    # FIXME:
    # if this `z` tensor is on PSUM, it might trigger
    # In `codegenPartitionReduceOp` we have `assert inst.op.op != np.mean, 'There is not reduce mean!'`
    # z = np.mean(z, axis=-1, keepdims=True)

    # FIXME: this is another workaround because there is no mean reduction on partition dim
    z = np.divide(z, x.shape[-1])
    z = np.sum(z, axis=-1, keepdims=True)

    z = z + eps
    z = x / np.sqrt(z)

    matmul_result = np.matmul(z, weight)

    return matmul_result


def fused_rmsnorm_gemm_golden(lhs: np.ndarray, rhs: np.ndarray, epsilon: float = 1e-5):
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


def fused_rmsnorm_gemm_mkn(lhs: np.ndarray, rhs: np.ndarray, epsilon: float = 1e-5):
    """
    Implements Fused RMSNorm + GEMM algorithm with MKN loop ordering

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
    BLOCK_M = 1024
    BLOCK_N = 1024
    BLOCK_K = 1024

    # Calculate number of blocks in each dimension
    NUM_BLOCK_M = math.ceil(m / BLOCK_M)
    NUM_BLOCK_N = math.ceil(n / BLOCK_N)
    NUM_BLOCK_K = math.ceil(k / BLOCK_K)

    # Initialize output tensor
    O = np.zeros((batch, m, n))

    # Process in batches
    for batch_id in range(batch):
        # Process blocks of rows (M dimension)
        for block_id_M in range(NUM_BLOCK_M):
            block_start_M = block_id_M * BLOCK_M
            block_end_M = min((block_id_M + 1) * BLOCK_M, m)
            block_size_M = block_end_M - block_start_M

            # Initialize square sums for this M block
            square_sums = np.zeros(block_size_M)
            prev_square_sums = np.zeros(block_size_M)

            # Initialize result blocks for all N blocks for this M block
            result_blocks = np.zeros((NUM_BLOCK_N, block_size_M, BLOCK_N))

            # Process K dimension blocks
            for block_id_K in range(NUM_BLOCK_K):
                block_start_K = block_id_K * BLOCK_K
                block_end_K = min((block_id_K + 1) * BLOCK_K, k)

                # Get lhs block for this K
                lhs_block = lhs[batch_id, block_start_M:block_end_M, block_start_K:block_end_K]

                # Update previous sum of squares
                prev_square_sums[:] = square_sums

                # Calculate sum of squares for this K block
                square_sums_block = np.sum(lhs_block**2, axis=1)
                square_sums += square_sums_block

                # Compute normalization factors
                norm_factor_prevs = np.sqrt(prev_square_sums / k + epsilon)
                norm_factors = np.sqrt(square_sums / k + epsilon)

                # Rescale previous results if this is not the first K block
                if block_id_K > 0:
                    rescale = (norm_factor_prevs / norm_factors)[:, np.newaxis]
                    for block_id_N in range(NUM_BLOCK_N):
                        block_end_N = min((block_id_N + 1) * BLOCK_N, n)
                        block_size_N = block_end_N - block_id_N * BLOCK_N
                        result_blocks[block_id_N, :, :block_size_N] *= rescale

                # Normalize the lhs block
                norm_block = lhs_block / norm_factors[:, np.newaxis]

                # Process N dimension blocks (innermost loop)
                for block_id_N in range(NUM_BLOCK_N):
                    block_start_N = block_id_N * BLOCK_N
                    block_end_N = min((block_id_N + 1) * BLOCK_N, n)
                    block_size_N = block_end_N - block_start_N

                    # Get rhs block
                    rhs_block = rhs[block_start_K:block_end_K, block_start_N:block_end_N]

                    # Matrix multiply the normalized block with weights
                    block_contribution = np.matmul(norm_block, rhs_block)

                    # Accumulate into result
                    result_blocks[block_id_N, :, :block_size_N] += block_contribution

            # After processing all K blocks, write results to output tensor
            for block_id_N in range(NUM_BLOCK_N):
                block_start_N = block_id_N * BLOCK_N
                block_end_N = min((block_id_N + 1) * BLOCK_N, n)
                block_size_N = block_end_N - block_start_N

                O[batch_id, block_start_M:block_end_M, block_start_N:block_end_N] = result_blocks[
                    block_id_N, :, :block_size_N
                ]

    return O
