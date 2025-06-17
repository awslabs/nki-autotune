import math

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
from neuronxcc import nki
from neuronxcc.nki.typing import tensor

from autotune.core.dma import load_tensor_block, save_result_dma
from autotune.core.utils import GEMMCompatibility
from autotune.typing import INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE, OUTPUT_TENSOR_DTYPE


def softmax_gemm_correctness_postprocessing(
    input_tensors: INPUT_TENSORS_DTYPE, kernel_kwargs: KERNEL_KWARGS_DTYPE, kernel_output: OUTPUT_TENSOR_DTYPE
) -> None:
    kernel_output = nl.static_cast(kernel_output, np.float32)
    lhs, rhs = input_tensors
    golden = softmax_gemm_np(lhs, rhs)
    np.testing.assert_allclose(actual=kernel_output, desired=golden, atol=1e-3, rtol=1e-3, err_msg="", verbose=True)


def softmax_gemm_np(lhs, rhs):
    """
    Apply softmax to lhs first, then perform GEMM with rhs.

    Args:
        lhs: Left-hand side matrix of shape (batch, M, K)
        rhs: Right-hand side matrix of shape (K, N)

    Returns:
        Matrix of shape (batch, M, N) resulting from (softmax(lhs) @ rhs)
    """
    # Apply softmax to lhs along K dimension (axis=-1)
    lhs_max = np.max(lhs, axis=-1, keepdims=True)
    exp_lhs = np.exp(lhs - lhs_max)
    sum_exp = np.sum(exp_lhs, axis=-1, keepdims=True)
    softmax_lhs = exp_lhs / sum_exp

    # Perform batch matrix multiplication with rhs
    result = np.matmul(softmax_lhs, rhs)

    return result


def online_softmax_gemm_np(lhs, rhs):
    """
    Implements the Online Softmax + GEMM algorithm with vectorized block processing.

    Parameters:
        lhs (numpy.ndarray): Left-hand side matrix of shape (batch, M, K)
        rhs (numpy.ndarray): Right-hand side matrix of shape (K, N)

    Returns:
        numpy.ndarray: Result matrix of shape (batch, M, N)
    """
    batch, M, K = lhs.shape
    K_check, N = rhs.shape

    # Check dimensions
    if K != K_check:
        raise ValueError(f"Incompatible dimensions: lhs K dimension is {K} and rhs K dimension is {K_check}")

    # Hard-coded block sizes
    TILES_IN_BLOCK_M = 1024
    TILES_IN_BLOCK_N = 1024
    TILES_IN_BLOCK_K = 1024

    # Calculate number of blocks in each dimension
    NUM_BLOCK_M = math.ceil(M / TILES_IN_BLOCK_M)
    NUM_BLOCK_N = math.ceil(N / TILES_IN_BLOCK_N)
    NUM_BLOCK_K = math.ceil(K / TILES_IN_BLOCK_K)

    # Initialize output matrix
    output = np.zeros((batch, M, N))

    # Process each batch
    for b in range(batch):
        # Process blocks of M dimension
        for block_m in range(NUM_BLOCK_M):
            m_start = block_m * TILES_IN_BLOCK_M
            m_end = min(M, (block_m + 1) * TILES_IN_BLOCK_M)
            M_block_size = m_end - m_start

            # Process blocks of N dimension
            for block_n in range(NUM_BLOCK_N):
                n_start = block_n * TILES_IN_BLOCK_N
                n_end = min(N, (block_n + 1) * TILES_IN_BLOCK_N)
                N_block_size = n_end - n_start

                # Initialize state arrays for this M,N block
                a_vals = np.full((M_block_size, 1), float("-inf"))  # max values, shape (M_block_size, 1)
                b_vals = np.zeros((M_block_size, 1))  # normalization terms, shape (M_block_size, 1)
                o_vals = np.zeros((M_block_size, N_block_size))  # output values

                # Process K dimension in blocks
                for block_k in range(NUM_BLOCK_K):
                    k_start = block_k * TILES_IN_BLOCK_K
                    k_end = min(K, (block_k + 1) * TILES_IN_BLOCK_K)
                    K_block_size = k_end - k_start

                    # Extract the current K-block from inputs
                    lhs_block = lhs[b, m_start:m_end, k_start:k_end]  # Shape: (M_block_size, K_block_size)
                    rhs_block = rhs[k_start:k_end, n_start:n_end]  # Shape: (K_block_size, N_block_size)

                    # Store previous values
                    a_prev = a_vals.copy()
                    b_prev = b_vals.copy()

                    # Calculate row-wise max values for this K block
                    block_max_vals = np.max(lhs_block, axis=1, keepdims=True)  # Shape: (M_block_size, 1)

                    # Update global max for each row (m)
                    new_a_vals = np.maximum(a_vals, block_max_vals)

                    # Calculate scaling factor for the change in max values
                    scale_factor = np.exp(a_vals - new_a_vals)  # Shape: (M_block_size, 1)

                    # Update a_vals
                    a_vals = new_a_vals

                    # Calculate exp(x - a) for all elements in the block
                    exp_block = np.exp(lhs_block - a_vals)  # Shape: (M_block_size, K_block_size)

                    # Calculate sum of exp values for this block
                    exp_sum_block = np.sum(exp_block, axis=1, keepdims=True)  # Shape: (M_block_size, 1)

                    # Update b values (normalization term)
                    is_first_block = a_prev == float("-inf")  # Mask for first valid block

                    # For rows that have previous valid values
                    b_vals = np.where(
                        is_first_block,
                        exp_sum_block,  # First valid block
                        b_prev * scale_factor + exp_sum_block,  # Previous valid values exist
                    )

                    # Scale previous output by the change in max and normalization factor
                    valid_prev = ~is_first_block & (b_vals > 0)
                    if np.any(valid_prev):
                        # Only apply scaling where we have valid previous values
                        rescale = np.where(valid_prev, scale_factor * (b_prev / b_vals), np.ones_like(scale_factor))
                        o_vals *= rescale

                    # Calculate contribution from this block
                    # Normalize exp_block by its sum for softmax weighting
                    softmax_block = exp_block / b_vals  # Shape: (M_block_size, K_block_size)

                    # Calculate weighted contribution from this block
                    contribution = np.matmul(softmax_block, rhs_block)  # Shape: (M_block_size, N_block_size)

                    # Add contribution to output
                    o_vals = o_vals + contribution

                # Store results
                output[b, m_start:m_end, n_start:n_end] = o_vals

    return output


@nki.jit()
def online_softmax_linear_MKN(
    lhs: tensor, rhs: tensor, NUM_BLOCK_M: int, NUM_BLOCK_N: int, NUM_BLOCK_K: int, norm_dtype=nl.float32
):
    assert len(lhs.shape) == 3, f"Expecting (batch, M, K) in LHS. Received {lhs.shape}."
    mm = GEMMCompatibility(transposed_lhs=False)
    mm((lhs, rhs), {"NUM_BLOCK_M": NUM_BLOCK_M, "NUM_BLOCK_N": NUM_BLOCK_N, "NUM_BLOCK_K": NUM_BLOCK_K})
    batch_size = lhs.shape[0]
    result = nl.ndarray((batch_size, mm.M, mm.N), dtype=lhs.dtype, buffer=nl.shared_hbm)
    for batch_id in nl.affine_range(batch_size):
        for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
            result_blocks = nl.zeros(
                (nl.par_dim(mm.TILE_M), mm.NUM_BLOCK_N, mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                dtype=result.dtype,
                buffer=nl.sbuf,
            )
            max_vals = nl.zeros((nl.par_dim(mm.TILE_M), mm.TILES_IN_BLOCK_M, 1), dtype=norm_dtype, buffer=nl.sbuf)
            exp_sums = nl.zeros(max_vals.shape, dtype=max_vals.dtype, buffer=nl.sbuf)
            prev_exp_sums = nl.zeros(max_vals.shape, dtype=max_vals.dtype, buffer=nl.sbuf)
            for block_id_K in nl.sequential_range(mm.NUM_BLOCK_K):
                lhs_block = load_tensor_block(
                    input_tensor=lhs[batch_id],
                    ofs=(block_id_M * mm.BLOCK_M, block_id_K * mm.BLOCK_K),
                    load_shape=(mm.TILE_M, mm.TILES_IN_BLOCK_M, 1, mm.BLOCK_K),
                )
                prev_exp_sums[...] = nl.copy(exp_sums[...], dtype=exp_sums.dtype)
                update_max_vals(lhs_block, max_vals)
                # update_exp_sums(lhs_block, exp_sums)
                # # calculate_rms_factors(rms_factors, square_sums, scale=1 / mm.K, eps=eps)
                # if block_id_K > 0:
                #     scale_prev_results(result_blocks, rms_factors, prev_rms_factors)
                # scale_lhs(lhs_block, rms_factors)
                # transpose_tiles_in_block(lhs_block)
                # for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
                #     rhs_block = load_tensor_block(
                #         input_tensor=rhs,
                #         ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                #         load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                #     )
                #     matmul_blocks_tile_transposed_lhs(lhs_block, rhs_block, result_blocks, block_id_N)

            save_result_dma(result[batch_id], result_blocks, block_id_M)

    return result


def update_max_vals(lhs, max_vals):
    """
    Update the max values for the lhs_block
    Args:
        lhs_block: 3D input tensor tile (TILE_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_K, TILE_K)
        max_vals: 3D tensor to store the max values
    """
    TILE_M, TILES_IN_M, unity, K = lhs.shape
    assert unity == 1
    assert max_vals.shape == (TILE_M, TILES_IN_M, 1)
    i_lhs = nl.mgrid[0:TILE_M, 0:K]
    i_max_vals = nl.mgrid[0:TILE_M, 0:1]
    acc_dtype = lhs.dtype  # FIXME
    for tile_id_M in nl.affine_range(TILES_IN_M):
        max_vals[i_max_vals.p, tile_id_M, i_max_vals.x] = nisa.tensor_reduce(
            op=nl.maximum, data=lhs[i_lhs.p, tile_id_M, 0, i_lhs.x], axis=(1,), dtype=acc_dtype, negate=False
        )


def update_exp_sums(lhs, square_sums):
    """
    Update the exp sums for the lhs_block
    Args:
        lhs_block: 3D input tensor tile (TILE_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_K, TILE_K)
        square_sums: 3D tensor to store the square sums
    """
    TILE_M, TILES_IN_M, unity, K = lhs.shape
    assert unity == 1
    assert square_sums.shape == (TILE_M, TILES_IN_M, 1)
    i_lhs = nl.mgrid[0:TILE_M, 0:K]
    i_square_sums = nl.mgrid[0:TILE_M, 0:1]
    for tile_id_M in nl.affine_range(TILES_IN_M):
        tmp_square_sums = nl.ndarray((nl.par_dim(TILE_M), 1), dtype=square_sums.dtype, buffer=nl.sbuf)
        nisa.activation_reduce(
            op=nl.square, data=lhs[i_lhs.p, tile_id_M, 0, i_lhs.x], reduce_op=np.add, reduce_res=tmp_square_sums[...]
        )
        square_sums[i_square_sums.p, tile_id_M, i_square_sums.x] += tmp_square_sums[...]
