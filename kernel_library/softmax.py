import math

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
from neuronxcc import nki
from neuronxcc.nki.typing import tensor

from autotune.core.dma import load_tensor_block, save_result_dma
from autotune.core.layout import transpose_tiles_in_block
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
    BLOCK_M = 1024
    BLOCK_N = 1024
    BLOCK_K = 1024

    # Calculate number of blocks in each dimension
    NUM_BLOCK_M = math.ceil(M / BLOCK_M)
    NUM_BLOCK_N = math.ceil(N / BLOCK_N)
    NUM_BLOCK_K = math.ceil(K / BLOCK_K)

    # Initialize output matrix
    output = np.zeros((batch, M, N))

    # Process each batch
    for b in range(batch):
        # Process blocks of M dimension
        for block_m in range(NUM_BLOCK_M):
            m_start = block_m * BLOCK_M
            m_end = min(M, (block_m + 1) * BLOCK_M)

            # Process blocks of N dimension
            for block_n in range(NUM_BLOCK_N):
                n_start = block_n * BLOCK_N
                n_end = min(N, (block_n + 1) * BLOCK_N)

                # Initialize state arrays for this M,N block
                a_vals = np.zeros((BLOCK_M, 1))
                a_prev = np.zeros(a_vals.shape)
                b_vals = np.zeros((BLOCK_M, 1))  # normalization terms, shape (BLOCK_M, 1)
                b_prev = np.zeros(b_vals.shape)
                o_vals = np.zeros((BLOCK_M, BLOCK_N))  # output values

                # Process K dimension in blocks
                for block_k in range(NUM_BLOCK_K):
                    k_start = block_k * BLOCK_K
                    k_end = min(K, (block_k + 1) * BLOCK_K)

                    # Extract the current K-block from inputs
                    lhs_block = lhs[b, m_start:m_end, k_start:k_end]  # Shape: (BLOCK_M, BLOCK_K)
                    rhs_block = rhs[k_start:k_end, n_start:n_end]  # Shape: (BLOCK_K, BLOCK_N)

                    # Calculate row-wise max values for this K block
                    block_max_vals = np.max(lhs_block, axis=1, keepdims=True)  # Shape: (BLOCK_M, 1)

                    if block_k == 0:
                        # Initialize a_vals for the first block
                        a_vals = block_max_vals  # Shape: (BLOCK_M, 1)
                    else:

                        # Update global max for each row (m)
                        a_vals = np.maximum(a_prev, block_max_vals)

                        # Calculate scaling factor for the change in max values
                        scale_factor = np.exp(a_prev - a_vals)  # Shape: (BLOCK_M, 1)

                    # Calculate exp(x - a) for all elements in the block
                    exp_block = np.exp(lhs_block - a_vals)  # Shape: (BLOCK_M, BLOCK_K)

                    # Calculate sum of exp values for this block
                    exp_sum_block = np.sum(exp_block, axis=1, keepdims=True)  # Shape: (BLOCK_M, 1)

                    # Update b values (normalization term)
                    # Scale previous output by the change in max and normalization factor
                    if block_k == 0:
                        b_vals = exp_sum_block
                    else:
                        b_vals = b_prev * scale_factor + exp_sum_block
                        rescale = scale_factor * (b_prev / b_vals)
                        o_vals *= rescale

                    # Calculate contribution from this block
                    # Normalize exp_block by its sum for softmax weighting
                    softmax_block = exp_block / b_vals  # Shape: (BLOCK_M, BLOCK_K)

                    # Calculate weighted contribution from this block
                    contribution = np.matmul(softmax_block, rhs_block)  # Shape: (BLOCK_M, BLOCK_N)

                    # Add contribution to output
                    o_vals = o_vals + contribution

                    # Store previous values
                    b_prev = b_vals.copy()
                    a_prev = a_vals.copy()

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
            result_block = nl.zeros(
                (nl.par_dim(mm.TILE_M), mm.NUM_BLOCK_N, mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                dtype=result.dtype,
                buffer=nl.sbuf,
            )
            max_vals = nl.ndarray((nl.par_dim(mm.TILE_M), mm.TILES_IN_BLOCK_M, 1), dtype=norm_dtype, buffer=nl.sbuf)
            prev_max_vals = nl.ndarray(
                (nl.par_dim(mm.TILE_M), mm.TILES_IN_BLOCK_M, 1), dtype=norm_dtype, buffer=nl.sbuf
            )
            exp_sums = nl.zeros(max_vals.shape, dtype=max_vals.dtype, buffer=nl.sbuf)
            prev_exp_sums = nl.zeros(max_vals.shape, dtype=max_vals.dtype, buffer=nl.sbuf)
            for block_id_K in nl.sequential_range(mm.NUM_BLOCK_K):
                lhs_block = load_tensor_block(
                    input_tensor=lhs[batch_id],
                    ofs=(block_id_M * mm.BLOCK_M, block_id_K * mm.BLOCK_K),
                    load_shape=(mm.TILE_M, mm.TILES_IN_BLOCK_M, 1, mm.BLOCK_K),
                )
                block_max_vals = nl.ndarray(
                    (nl.par_dim(mm.TILE_M), mm.TILES_IN_BLOCK_M, 1), dtype=lhs_block.dtype, buffer=nl.sbuf
                )
                exp_block = nl.ndarray(
                    (nl.par_dim(mm.TILE_M), mm.TILES_IN_BLOCK_M, 1, mm.BLOCK_K), dtype=lhs_block.dtype, buffer=nl.sbuf
                )
                sum_exp_block = nl.ndarray(
                    (nl.par_dim(mm.TILE_M), mm.TILES_IN_BLOCK_M, 1), dtype=lhs_block.dtype, buffer=nl.sbuf
                )
                scaling_factors = nl.ndarray(
                    (nl.par_dim(mm.TILE_M), mm.TILES_IN_BLOCK_M, 1), dtype=max_vals.dtype, buffer=nl.sbuf
                )
                compute_max_vals(input_tile=lhs_block, tile_max_vals=block_max_vals)
                if block_id_K == 0:
                    max_vals[...] = nl.copy(block_max_vals[...], dtype=block_max_vals.dtype)
                else:
                    max_vals[...] = nl.maximum(prev_max_vals, block_max_vals)
                    compute_scaling_factors(max_vals, prev_max_vals, scaling_factors)
                compute_safe_exp(lhs_block, exp_block, max_vals)
                compute_sum_exp(exp_block, sum_exp_block)
                if block_id_K == 0:
                    exp_sums[...] = nl.copy(sum_exp_block[...], dtype=sum_exp_block.dtype)
                else:
                    update_exp_sums(prev_exp_sums, exp_sums, scaling_factors, sum_exp_block)
                if block_id_K > 0:
                    scale_prev_results(result_block, scaling_factors, exp_sums, prev_exp_sums)
                scale_lhs(exp_block, exp_sums)
                transpose_tiles_in_block(exp_block)
                for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
                    rhs_block = load_tensor_block(
                        input_tensor=rhs,
                        ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                        load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                    )
                    matmul_blocks_tile_transposed_lhs(exp_block, rhs_block, result_block, block_id_N)

                # Update previous values
                prev_max_vals[...] = nl.copy(max_vals[...], dtype=max_vals.dtype)
                prev_exp_sums[...] = nl.copy(exp_sums[...], dtype=exp_sums.dtype)

            save_result_dma(result[batch_id], result_block, block_id_M)

    return result


def scale_lhs(lhs, scaling_factors):
    TILE_M, TILES_IN_BLOCK_M, unity, BLOCK_K = lhs.shape
    assert unity == 1
    assert scaling_factors.shape == lhs.shape[:-1], f"scaling_factors {scaling_factors.shape} lhs {lhs.shape}"
    i_lhs = nl.mgrid[0:TILE_M, 0:BLOCK_K]
    idx_scaling_factors = nl.mgrid[0:TILE_M, 0:1]
    for tile_id_M in nl.affine_range(TILES_IN_BLOCK_M):
        lhs[i_lhs.p, tile_id_M, 0, i_lhs.x] = nl.divide(
            lhs[i_lhs.p, tile_id_M, 0, i_lhs.x],
            scaling_factors[idx_scaling_factors.p, tile_id_M, idx_scaling_factors.x],
        )


def scale_prev_results(result_block, scaling_factors, exp_sums, prev_exp_sums):
    TILE_M, NUM_BLOCK_N, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILE_N = result_block.shape
    assert scaling_factors.shape == (TILE_M, TILES_IN_BLOCK_M, 1)
    assert exp_sums.shape == (TILE_M, TILES_IN_BLOCK_M, 1)
    assert prev_exp_sums.shape == exp_sums.shape
    idx_scaling_factors = nl.mgrid[0:TILE_M, 0:1]
    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    tmp_scaling_factors = nl.ndarray(
        (nl.par_dim(TILE_M), TILES_IN_BLOCK_M, 1), dtype=scaling_factors.dtype, buffer=nl.sbuf
    )
    for tile_id_M in nl.affine_range(TILES_IN_BLOCK_M):
        tmp_scaling_factors[idx_scaling_factors.p, tile_id_M, idx_scaling_factors.x] = nl.divide(
            prev_exp_sums[idx_scaling_factors.p, tile_id_M, idx_scaling_factors.x],
            exp_sums[idx_scaling_factors.p, tile_id_M, idx_scaling_factors.x],
        )
        tmp_scaling_factors[idx_scaling_factors.p, tile_id_M, idx_scaling_factors.x] = nl.multiply(
            tmp_scaling_factors[idx_scaling_factors.p, tile_id_M, idx_scaling_factors.x],
            scaling_factors[idx_scaling_factors.p, tile_id_M, idx_scaling_factors.x],
        )
    for block_id_N in nl.affine_range(NUM_BLOCK_N):
        for tile_id_M in nl.affine_range(TILES_IN_BLOCK_M):
            for tile_id_N in nl.affine_range(TILES_IN_BLOCK_N):
                result_block[idx_res.p, block_id_N, tile_id_M, tile_id_N, idx_res.x] = nl.multiply(
                    result_block[idx_res.p, block_id_N, tile_id_M, tile_id_N, idx_res.x],
                    tmp_scaling_factors[idx_scaling_factors.p, tile_id_M, idx_scaling_factors.x],
                )


def compute_safe_exp(input_tile, exp_tile, max_vals):
    TILE_M, TILES_IN_M, unity, K = input_tile.shape
    assert unity == 1
    assert exp_tile.shape == input_tile.shape
    assert max_vals.shape == (TILE_M, TILES_IN_M, 1)
    i_input = nl.mgrid[0:TILE_M, 0:K]
    i_max_vals = nl.mgrid[0:TILE_M, 0:1]
    for tile_id_M in nl.affine_range(TILES_IN_M):
        exp_tile[i_input.p, tile_id_M, 0, i_input.x] = nisa.activation(
            op=np.exp,
            data=input_tile[i_input.p, tile_id_M, 0, i_input.x],
            bias=-1 * max_vals[i_max_vals.p, tile_id_M, i_max_vals.x],
            dtype=input_tile.dtype,
        )


def compute_max_vals(input_tile, tile_max_vals):
    """
    Update the max values for the input_tile
    Args:
        input_tile: 4D input tensor tile (TILE_M, TILES_IN_BLOCK_M, 1, BLOCK_K)
        tile_max_vals: 3D input tensor tile (TILE_M, TILES_IN_BLOCK_M, 1)
    """
    TILE_M, TILES_IN_M, unity, K = input_tile.shape
    assert unity == 1
    assert tile_max_vals.shape == (TILE_M, TILES_IN_M, 1)
    i_input = nl.mgrid[0:TILE_M, 0:K]
    i_max_vals = nl.mgrid[0:TILE_M, 0:1]
    for tile_id_M in nl.affine_range(TILES_IN_M):
        tile_max_vals[i_max_vals.p, tile_id_M, i_max_vals.x] = nisa.tensor_reduce(
            op=np.max,
            data=input_tile[i_input.p, tile_id_M, 0, i_input.x],
            axis=(1,),
            dtype=input_tile.dtype,
            negate=False,
        )


def compute_scaling_factors(max_vals, prev_max_vals, scaling_factors):
    TILE_M, TILES_IN_M, unity = max_vals.shape
    assert unity == 1
    assert prev_max_vals.shape == max_vals.shape
    assert scaling_factors.shape == max_vals.shape
    i_max_vals = nl.mgrid[0:TILE_M, 0:1]
    for tile_id_M in nl.affine_range(TILES_IN_M):
        scaling_factors[i_max_vals.p, tile_id_M, i_max_vals.x] = nisa.activation(
            op=np.exp,
            data=max_vals[i_max_vals.p, tile_id_M, i_max_vals.x],
            bias=prev_max_vals[i_max_vals.p, tile_id_M, i_max_vals.x],
            dtype=max_vals.dtype,
            scale=-1.0,
        )


def compute_sum_exp(exp_block, sum_exp_block):
    TILE_M, TILES_IN_M, unity, K = exp_block.shape
    assert unity == 1
    assert sum_exp_block.shape == (TILE_M, TILES_IN_M, 1)
    i_exp_block = nl.mgrid[0:TILE_M, 0:K]
    i_sum_exp_block = nl.mgrid[0:TILE_M, 0:1]
    for tile_id_M in nl.affine_range(TILES_IN_M):
        sum_exp_block[i_sum_exp_block.p, tile_id_M, i_sum_exp_block.x] = nisa.tensor_reduce(
            op=np.add,
            data=exp_block[i_exp_block.p, tile_id_M, 0, i_exp_block.x],
            axis=(1,),
            dtype=exp_block.dtype,
            negate=False,
        )


def update_exp_sums(prev_exp_sums, exp_sums, scaling_factors, curr_block_exp_sums):
    """
    Update the exp sums
    """
    TILE_M, TILES_IN_M, unity = prev_exp_sums.shape
    assert unity == 1
    assert exp_sums.shape == prev_exp_sums.shape
    assert scaling_factors.shape == prev_exp_sums.shape
    assert curr_block_exp_sums.shape == prev_exp_sums.shape
    i_exp_sums = nl.mgrid[0:TILE_M, 0:1]
    for tile_id_M in nl.affine_range(TILES_IN_M):
        exp_sums[i_exp_sums.p, tile_id_M, i_exp_sums.x] = nisa.activation(
            op=np.copy,
            data=prev_exp_sums[i_exp_sums.p, tile_id_M, i_exp_sums.x],
            bias=curr_block_exp_sums[i_exp_sums.p, tile_id_M, i_exp_sums.x],
            scale=scaling_factors[i_exp_sums.p, tile_id_M, i_exp_sums.x],
        )


def matmul_blocks_tile_transposed_lhs(tileT_lhs_block, rhs_block, result_blocks, block_id_N):
    """
    Accumulate matmul result tiles between lhs and rhs into result_block
    LHS is transposed at the tile level.
    Note that this is not the same as lhsT.
    'matmul_block' module computes lhsT @ rhs.

    Args:
    tileT_lhs_block: TILE_M, TILE_K, TILES_IN_M, TILES_IN_K
    rhs_block: TILE_K, TILE_N, TILES_IN_K, TILES_IN_N
    result_block : TILE_M, TILE_N, TILES_IN_M, TILES_IN_N
    """
    TILE_M, TILES_IN_BLOCK_M, unity, BLOCK_K = tileT_lhs_block.shape
    TILE_K, TILES_IN_BLOCK_K, TILES_IN_BLOCK_N, TILE_N = rhs_block.shape
    _TILE_M, NUM_BLOCK_N, _TILES_IN_BLOCK_M, _TILES_IN_BLOCK_N, _TILE_N = result_blocks.shape
    assert unity == 1
    assert (
        TILE_K * TILES_IN_BLOCK_K == BLOCK_K
    ), f"K dimension mismatch: tileT_lhs_block {tileT_lhs_block.shape}. rhs_block {rhs_block.shape}."
    assert (
        TILE_M == _TILE_M and TILES_IN_BLOCK_M == _TILES_IN_BLOCK_M
    ), f"LHS and result shape mismatch: tileT_lhs_block {tileT_lhs_block.shape}. result_blocks {result_blocks.shape}."
    assert (
        TILE_N == _TILE_N and TILES_IN_BLOCK_N == _TILES_IN_BLOCK_N
    ), f"RHS and result shape mismatch: rhs_block {rhs_block.shape}. result_blocks {result_blocks.shape}."

    idx_lhs = nl.mgrid[0:TILE_M, 0:TILE_K]
    idx_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    for tile_id_M in nl.affine_range(TILES_IN_BLOCK_M):
        for tile_id_N in nl.affine_range(TILES_IN_BLOCK_N):
            result_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            """
            Use PSUM buffer to accumulate into a single hardware tile
            """
            for tile_id_K in nl.affine_range(TILES_IN_BLOCK_K):
                k_ofs = tile_id_K * TILE_K
                result_tile += nisa.nc_matmul(
                    tileT_lhs_block[idx_lhs.p, tile_id_M, 0, k_ofs + idx_lhs.x],
                    rhs_block[idx_rhs.p, tile_id_K, tile_id_N, idx_rhs.x],
                )
            result_blocks[idx_res.p, block_id_N, tile_id_M, tile_id_N, idx_res.x] = nl.add(
                result_blocks[idx_res.p, block_id_N, tile_id_M, tile_id_N, idx_res.x], result_tile[idx_res.p, idx_res.x]
            )
