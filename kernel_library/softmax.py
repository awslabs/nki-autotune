import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
from neuronxcc import nki
from neuronxcc.nki.typing import tensor

from autotune.core.dma import load_tensor_block, save_result_dma
from autotune.core.layout import transpose_tiles_in_block
from autotune.core.reductions import compute_max_vals
from autotune.core.scalar_ops import blocked_activation, scale_block
from autotune.core.utils import GEMMCompatibility
from autotune.typing import INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE, OUTPUT_TENSOR_DTYPE


def softmax_gemm_correctness_postprocessing(
    input_tensors: INPUT_TENSORS_DTYPE, kernel_kwargs: KERNEL_KWARGS_DTYPE, kernel_output: OUTPUT_TENSOR_DTYPE
) -> None:
    lhs, rhs = input_tensors
    golden = softmax_gemm_np(lhs, rhs)
    online_golden = online_softmax_gemm_np_mkn(lhs, rhs, **kernel_kwargs)
    kernel_output = nl.static_cast(kernel_output, np.float32)

    atol, rtol = 1e-3, 1e-3
    np.testing.assert_allclose(
        actual=kernel_output,
        desired=online_golden,
        atol=atol,
        rtol=rtol,
        err_msg="kernel_output vs online_golden",
        verbose=True,
    )


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


def online_softmax_gemm_np_mkn(lhs, rhs, NUM_BLOCK_M: int, NUM_BLOCK_N: int, NUM_BLOCK_K: int):
    """
    Online Softmax + GEMM algorithm with vectorized block processing using MKN loop ordering.
    """
    batch, M, K = lhs.shape
    K_check, N = rhs.shape

    if K != K_check:
        raise ValueError(f"Incompatible dimensions: lhs K dimension is {K} and rhs K dimension is {K_check}")

    # Calculate block sizes in each dimension
    BLOCK_M = M // NUM_BLOCK_M
    BLOCK_N = N // NUM_BLOCK_N
    BLOCK_K = K // NUM_BLOCK_K

    # Initialize output matrix
    output = np.zeros((batch, M, N))

    # Process each batch
    for b in range(batch):
        # Process blocks of M dimension
        for block_m in range(NUM_BLOCK_M):
            m_start = block_m * BLOCK_M
            m_end = min(M, (block_m + 1) * BLOCK_M)
            m_size = m_end - m_start

            # Initialize state arrays for the entire M block
            a_vals = np.zeros((m_size, 1))  # Current max values
            a_prev = np.zeros((m_size, 1))  # Previous max values
            b_vals = np.zeros((m_size, 1))  # Current normalization terms
            b_prev = np.zeros((m_size, 1))  # Previous normalization terms
            outputs = np.zeros((m_size, N))  # Collect outputs for all N

            # Process K dimension in blocks
            for block_k in range(NUM_BLOCK_K):
                k_start = block_k * BLOCK_K
                k_end = min(K, (block_k + 1) * BLOCK_K)

                # Extract current K block for all M rows in this block
                lhs_block = lhs[b, m_start:m_end, k_start:k_end]  # Shape: (m_size, BLOCK_K)

                # Calculate block max values once for all N
                block_max_vals = np.max(lhs_block, axis=1, keepdims=True)  # Shape: (m_size, 1)

                if block_k == 0:
                    # For first K block, simply set the values directly
                    a_vals = block_max_vals
                else:
                    # For subsequent blocks, update max and calculate scaling
                    a_vals = np.maximum(a_prev, block_max_vals)
                    # FIXME: proper calculation of scale_factor
                    # scale_factor = np.exp(a_prev - a_vals)  # Shape: (m_size, 1)
                    scale_factor = a_vals

                # Calculate exp(x - a) for all elements in the block
                exp_block = np.exp(lhs_block - a_vals)  # Shape: (m_size, BLOCK_K)
                # Calculate sum of exp values for this block
                exp_sum_block = np.sum(exp_block, axis=1, keepdims=True)  # Shape: (m_size, 1)

                if block_k == 0:
                    b_vals = exp_sum_block
                else:
                    # Update b values (normalization term)
                    b_vals = b_prev * scale_factor + exp_sum_block
                    # Apply rescaling to existing outputs (for all N blocks at once)
                    combined_rescale = scale_factor * (b_prev / b_vals)
                    outputs *= combined_rescale

                # Normalize exp_block for softmax weighting
                softmax_block = exp_block / b_vals

                # Process each N block with the pre-calculated softmax weights
                for block_n in range(NUM_BLOCK_N):
                    n_start = block_n * BLOCK_N
                    n_end = min(N, (block_n + 1) * BLOCK_N)

                    # Extract current N block for this K block
                    rhs_block = rhs[k_start:k_end, n_start:n_end]  # Shape: (BLOCK_K, n_size)

                    # Calculate contribution and add to outputs
                    contribution = np.matmul(softmax_block, rhs_block)
                    outputs[:, n_start:n_end] += contribution

                # Store current values as previous for next iteration
                a_prev[:] = a_vals
                b_prev[:] = b_vals

            # Store final results for this M block
            output[b, m_start:m_end, :] = outputs

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
            a_vals = nl.ndarray((mm.TILE_M, mm.TILES_IN_BLOCK_M, 1, 1), dtype=norm_dtype, buffer=nl.sbuf)
            a_prev = nl.ndarray(a_vals.shape, dtype=a_vals.dtype, buffer=nl.sbuf)
            b_vals = nl.zeros(a_vals.shape, dtype=a_vals.dtype, buffer=nl.sbuf)
            b_prev = nl.zeros(a_vals.shape, dtype=a_vals.dtype, buffer=nl.sbuf)
            for block_id_K in nl.sequential_range(mm.NUM_BLOCK_K):
                lhs_block = load_tensor_block(
                    input_tensor=lhs[batch_id],
                    ofs=(block_id_M * mm.BLOCK_M, block_id_K * mm.BLOCK_K),
                    load_shape=(mm.TILE_M, mm.TILES_IN_BLOCK_M, 1, mm.BLOCK_K),
                )
                block_max_vals = compute_max_vals(input_block=lhs_block)
                scale_factor = nl.ndarray(a_vals.shape, dtype=a_vals.dtype, buffer=nl.sbuf)
                if block_id_K == 0:
                    a_vals[...] = nl.copy(block_max_vals[...], dtype=block_max_vals.dtype)
                else:
                    a_vals[...] = nl.maximum(a_prev, block_max_vals)
                    # FIXME: proper calculation of scale_factor
                    scale_factor[...] = nl.copy(a_vals[...])
                    # compute_scale_factors(scale_factor, a_vals, a_prev)
                exp_block = compute_safe_exp(lhs_block, a_vals)
                exp_sum_block = nl.sum(x=exp_block, axis=[-1], keepdims=True)
                if block_id_K == 0:
                    b_vals[...] = nl.copy(exp_sum_block[...], dtype=exp_sum_block.dtype)
                else:
                    blocked_activation(
                        out_block=b_vals,
                        op=np.copy,
                        data_block=b_prev,
                        scale_block=scale_factor,
                        bias_block=exp_sum_block,
                    )
                    combined_rescale = compute_combined_scale(scale_factor, b_vals, b_prev)
                    scale_prev_result(result_block, combined_rescale)
                scale_block(exp_block, b_vals, "divide")
                transpose_tiles_in_block(exp_block)
                for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
                    rhs_block = load_tensor_block(
                        input_tensor=rhs,
                        ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                        load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                    )
                    matmul_blocks_tile_transposed_lhs(exp_block, rhs_block, result_block, block_id_N)

                # Update previous values
                a_prev[...] = nl.copy(a_vals[...], dtype=a_vals.dtype)
                b_prev[...] = nl.copy(b_vals[...], dtype=b_vals.dtype)

            save_result_dma(result[batch_id], result_block, block_id_M)

    return result


def compute_combined_scale(scale_factor, b_vals, b_prev):
    assert b_vals.shape == scale_factor.shape
    assert b_prev.shape == scale_factor.shape

    combined_scale_factor = nl.ndarray(scale_factor.shape, dtype=scale_factor.dtype, buffer=nl.sbuf)

    par_size = scale_factor.shape[0]
    free_size = scale_factor.shape[-1]
    idx_scale_factor = nl.mgrid[0:par_size, 0:free_size]
    block_sizes = scale_factor.shape[1:-1]
    num_blocks = 1
    for dim in block_sizes:
        num_blocks *= dim
    for block_id in nl.affine_range(num_blocks):
        block_indices = []
        remaining = block_id
        for dim in reversed(block_sizes):
            block_indices.insert(0, remaining % dim)
            remaining //= dim
        coordinates = tuple([idx_scale_factor.p] + block_indices + [idx_scale_factor.x])
        combined_scale_factor[coordinates] = nl.divide(b_prev[coordinates], b_vals[coordinates])
        combined_scale_factor[coordinates] = nl.multiply(combined_scale_factor[coordinates], scale_factor[coordinates])
    return combined_scale_factor


def scale_prev_result(result_block, scale_factor):
    TILE_M, NUM_BLOCK_N, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILE_N = result_block.shape
    assert scale_factor.shape == (TILE_M, TILES_IN_BLOCK_M, 1, 1)
    idx_scale_factor = nl.mgrid[0:TILE_M, 0:1]
    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    for block_id_N in nl.affine_range(NUM_BLOCK_N):
        for tile_id_M in nl.affine_range(TILES_IN_BLOCK_M):
            scale_factor_coordinates = idx_scale_factor.p, tile_id_M, 0, idx_scale_factor.x
            for tile_id_N in nl.affine_range(TILES_IN_BLOCK_N):
                result_coordinates = idx_res.p, block_id_N, tile_id_M, tile_id_N, idx_res.x
                result_block[result_coordinates] = nl.multiply(
                    result_block[result_coordinates], scale_factor[scale_factor_coordinates]
                )


def compute_safe_exp(input_block, a_vals):
    """
    exp_block = np.exp(input_block - a_vals)

    Args:
        input_block (_type_): (par_size, *block_sizes, bcast_size)
        a_vals (_type_): (par_size, *block_sizes, 1)
    """
    assert a_vals.shape[:-1] == input_block.shape[:-1]
    exp_block = nl.ndarray(input_block.shape, dtype=input_block.dtype, buffer=nl.sbuf)
    par_size = input_block.shape[0]
    bcast_size = input_block.shape[-1]
    free_size = a_vals.shape[-1]
    assert free_size == 1

    i_input = nl.mgrid[0:par_size, 0:bcast_size]
    i_a_vals = nl.mgrid[0:par_size, 0:free_size]

    block_sizes = input_block.shape[1:-1]
    num_blocks = 1
    for dim in block_sizes:
        num_blocks *= dim
    for block_id in nl.affine_range(num_blocks):
        block_indices = []
        remaining = block_id
        for dim in reversed(block_sizes):
            block_indices.insert(0, remaining % dim)
            remaining //= dim
        a_vals_coordinates = [i_a_vals.p] + block_indices + [i_a_vals.x]
        input_coordinates = [i_input.p] + block_indices + [i_input.x]
        exp_block[tuple(input_coordinates)] = nisa.activation(
            op=np.exp,
            data=input_block[tuple(input_coordinates)],
            bias=-1 * a_vals[tuple(a_vals_coordinates)],
            dtype=input_block.dtype,
        )
    return exp_block


def compute_scale_factors(scale_factor, a_vals, a_prev):
    """
    Computes scale factors by applying exp(a_prev - a_vals) element-wise.

    This function calculates scaling factors by applying the exponential function to
    the difference between previous activation values and current activation values.
    The computation is performed independently for each block in the multi-dimensional tensor.

    Args:
        a_vals (nl.ndarray): Current activation values tensor with shape (par_size, *block_sizes, free_size).
        a_prev (nl.ndarray): Previous activation values tensor with the same shape as a_vals.

    Returns:
        nl.ndarray: Scale factors tensor with the same shape as inputs, containing
                    exp(a_prev - a_vals) for each element.

    Raises:
        AssertionError: If a_vals and a_prev have different shapes.

    Note:
        This function processes the input tensors block by block, where blocks are determined
        by the middle dimensions of the input tensors.
    """
    assert a_vals.shape == a_prev.shape
    par_size = a_vals.shape[0]
    free_size = a_vals.shape[-1]
    num_blocks = 1
    for dim in a_vals.shape[1:-1]:
        num_blocks *= dim
    i_a_vals = nl.mgrid[0:par_size, 0:free_size]
    for block_id in nl.affine_range(num_blocks):
        block_indices = []
        remaining = block_id
        for dim in reversed(a_vals.shape[1:-1]):
            block_indices.insert(0, remaining % dim)
            remaining //= dim
        a_vals_coordinates = [i_a_vals.p] + block_indices + [i_a_vals.x]
        scale_factor[tuple(a_vals_coordinates)] = nisa.activation(
            op=np.exp,
            data=a_vals[tuple(a_vals_coordinates)],
            bias=a_prev[tuple(a_vals_coordinates)],
            dtype=a_vals.dtype,
            scale=-1.0,
        )


def matmul_blocks_tile_transposed_lhs(tileT_lhs_block, rhs_block, result_blocks, block_id_N):
    """
    Accumulate matmul result tiles between lhs and rhs into result_block
    LHS is transposed at the tile level.
    Note that this is not the same as lhsT.
    'matmul_block' module computes lhsT @ rhs.

    Args:
    tileT_lhs_block: TILE_M, TILES_IN_BLOCK_M, unity, BLOCK_K
    rhs_block: TILE_K, TILES_IN_BLOCK_K, TILES_IN_BLOCK_N, TILE_N
    result_block : TILE_M, NUM_BLOCK_N, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILE_N
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
