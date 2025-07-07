import neuronxcc.nki.language as nl
import numpy as np

from autotune.typing import INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE, OUTPUT_TENSORS_DTYPE


class GEMMCorrectness:
    def __init__(self, transposed_lhs: bool) -> None:
        self.transposed_lhs = transposed_lhs

    def __call__(
        self,
        input_tensors: INPUT_TENSORS_DTYPE,
        kernel_kwargs: KERNEL_KWARGS_DTYPE,
        nki_out_tensors: OUTPUT_TENSORS_DTYPE,
    ):
        data_type = np.float32
        atol, rtol = 1e-2, 1e-2
        lhs, rhs = input_tensors
        if self.transposed_lhs:
            golden = nl.static_cast(lhsT_rhs_gemm_np(lhs, rhs), data_type)
        else:
            golden = nl.static_cast(lhs_rhs_gemm_np(lhs, rhs), data_type)
        nki_out_tensor = nl.static_cast(nki_out_tensors[0], data_type)
        np.testing.assert_allclose(
            actual=nki_out_tensor, desired=golden, atol=atol, rtol=rtol, err_msg="", verbose=True
        )


def lhs_rhs_gemm_np(lhs, rhs):
    """
    Calculate the general matrix multiplication (GEMM) between lhs and rhs.

    Parameters:
    -----------
    lhs : numpy.ndarray
        Left-hand side matrix or tensor. Can have an extra batch dimension.
    rhs : numpy.ndarray
        Right-hand side matrix.

    Returns:
    --------
    numpy.ndarray
        Result of the matrix multiplication.
    """
    return np.matmul(lhs, rhs)


def lhsT_rhs_gemm_np(lhsT, rhs):
    """
    Calculate the general matrix multiplication (GEMM) between lhsT and rhs.

    Parameters:
    -----------
    lhs : numpy.ndarray
        Left-hand side matrix or tensor. Can have an extra batch dimension.
    rhs : numpy.ndarray
        Right-hand side matrix.

    Returns:
    --------
    numpy.ndarray
        Result of the matrix multiplication.
    """
    if len(lhsT.shape) == 3:  # Batch dimension exists
        lhs = np.transpose(lhsT, (0, 2, 1))
    else:
        lhs = lhsT.T
    return np.matmul(lhs, rhs)


def blocked_gemm_np_mkn(lhs, rhs, NUM_BLOCK_M: int, NUM_BLOCK_N: int, NUM_BLOCK_K: int):
    """
    GEMM algorithm with vectorized block processing using MKN loop ordering.
    """
    batch, M, K = lhs.shape
    K_check, N = rhs.shape

    if K != K_check:
        raise ValueError(f"Incompatible dimensions: lhs K dimension is {K} and rhs K dimension is {K_check}")

    # Calculate block sizes in each dimension
    BLOCK_M = M // NUM_BLOCK_M
    BLOCK_N = N // NUM_BLOCK_N
    BLOCK_K = K // NUM_BLOCK_K

    # Initialize output matrix with the input data type
    output = np.zeros((batch, M, N), dtype=lhs.dtype)

    # Process each batch
    for b in range(batch):
        # Process blocks of M dimension
        for block_m in range(NUM_BLOCK_M):
            m_start = block_m * BLOCK_M
            m_end = min(M, (block_m + 1) * BLOCK_M)
            m_size = m_end - m_start

            # Initialize accumulator for the entire M block with higher precision (float64)
            outputs = np.zeros((m_size, N), dtype=np.float64)

            # Process K dimension in blocks
            for block_k in range(NUM_BLOCK_K):
                k_start = block_k * BLOCK_K
                k_end = min(K, (block_k + 1) * BLOCK_K)

                # Extract current K block for all M rows in this block
                lhs_block = lhs[b, m_start:m_end, k_start:k_end]

                # Process each N block
                for block_n in range(NUM_BLOCK_N):
                    n_start = block_n * BLOCK_N
                    n_end = min(N, (block_n + 1) * BLOCK_N)

                    # Extract current N block for this K block
                    rhs_block = rhs[k_start:k_end, n_start:n_end]

                    # Calculate contribution and add to the higher precision accumulator
                    contribution = np.matmul(lhs_block, rhs_block)
                    outputs[:, n_start:n_end] += contribution

            # Store final results for this M block, casting back to the original data type
            output[b, m_start:m_end, :] = outputs.astype(lhs.dtype)

    return output
