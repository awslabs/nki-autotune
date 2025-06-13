import neuronxcc.nki.language as nl
import numpy as np

from autotune.typing import INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE, OUTPUT_TENSOR_DTYPE


class GEMMCorrectness:
    def __init__(self, transposed_lhs: bool) -> None:
        self.transposed_lhs = transposed_lhs

    def __call__(
        self,
        input_tensors: INPUT_TENSORS_DTYPE,
        kernel_kwargs: KERNEL_KWARGS_DTYPE,
        nki_out_tensor: OUTPUT_TENSOR_DTYPE,
    ):
        data_type = np.float32
        atol, rtol = 1e-2, 1e-2
        lhs, rhs = input_tensors
        if self.transposed_lhs:
            golden = nl.static_cast(lhsT_rhs_gemm_np(lhs, rhs), data_type)
        else:
            golden = nl.static_cast(lhs_rhs_gemm_np(lhs, rhs), data_type)
        nki_out_tensor = nl.static_cast(nki_out_tensor, data_type)
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
