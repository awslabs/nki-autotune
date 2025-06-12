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
        golden = nl.static_cast(gemm_cpu_golden(lhs, rhs, self.transposed_lhs), data_type)
        nki_out_tensor = nl.static_cast(nki_out_tensor, data_type)
        np.testing.assert_allclose(
            actual=nki_out_tensor, desired=golden, atol=atol, rtol=rtol, err_msg="", verbose=True
        )


def gemm_cpu_golden(lhs, rhs, transposed_lhs=False):
    """
    Calculate the general matrix multiplication (GEMM) between lhs and rhs.

    Parameters:
    -----------
    lhs : numpy.ndarray
        Left-hand side matrix or tensor. Can have an extra batch dimension.
        If transposed_lhs=True, this is actually lhs_T (already transposed).
    rhs : numpy.ndarray
        Right-hand side matrix.
    transposed_lhs : bool, default=False
        Indicates if the input lhs is actually lhs_T (the transposed version).
        If True, function will transpose it back before multiplication.

    Returns:
    --------
    numpy.ndarray
        Result of the matrix multiplication.
    """
    if transposed_lhs:
        if len(lhs.shape) == 3:  # Batch dimension exists
            lhs = np.transpose(lhs, (0, 2, 1))
        else:
            lhs = lhs.T
    return np.matmul(lhs, rhs)
