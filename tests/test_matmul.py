import pytest
import numpy as np

from src.matmul import (
    matmul_NMK,
    matmul_NKM,
)

import neuronxcc.nki.language as nl
from neuronxcc.starfish.support.util import allclose
from neuronxcc.nki import baremetal

from itertools import product


def cpu_golden(lhsT, rhs):
    # lhsT is the transposed left matrix and rhs is the right matrix
    assert (
        lhsT.shape[1] == rhs.shape[0]
    ), "Matrix dimensions don't match for multiplication"

    # Initialize result matrix with zeros
    result = np.matmul(lhsT.T, rhs)

    return result


def get_tests():
    kernels = [matmul_NMK, matmul_NKM]
    K_sizes = [1024]
    M_sizes = [1024]
    N_sizes = [1024]
    TILES_IN_BLOCK_K = [1]
    TILES_IN_BLOCK_M = [1]
    TILES_IN_BLOCK_N = [1]
    tests = list(
        product(
            kernels,
            K_sizes,
            M_sizes,
            N_sizes,
            TILES_IN_BLOCK_K,
            TILES_IN_BLOCK_M,
            TILES_IN_BLOCK_N,
        )
    )
    return tests


@pytest.mark.parametrize(
    "matmul_kernel, K, M, N, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N",
    get_tests(),
)
def test_matmul(
    matmul_kernel, K, M, N, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N
):
    data_type = np.float32
    lhsT = np.random.random_sample((K, M)).astype(data_type)
    rhs = np.random.random_sample((K, N)).astype(data_type)
    golden_output = nl.static_cast(cpu_golden(lhsT, rhs), np.float32)

    lhsT_dev = nl.static_cast(lhsT, data_type)
    rhs_dev = nl.static_cast(rhs, data_type)
    numeric_func = baremetal(matmul_kernel)
    nki_out = numeric_func(
        lhsT_dev, rhs_dev, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N
    )
    nki_out = nl.static_cast(nki_out, np.float32)

    atol = 1e-2
    rtol = 1e-3
    match = allclose(nki_out, golden_output, atol=atol, rtol=rtol, verbose=1)
    assert match
