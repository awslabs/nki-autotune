import pytest
import numpy as np

import sys

sys.path.append("../")

from src.matmul import (
    matmul_NMK,
    matmul_MNK,
    matmul_KMN,
    matmul_KNM,
    matmul_NKM,
    matmul_MKN,
)

import neuronxcc.nki.language as nl
from neuronxcc.starfish.support.util import allclose
from neuronxcc.nki import baremetal

from itertools import product


def cpu_golden(lhsT, rhs):
    # lhsT is the transposed left matrix and rhs is the right matrix
    assert (
        lhsT.shape[0] == rhs.shape[0]
    ), f"Contraction dimensions don't match: {lhsT.shape} {rhs.shape}"

    # Initialize result matrix with zeros
    result = np.matmul(lhsT.T, rhs)

    return result


def get_tests():
    kernels = [matmul_NMK, matmul_MNK, matmul_KMN, matmul_KNM, matmul_NKM]

    TILE_K = nl.tile_size.pmax  # 128
    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    TILES_IN_BLOCK_K_vals = [4, 8]
    TILES_IN_BLOCK_M_vals = [1, 8]
    TILES_IN_BLOCK_N_vals = [1, 4]

    NUM_BLOCK_K_vals = [8]
    NUM_BLOCK_M_vals = [4]
    NUM_BLOCK_N_vals = [8]

    configs = list(
        product(
            kernels,
            TILES_IN_BLOCK_K_vals,
            TILES_IN_BLOCK_M_vals,
            TILES_IN_BLOCK_N_vals,
            NUM_BLOCK_K_vals,
            NUM_BLOCK_M_vals,
            NUM_BLOCK_N_vals,
        )
    )
    tests = []
    for config in configs:
        (
            kernel,
            TILES_IN_BLOCK_K,
            TILES_IN_BLOCK_M,
            TILES_IN_BLOCK_N,
            NUM_BLOCK_K,
            NUM_BLOCK_M,
            NUM_BLOCK_N,
        ) = config
        K = NUM_BLOCK_K * TILES_IN_BLOCK_K * TILE_K
        M = NUM_BLOCK_M * TILES_IN_BLOCK_M * TILE_M
        N = NUM_BLOCK_N * TILES_IN_BLOCK_N * TILE_N
        if all([M >= 1024, N >= 1024, K >= 1024]):
            test = (
                kernel,
                K,
                M,
                N,
                TILES_IN_BLOCK_K,
                TILES_IN_BLOCK_M,
                TILES_IN_BLOCK_N,
            )
            tests.append(test)
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
