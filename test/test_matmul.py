import pytest, random, sys, warnings
import numpy as np
from typing import List, Tuple

sys.path.append("../")

from src.kernels.matmul import matmul_main, MatMulCompatibility, gemm_with_non_transposed_lhs

import neuronxcc.nki.language as nl
from neuronxcc.starfish.support.util import allclose
from neuronxcc.nki import baremetal

from itertools import product, permutations


def cpu_golden(lhsT, rhs):
    # lhsT is the transposed left matrix and rhs is the right matrix
    assert lhsT.shape[0] == rhs.shape[0], f"Contraction dimensions don't match: {lhsT.shape} {rhs.shape}"

    # Initialize result matrix with zeros
    result = np.matmul(lhsT.T, rhs)

    return result


def non_transposed_cpu_golden(lhs, rhs):
    assert lhs.shape[1] == rhs.shape[0], f"Contraction dimensions don't match: {lhs.shape} {rhs.shape}"

    # Initialize result matrix with zeros
    result = np.matmul(lhs, rhs)

    return result


def get_tests(num_tests: int, mutate_loop_order: bool) -> List[Tuple]:
    NUM_BLOCK_M_vals = [2, 4, 16]
    NUM_BLOCK_N_vals = [2, 8, 16]
    NUM_BLOCK_K_vals = [2, 8, 16]

    TILES_IN_BLOCK_M_vals = [1, 4, 8]
    TILES_IN_BLOCK_N_vals = [1, 4, 8]
    TILES_IN_BLOCK_K_vals = [1, 4, 8]

    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512
    TILE_K = nl.tile_size.pmax  # 128

    BUFFER_M_vals = [2, 4, 8]
    BUFFER_N_vals = [2, 4, 8]
    BUFFER_K_vals = [1, 2, 4]

    configs = list(
        product(
            NUM_BLOCK_M_vals,
            NUM_BLOCK_N_vals,
            NUM_BLOCK_K_vals,
            TILES_IN_BLOCK_M_vals,
            TILES_IN_BLOCK_N_vals,
            TILES_IN_BLOCK_K_vals,
            BUFFER_M_vals,
            BUFFER_N_vals,
            BUFFER_K_vals,
        )
    )
    random.shuffle(configs)
    valid_tests = []
    for config in configs:
        (
            NUM_BLOCK_M,
            NUM_BLOCK_N,
            NUM_BLOCK_K,
            TILES_IN_BLOCK_M,
            TILES_IN_BLOCK_N,
            TILES_IN_BLOCK_K,
            BUFFER_M,
            BUFFER_N,
            BUFFER_K,
        ) = config
        M = NUM_BLOCK_M * TILES_IN_BLOCK_M * TILE_M
        N = NUM_BLOCK_N * TILES_IN_BLOCK_N * TILE_N
        K = NUM_BLOCK_K * TILES_IN_BLOCK_K * TILE_K

        try:
            MatMulCompatibility((M, K), (K, N), NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K)
            assert max(M, N, K) <= 8192, f"Input sizes are too large for testing"
            test = (M, N, K, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K)
            valid_tests.append(test)
        except Exception as e:
            print(e)
        if len(valid_tests) == num_tests:
            break
    assert valid_tests, f"No valid tests found"

    if mutate_loop_order:
        loop_orders = ["".join(p) for p in permutations("MNK")]
        tests = []
        for loop_order in loop_orders:
            loop_order_tests = [test + (loop_order,) for test in valid_tests]
            tests.extend(loop_order_tests)
    else:
        tests = valid_tests
    return tests


@pytest.mark.parametrize(
    "M, N, K, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K, loop_order",
    get_tests(2, mutate_loop_order=True),
)
def test_matmul_correctness(M, N, K, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K, loop_order):
    data_type = np.float32
    lhsT = np.random.random_sample((K, M)).astype(data_type)
    rhs = np.random.random_sample((K, N)).astype(data_type)
    golden_output = nl.static_cast(cpu_golden(lhsT, rhs), np.float32)

    lhsT_dev = nl.static_cast(lhsT, data_type)
    rhs_dev = nl.static_cast(rhs, data_type)
    numeric_func = baremetal(matmul_main)
    nki_out = numeric_func(
        lhsT_dev, rhs_dev, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K, loop_order=loop_order
    )
    nki_out = nl.static_cast(nki_out, np.float32)

    atol = 1e-2
    rtol = 1e-3
    match = allclose(nki_out, golden_output, atol=atol, rtol=rtol, verbose=1)
    assert match


@pytest.mark.parametrize(
    "M, N, K, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K",
    get_tests(1, mutate_loop_order=False),
)
def test_non_transposed_matmul_correctness(
    M, N, K, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K
):
    data_type = np.float32
    lhs = np.random.random_sample((M, K)).astype(data_type)
    rhs = np.random.random_sample((K, N)).astype(data_type)
    golden_output = nl.static_cast(non_transposed_cpu_golden(lhs, rhs), np.float32)

    lhs_dev = nl.static_cast(lhs, data_type)
    rhs_dev = nl.static_cast(rhs, data_type)
    numeric_func = baremetal(gemm_with_non_transposed_lhs)
    nki_out = numeric_func(lhs_dev, rhs_dev, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K)
    nki_out = nl.static_cast(nki_out, np.float32)

    atol = 1e-2
    rtol = 1e-3
    match = allclose(nki_out, golden_output, atol=atol, rtol=rtol, verbose=1)
    assert match
