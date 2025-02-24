import pytest, random, sys
import numpy as np
from typing import List, Tuple

sys.path.append("../")

from src.matmul import matmul_main, MatMulCompatibility

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


def get_tests(num_tests: int) -> List[Tuple]:
    TILE_K = nl.tile_size.pmax  # 128
    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    TILES_IN_BLOCK_K_vals = [1, 4, 8]
    TILES_IN_BLOCK_M_vals = [1, 4, 8]
    TILES_IN_BLOCK_N_vals = [1, 4, 8]

    BUFFER_K_vals = [1, 2, 4]
    BUFFER_M_vals = [2, 4, 8]
    BUFFER_N_vals = [2, 4, 8]

    NUM_BLOCK_K_vals = [2, 8, 16]
    NUM_BLOCK_M_vals = [2, 4, 16]
    NUM_BLOCK_N_vals = [2, 8, 16]

    configs = list(
        product(
            TILES_IN_BLOCK_K_vals,
            TILES_IN_BLOCK_M_vals,
            TILES_IN_BLOCK_N_vals,
            NUM_BLOCK_K_vals,
            NUM_BLOCK_M_vals,
            NUM_BLOCK_N_vals,
            BUFFER_K_vals,
            BUFFER_M_vals,
            BUFFER_N_vals,
        )
    )
    valid_tests = []
    for config in configs:
        (
            TILES_IN_BLOCK_K,
            TILES_IN_BLOCK_M,
            TILES_IN_BLOCK_N,
            NUM_BLOCK_K,
            NUM_BLOCK_M,
            NUM_BLOCK_N,
            BUFFER_K,
            BUFFER_M,
            BUFFER_N,
        ) = config
        K = NUM_BLOCK_K * TILES_IN_BLOCK_K * TILE_K
        M = NUM_BLOCK_M * TILES_IN_BLOCK_M * TILE_M
        N = NUM_BLOCK_N * TILES_IN_BLOCK_N * TILE_N

        try:
            MatMulCompatibility(
                (K, M), (K, N), TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, BUFFER_K, BUFFER_M, BUFFER_N
            )
            assert max(M, N, K) <= 16384, f"Input sizes are too large"
            test = (K, M, N, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, BUFFER_K, BUFFER_M, BUFFER_N)
            valid_tests.append(test)
        except:
            continue
    # Handle loop orders
    loop_orders = ["".join(p) for p in permutations("MNK")]
    num_tests_per_loop_order = num_tests // len(loop_orders)
    final_tests = []

    for loop_order in loop_orders:
        if not valid_tests:
            break

        # Sample tests for this loop order
        loop_order_tests = random.sample(valid_tests, min(num_tests_per_loop_order, len(valid_tests)))

        # Add loop order to each test
        loop_order_tests = [test + (loop_order,) for test in loop_order_tests]
        final_tests.extend(loop_order_tests)
    return final_tests


@pytest.mark.parametrize(
    "K, M, N, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, BUFFER_K, BUFFER_M, BUFFER_N, loop_order",
    get_tests(30),
)
def test_matmul_correctness(
    K, M, N, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, BUFFER_K, BUFFER_M, BUFFER_N, loop_order
):
    data_type = np.float32
    lhsT = np.random.random_sample((K, M)).astype(data_type)
    rhs = np.random.random_sample((K, N)).astype(data_type)
    golden_output = nl.static_cast(cpu_golden(lhsT, rhs), np.float32)

    lhsT_dev = nl.static_cast(lhsT, data_type)
    rhs_dev = nl.static_cast(rhs, data_type)
    numeric_func = baremetal(matmul_main)
    nki_out = numeric_func(
        lhsT_dev,
        rhs_dev,
        TILES_IN_BLOCK_K,
        TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N,
        BUFFER_K,
        BUFFER_M,
        BUFFER_N,
        loop_order=loop_order,
    )
    nki_out = nl.static_cast(nki_out, np.float32)

    atol = 1e-2
    rtol = 1e-3
    match = allclose(nki_out, golden_output, atol=atol, rtol=rtol, verbose=1)
    assert match
