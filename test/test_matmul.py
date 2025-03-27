import pytest, random, sys, warnings
import numpy as np
from typing import List, Tuple, Dict

sys.path.append("../")

from src.kernels.matmul import (
    matmul_main,
    MatMulCompatibility,
    gemm_with_non_transposed_lhs_MN,
    gemm_with_non_transposed_lhs_MNK,
)
from src.golden.gemm import gemm_core, gemm_cpu_golden
from test_generation import GenTests

import neuronxcc.nki.language as nl
from neuronxcc.starfish.support.util import allclose
from neuronxcc.nki import baremetal

from itertools import permutations


class GEMMTestConfig(GenTests):
    def process_test_config(self, config: Dict[str, int]) -> Tuple | None:
        TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
        TILE_N = nl.tile_size.gemm_moving_fmax  # 512
        TILE_K = nl.tile_size.pmax  # 128

        NUM_BLOCK_M = config.get("NUM_BLOCK_M", 1)
        NUM_BLOCK_N = config.get("NUM_BLOCK_N", 1)
        NUM_BLOCK_K = config.get("NUM_BLOCK_K", 1)

        TILES_IN_BLOCK_M = config.get("TILES_IN_BLOCK_M", 1)
        TILES_IN_BLOCK_N = config.get("TILES_IN_BLOCK_N", 1)
        TILES_IN_BLOCK_K = config.get("TILES_IN_BLOCK_K", 1)

        BUFFER_M = config.get("BUFFER_M", 1)
        BUFFER_N = config.get("BUFFER_N", 1)
        BUFFER_K = config.get("BUFFER_K", 1)

        M = config.get("M", NUM_BLOCK_M * TILES_IN_BLOCK_M * TILE_M)
        N = config.get("N", NUM_BLOCK_N * TILES_IN_BLOCK_N * TILE_N)
        K = config.get("K", NUM_BLOCK_K * TILES_IN_BLOCK_K * TILE_K)

        try:
            assert max(M, N, K) <= 8192, f"Input sizes are too large for testing"
            MatMulCompatibility((M, K), (K, N), NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K)
            if "M" in config:
                config_tuple = (M, N, K)
            else:
                config_tuple = (M, N, K, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K)
            return config_tuple
        except Exception as e:
            return None


def get_tests_with_loop_order(num_tests: int, test_loop_order: bool) -> List[Tuple]:
    gen = GEMMTestConfig(
        NUM_BLOCK_M=[1, 2, 4, 16],
        NUM_BLOCK_N=[1, 2, 8, 16],
        NUM_BLOCK_K=[1, 2, 8, 16],
        TILES_IN_BLOCK_M=[1, 4, 8],
        TILES_IN_BLOCK_N=[1, 4, 8],
        TILES_IN_BLOCK_K=[1, 4, 8],
        BUFFER_M=[1, 2, 4, 8],
        BUFFER_N=[1, 2, 4, 8],
        BUFFER_K=[1, 2, 4],
    )
    valid_tests = gen.valid_tests[:num_tests]

    # We want to test every single loop order so it should not be part of the random test generation process
    if test_loop_order:
        loop_orders = ["".join(p) for p in permutations("MNK")]
        tests = []
        for test in valid_tests:
            for loop_order in loop_orders:
                tests.append(test + (loop_order,))
    else:
        tests = valid_tests
    return tests


@pytest.mark.parametrize(
    "M, N, K", GEMMTestConfig(M=[1024, 2048], N=[2048, 4096], K=[1024, 2048, 4096]).valid_tests[:10]
)
def test_golden_matmul_correctness(M, N, K):
    batch = 1
    data_type = np.float32
    atol, rtol = 1e-2, 1e-3
    lhs = np.random.random_sample((batch, M, K)).astype(data_type)
    rhs = np.random.random_sample((K, N)).astype(data_type)

    golden_1 = nl.static_cast(gemm_core(lhs[0], rhs, False), data_type)
    golden_2 = nl.static_cast(gemm_cpu_golden(lhs, rhs, False), data_type)
    assert allclose(golden_1, golden_2, atol=atol, rtol=rtol, verbose=1)


@pytest.mark.parametrize(
    "M, N, K, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K, loop_order",
    get_tests_with_loop_order(2, test_loop_order=True),
)
def test_matmul_correctness(M, N, K, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K, loop_order):
    data_type = np.float32
    atol, rtol = 1e-2, 1e-3
    lhsT = np.random.random_sample((K, M)).astype(data_type)
    rhs = np.random.random_sample((K, N)).astype(data_type)
    golden_output = nl.static_cast(gemm_cpu_golden(lhsT, rhs, lhs_is_transposed=True), np.float32)

    lhsT_dev = nl.static_cast(lhsT, data_type)
    rhs_dev = nl.static_cast(rhs, data_type)
    numeric_func = baremetal(matmul_main)
    nki_out = numeric_func(
        lhsT_dev, rhs_dev, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K, loop_order=loop_order
    )
    nki_out = nl.static_cast(nki_out, np.float32)

    assert allclose(nki_out, golden_output, atol=atol, rtol=rtol, verbose=1)


@pytest.mark.parametrize(
    "M, N, K, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K",
    get_tests_with_loop_order(5, test_loop_order=False),
)
def test_non_transposed_matmul_correctness(
    M, N, K, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K
):
    data_type = np.float32
    atol, rtol = 1e-2, 1e-3
    lhs = np.random.random_sample((M, K)).astype(data_type)
    rhs = np.random.random_sample((K, N)).astype(data_type)
    golden_output = nl.static_cast(gemm_cpu_golden(lhs, rhs, lhs_is_transposed=False), data_type)

    numeric_func = baremetal(gemm_with_non_transposed_lhs_MN)
    nki_out = numeric_func(lhs, rhs, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K)
    nki_out = nl.static_cast(nki_out, data_type)

    assert allclose(
        nki_out, golden_output, atol=atol, rtol=rtol, verbose=1
    ), f"nki_out\n{nki_out}\ngolden_output\n{golden_output}"
