import sys
from typing import Dict

import numpy as np
import pytest

sys.path.append("../")

from itertools import permutations

import neuronxcc.nki.language as nl
from neuronxcc.nki import baremetal
from neuronxcc.starfish.support.util import allclose

from autotune.core.lhs_rhs import gemm_main
from autotune.core.test_generation import GenTests
from autotune.core.utils import GEMMCompatibility
from autotune.golden.gemm import gemm_core, gemm_cpu_golden
from autotune.tune.utils import run_kernel

SHAPES = [1, 2, 4, 8]


class GEMMTestConfig(GenTests):
    def process_test_config(self, config: Dict[str, int]) -> bool:
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
            check = GEMMCompatibility(transposed_lhs=False)
            check((M, K), (K, N), NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K)
            return True
        except Exception as e:
            return False


@pytest.mark.parametrize(
    "M, N, K", GEMMTestConfig(M=[1024, 2048, 4096], N=[1024, 2048, 4096], K=[1024, 2048, 4096]).sample_tests(10)
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
    "NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K, loop_order",
    GEMMTestConfig(
        NUM_BLOCK_M=SHAPES,
        NUM_BLOCK_N=SHAPES,
        NUM_BLOCK_K=SHAPES,
        TILES_IN_BLOCK_M=SHAPES,
        TILES_IN_BLOCK_N=SHAPES,
        TILES_IN_BLOCK_K=SHAPES,
        loop_order=["".join(p) for p in permutations("MNK")],
    ).sample_tests(10),
)
def test_matmul_correctness(
    NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K, loop_order
):
    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512
    TILE_K = nl.tile_size.pmax  # 128
    M = NUM_BLOCK_M * TILES_IN_BLOCK_M * TILE_M
    N = NUM_BLOCK_N * TILES_IN_BLOCK_N * TILE_N
    K = NUM_BLOCK_K * TILES_IN_BLOCK_K * TILE_K
    data_type = np.float32
    lhsT = np.random.random_sample((K, M)).astype(data_type)
    rhs = np.random.random_sample((K, N)).astype(data_type)
    golden_output = nl.static_cast(gemm_cpu_golden(lhsT, rhs, lhs_is_transposed=True), np.float32)

    lhsT_dev = nl.static_cast(lhsT, data_type)
    rhs_dev = nl.static_cast(rhs, data_type)

    nki_out, metrics = run_kernel(
        kernel_name="matmul_main",
        input_tensors=(lhsT_dev, rhs_dev),
        NUM_BLOCK_M=NUM_BLOCK_M,
        NUM_BLOCK_N=NUM_BLOCK_N,
        NUM_BLOCK_K=NUM_BLOCK_K,
        loop_order=loop_order,
    )
    nki_out = nl.static_cast(nki_out, np.float32)

    atol, rtol = 1e-2, 1e-3
    assert allclose(nki_out, golden_output, atol=atol, rtol=rtol, verbose=1)


@pytest.mark.parametrize(
    "NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K",
    GEMMTestConfig(
        NUM_BLOCK_M=SHAPES,
        NUM_BLOCK_N=[1, 2],
        NUM_BLOCK_K=SHAPES,
        TILES_IN_BLOCK_M=SHAPES,
        TILES_IN_BLOCK_N=SHAPES,
        TILES_IN_BLOCK_K=SHAPES,
    ).sample_tests(10),
)
def test_non_transposed_matmul_correctness(
    NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K
):
    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512
    TILE_K = nl.tile_size.pmax  # 128
    M = NUM_BLOCK_M * TILES_IN_BLOCK_M * TILE_M
    N = NUM_BLOCK_N * TILES_IN_BLOCK_N * TILE_N
    K = NUM_BLOCK_K * TILES_IN_BLOCK_K * TILE_K
    data_type = np.float32
    atol, rtol = 1e-5, 1e-5
    lhs = np.random.random_sample((M, K)).astype(data_type)
    rhs = np.random.random_sample((K, N)).astype(data_type)
    golden_output = nl.static_cast(gemm_cpu_golden(lhs, rhs, lhs_is_transposed=False), data_type)

    for template in ["MN", "MNK", "MKN"]:
        numeric_func = baremetal(gemm_main)
        nki_out = numeric_func(lhs, rhs, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, template)
        nki_out = nl.static_cast(nki_out, data_type)
        assert allclose(
            nki_out, golden_output, atol=atol, rtol=rtol, verbose=1
        ), f"Non transposed LHS GEMM {template} is numerically wrong."
