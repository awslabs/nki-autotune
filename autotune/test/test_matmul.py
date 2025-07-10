import sys
from typing import Dict

import numpy as np
import pytest

sys.path.append("../")

from itertools import permutations

import neuronxcc.nki.language as nl
from neuronxcc.nki import baremetal
from neuronxcc.starfish.support.util import allclose

from autotune.core.golden import blocked_gemm_np_mkn, lhs_rhs_gemm_np, lhsT_rhs_gemm_np
from autotune.modules.lhs_rhs import lhs_rhs_gemm
from autotune.modules.matmul import GEMMCompatibility
from autotune.test.generate_tests import GenTests
from autotune.tune.utils import run_kernel

SHAPES = [1, 2, 4, 8]


class GEMMTestConfig(GenTests):
    def process_test_config(self, config: Dict[str, int]) -> bool:
        batch = config.get("batch", 1)
        M = config.get("M", 1)
        N = config.get("N", 1)
        K = config.get("K", 1)
        lhs = np.zeros((batch, M, K))
        rhs = np.zeros((K, N))
        try:
            assert max(M, N, K) <= 8192, f"Input sizes are too large for testing"
            check = GEMMCompatibility(transposed_lhs=False)
            check(input_tensors=(lhs, rhs), kernel_kwargs={})
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
    lhs = np.random.normal((batch, M, K)).astype(data_type)
    rhs = np.random.normal((K, N)).astype(data_type)

    golden_1 = nl.static_cast(gemm_core(lhs[0], rhs, False), data_type)
    golden_2 = nl.static_cast(lhs_rhs_gemm_np(lhs, rhs, False), data_type)
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
    lhsT = np.random.normal((K, M)).astype(data_type)
    rhs = np.random.normal((K, N)).astype(data_type)
    golden_output = nl.static_cast(lhsT_rhs_gemm_np(lhsT, rhs), np.float32)

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
    lhs = np.random.normal((M, K)).astype(data_type)
    rhs = np.random.normal((K, N)).astype(data_type)
    golden_output = nl.static_cast(lhs_rhs_gemm_np(lhs, rhs, lhs_is_transposed=False), data_type)

    for template in ["MN", "MNK", "MKN"]:
        numeric_func = baremetal(lhs_rhs_gemm)
        nki_out = numeric_func(lhs, rhs, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, template)
        nki_out = nl.static_cast(nki_out, data_type)
        assert allclose(
            nki_out, golden_output, atol=atol, rtol=rtol, verbose=1
        ), f"Non transposed LHS GEMM {template} is numerically wrong."


@pytest.mark.parametrize(
    "batch, M, N, K, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K",
    GEMMTestConfig(
        batch=[1, 2, 4],
        M=[1024, 2048, 4096],
        N=[1024, 2048, 4096],
        K=[1024, 2048, 4096],
        NUM_BLOCK_M=[1, 2, 4],
        NUM_BLOCK_N=[1, 2, 4],
        NUM_BLOCK_K=[1, 2, 4],
    ).sample_tests(10),
)
def test_blocked_gemm_np_numerical(
    batch: int, M: int, N: int, K: int, NUM_BLOCK_M: int, NUM_BLOCK_N: int, NUM_BLOCK_K: int
):
    data_type = np.float32
    lhs = np.random.normal(size=(batch, M, K)).astype(data_type)
    rhs = np.random.normal(size=(K, N)).astype(data_type)

    gemm_golden = lhs_rhs_gemm_np(lhs, rhs)
    blocked_np = blocked_gemm_np_mkn(lhs, rhs, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K)
    np.testing.assert_allclose(
        actual=blocked_np, desired=gemm_golden, atol=1e-3, rtol=1e-3, err_msg="blocked numpy GEMM", verbose=True
    )
