import os
import sys
from typing import Dict

import numpy as np
import pytest

from autotune.core.generate_tests import GenTests
from autotune.core.utils import GEMMCompatibility

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_library.softmax import online_softmax_gemm_np, online_softmax_gemm_np_mkn, softmax_gemm_np


class SoftmaxGemmTestConfig(GenTests):
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
    "batch, M, N, K, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K",
    SoftmaxGemmTestConfig(
        batch=[1, 2, 4],
        M=[1024, 2048, 4096],
        N=[1024, 2048, 4096],
        K=[1024, 2048, 4096],
        NUM_BLOCK_M=[1, 2, 4],
        NUM_BLOCK_N=[1, 2, 4],
        NUM_BLOCK_K=[1, 2, 4],
    ).sample_tests(50),
)
def test_softmax_gemm_np_numerical(
    batch: int, M: int, N: int, K: int, NUM_BLOCK_M: int, NUM_BLOCK_N: int, NUM_BLOCK_K: int
):
    data_type = np.float32
    lhs = np.random.normal(size=(batch, M, K)).astype(data_type)
    rhs = np.random.normal(size=(K, N)).astype(data_type)
    golden = softmax_gemm_np(lhs, rhs)
    online_np = online_softmax_gemm_np(lhs, rhs)
    online_np_mkn = online_softmax_gemm_np_mkn(lhs, rhs, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K)
    np.testing.assert_allclose(
        actual=online_np, desired=golden, atol=1e-5, rtol=1e-5, err_msg="online_np", verbose=True
    )
    np.testing.assert_allclose(
        actual=online_np_mkn, desired=golden, atol=1e-5, rtol=1e-5, err_msg="online_np_mkn", verbose=True
    )
