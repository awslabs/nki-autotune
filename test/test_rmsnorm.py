from typing import Dict

import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np
import pytest
from neuronxcc.nki import baremetal
from neuronxcc.starfish.support.util import allclose

from autotune.core.test_generation import GenTests
from autotune.core.utils import GEMMCompatibility
from autotune.golden.rmsnorm_linear import fused_rmsnorm_gemm_golden, golden_fun, rmsnorm_gemm_golden
from kernel_library.rmsnorm_linear import (
    allocated_fused_rms_norm_qkv,
    blocked_fused_rms_norm_linear,
    stack_allocated_fused_rms_norm_qkv,
)
from kernel_library.rmsnorm_weighted import allocated_weighted_rmsnorm, weighted_rmsnorm


class RMSNormLinearTestConfig(GenTests):
    def process_test_config(self, config: Dict[str, int]) -> bool:
        M = config.get("M", 1)
        N = config.get("N", 1)
        K = config.get("K", 1)
        try:
            assert max(M, N, K) <= 8192, f"Input sizes are too large for testing"
            check = GEMMCompatibility(transposed_lhs=False)
            check((M, K), (K, N), **config)
            return True
        except Exception as e:
            return False


@pytest.mark.parametrize(
    "batch, M, N, K, eps",
    RMSNormLinearTestConfig(
        batch=[1, 2, 4], M=[1024, 2048, 4096], N=[1024, 2048, 4096], K=[1024, 2048, 4096], eps=[1e-6, 1e-3]
    ).sample_tests(10),
)
def test_rmsnorm_gemm_np_numerical(batch: int, M: int, N: int, K: int, eps: float):
    data_type = np.float32
    atol, rtol = 1e-2, 1e-3
    lhs = np.random.random_sample((batch, M, K)).astype(data_type)
    rhs = np.random.random_sample((K, N)).astype(data_type)

    golden = golden_fun(lhs, None, None, rhs, eps)
    np_out = rmsnorm_gemm_golden(lhs, rhs, eps)
    fused_np_out = fused_rmsnorm_gemm_golden(lhs, rhs, eps)
    assert allclose(
        np_out, golden, atol=atol, rtol=rtol, verbose=1
    ), f"{rmsnorm_gemm_golden} output does not match with golden."
    assert allclose(
        fused_np_out, golden, atol=atol, rtol=rtol, verbose=1
    ), f"{fused_rmsnorm_gemm_golden} output does not match with golden."


@pytest.mark.parametrize(
    "batch, seqlen, dim, eps", [(1, 1024, 4096, 1e-6), (1, 2048, 1024, 1e-3), (1, 4096, 2048, 1e-6)]
)
def test_weighted_rmsnorm(batch, seqlen, dim, eps):
    hidden = np.random.random_sample((batch, seqlen, dim))
    gamma = np.random.random_sample((dim))
    golden_output = nl.static_cast(golden_fun(hidden, None, gamma, None, eps), np.float32)

    data_type = np.float16
    hidden_dev = nl.static_cast(hidden, data_type)
    gamma_dev = nl.static_cast(gamma, data_type)
    numeric_func = baremetal(weighted_rmsnorm)
    nki_out = numeric_func(hidden_dev, gamma_dev, eps)
    nki_out = nl.static_cast(nki_out, np.float32)

    atol = 1e-2
    rtol = 1e-3
    match = allclose(nki_out, golden_output, atol=atol, rtol=rtol, verbose=1)
    assert match


@pytest.mark.parametrize(
    "batch, seqlen, dim, buffer_degree, eps",
    [(1, 1024, 4096, 1, 1e-6), (1, 2048, 1024, 2, 1e-3), (1, 4096, 2048, 4, 1e-6)],
)
def test_allocated_weighted_rmsnorm(batch, seqlen, dim, buffer_degree, eps):
    hidden = np.random.random_sample((batch, seqlen, dim))
    gamma = np.random.random_sample((dim))
    golden_output = nl.static_cast(golden_fun(hidden, None, gamma, None, eps), np.float32)

    data_type = np.float16
    hidden_dev = nl.static_cast(hidden, data_type)
    gamma_dev = nl.static_cast(gamma, data_type)
    numeric_func = baremetal(allocated_weighted_rmsnorm)
    allocated_out = numeric_func(hidden_dev, gamma_dev, buffer_degree, eps)
    allocated_out = nl.static_cast(allocated_out, np.float32)

    atol = 1e-2
    rtol = 1e-3
    match = allclose(allocated_out, golden_output, atol=atol, rtol=rtol, verbose=1)
    assert match


@pytest.mark.parametrize(
    "batch, seqlen, dim, d_head, buffer_degree, eps",
    [(1, 1024, 4096, 256, 1, 1e-6), (1, 2048, 1024, 512, 2, 1e-3), (1, 4096, 2048, 128, 4, 1e-6)],
)
def test_allocated_fused_rms_norm_qkv(batch, seqlen, dim, d_head, buffer_degree, eps):
    hidden = np.random.random_sample((batch, seqlen, dim))
    qkv_weights = np.random.random_sample((dim, d_head))
    golden_output = nl.static_cast(golden_fun(hidden, None, None, qkv_weights, eps), np.float32)

    data_type = np.float16
    hidden_dev = nl.static_cast(hidden, data_type)
    qkv_weights_dev = nl.static_cast(qkv_weights, data_type)
    numeric_func = baremetal(allocated_fused_rms_norm_qkv)
    allocated_out = numeric_func(hidden_dev, qkv_weights_dev, buffer_degree, eps)
    allocated_out = nl.static_cast(allocated_out, np.float32)

    atol = 1e-2
    rtol = 1e-3
    match = allclose(allocated_out, golden_output, atol=atol, rtol=rtol, verbose=1)
    assert match


# @pytest.mark.parametrize(
#     "batch, seqlen, dim, d_head, eps",
#     [(1, 1024, 4096, 256, 1e-6), (1, 2048, 1024, 512, 1e-3), (1, 4096, 2048, 128, 1e-6)],
# )
@pytest.mark.parametrize("batch, seqlen, dim, d_head, eps", [(1, 1024, 4096, 256, 1e-6)])
def test_stack_allocated_fused_rms_norm_qkv(batch, seqlen, dim, d_head, eps):
    lhs = np.random.random_sample((batch, seqlen, dim))
    rhs = np.random.random_sample((dim, d_head))
    golden_output = nl.static_cast(golden_fun(lhs, None, None, rhs, eps), np.float32)

    data_type = np.float16
    atol, rtol = 1e-2, 1e-3
    lhs = nl.static_cast(lhs, data_type)
    rhs = nl.static_cast(rhs, data_type)

    # nki_out, _ = run_kernel("stack_allocated_fused_rms_norm_qkv", (lhs, rhs), eps=eps)
    # nki_out = nl.static_cast(nki_out, np.float32)

    numeric_func = baremetal(stack_allocated_fused_rms_norm_qkv)
    nki_out = numeric_func(lhs, rhs, nl.float32, eps)

    match = allclose(nki_out, golden_output, atol=atol, rtol=rtol, verbose=1)
    assert match


@pytest.mark.parametrize(
    "batch, M, N, K, NUM_BLOCK_M, NUM_BLOCK_N, BUFFER_M, BUFFER_N, eps",
    RMSNormLinearTestConfig(
        batch=[1],
        M=[1024],
        N=[1024],
        K=[1024],
        NUM_BLOCK_M=[1],
        NUM_BLOCK_N=[1],
        BUFFER_M=[1],
        BUFFER_N=[1],
        eps=[1e-6, 1e-3],
    ).sample_tests(1),
)
def test_blocked_fused_rms_norm_linear_numerical(batch, M, N, K, NUM_BLOCK_M, NUM_BLOCK_N, BUFFER_M, BUFFER_N, eps):
    data_type = np.float32
    atol, rtol = 1e-2, 1e-3
    lhs = np.random.random_sample((batch, M, K)).astype(data_type)
    rhs = np.random.random_sample((K, N)).astype(data_type)

    golden = nl.static_cast(golden_fun(lhs, None, None, rhs, eps), data_type)

    # nki_out, _ = run_kernel(
    #     "blocked_fused_rms_norm_linear",
    #     (lhs, rhs),
    #     NUM_BLOCK_M=NUM_BLOCK_M,
    #     NUM_BLOCK_N=NUM_BLOCK_N,
    #     BUFFER_M=BUFFER_M,
    #     BUFFER_N=BUFFER_N,
    #     eps=eps,
    # )

    numeric_func = baremetal(blocked_fused_rms_norm_linear)
    nki_out = numeric_func(lhs, rhs, NUM_BLOCK_M, NUM_BLOCK_N, BUFFER_M, BUFFER_N, nl.float32, eps)

    nki_out = nl.static_cast(nki_out, data_type)

    assert allclose(nki_out, golden, atol=atol, rtol=rtol, verbose=1)


@pytest.mark.xfail(reason="Need to implement standalone kernel profile")
@pytest.mark.parametrize("batch, M, K, N, eps", [(1, 8192, 4096, 512, 1e-6)])
def test_blocked_fused_rms_norm_linear_perf(batch, M, K, N, eps):
    dtype = nl.bfloat16
    hidden = nt.tensor[[batch, M, K], dtype]
    qkv_weights = nt.tensor[[K, N], dtype]
    baseline_p99, _ = profile_kernel(stack_allocated_fused_rms_norm_qkv, (hidden, qkv_weights, nl.float32, eps))
    optimized_p99, _ = profile_kernel(blocked_fused_rms_norm_linear, (hidden, qkv_weights, 8, 1, 4, 1, nl.float32, eps))
    print(f"blocked_fused_rms_norm_linear {optimized_p99}ms. stack_allocated_fused_rms_norm_qkv {baseline_p99}ms.")
    assert (
        optimized_p99 < baseline_p99
    ), f"blocked_fused_rms_norm_linear {optimized_p99}ms should be faster than stack_allocated_fused_rms_norm_qkv {baseline_p99}ms"
