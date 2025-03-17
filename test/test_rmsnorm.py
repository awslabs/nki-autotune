import pytest
import numpy as np

from src.kernels.rmsnorm_weighted import weighted_rmsnorm, allocated_weighted_rmsnorm
from src.kernels.rmsnorm_linear import (
    allocated_fused_rms_norm_qkv,
    stack_allocated_fused_rms_norm_qkv,
    blocked_fused_rms_norm_linear,
)
from src.benchmark import profile_kernel
from src.golden.rmsnorm_linear import rmsnorm_linear_golden

import neuronxcc.nki.language as nl
from neuronxcc.starfish.support.util import allclose
from neuronxcc.nki import baremetal
import neuronxcc.nki.typing as nt


@pytest.mark.parametrize(
    "batch, seqlen, dim, eps", [(1, 1024, 4096, 1e-6), (1, 2048, 1024, 1e-3), (1, 4096, 2048, 1e-6)]
)
def test_weighted_rmsnorm(batch, seqlen, dim, eps):
    hidden = np.random.random_sample((batch, seqlen, dim))
    gamma = np.random.random_sample((dim))
    golden_output = nl.static_cast(rmsnorm_linear_golden(hidden, None, gamma, None, eps), np.float32)

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
    golden_output = nl.static_cast(rmsnorm_linear_golden(hidden, None, gamma, None, eps), np.float32)

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
    golden_output = nl.static_cast(rmsnorm_linear_golden(hidden, None, None, qkv_weights, eps), np.float32)

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


@pytest.mark.parametrize(
    "batch, seqlen, dim, d_head, buffer_degree, eps",
    [(1, 1024, 4096, 256, 1, 1e-6), (1, 2048, 1024, 512, 2, 1e-3), (1, 4096, 2048, 128, 4, 1e-6)],
)
def test_stack_allocated_fused_rms_norm_qkv(batch, seqlen, dim, d_head, buffer_degree, eps):
    hidden = np.random.random_sample((batch, seqlen, dim))
    qkv_weights = np.random.random_sample((dim, d_head))
    golden_output = nl.static_cast(rmsnorm_linear_golden(hidden, None, None, qkv_weights, eps), np.float32)

    data_type = np.float16
    hidden_dev = nl.static_cast(hidden, data_type)
    qkv_weights_dev = nl.static_cast(qkv_weights, data_type)
    numeric_func = baremetal(stack_allocated_fused_rms_norm_qkv)
    allocated_out = numeric_func(hidden_dev, qkv_weights_dev, nl.float32, eps)
    allocated_out = nl.static_cast(allocated_out, np.float32)

    atol = 1e-2
    rtol = 1e-3
    match = allclose(allocated_out, golden_output, atol=atol, rtol=rtol, verbose=1)
    assert match


@pytest.mark.parametrize(
    "batch, M, K, N, eps", [(1, 1024, 4096, 512, 1e-6), (1, 2048, 1024, 512, 1e-3), (1, 4096, 2048, 512, 1e-6)]
)
def test_blocked_fused_rms_norm_linear_numerical(batch, M, K, N, eps):
    data_type = np.float32
    atol, rtol = 1e-2, 1e-3
    lhs = np.random.random_sample((batch, M, K)).astype(data_type)
    rhs = np.random.random_sample((K, N)).astype(data_type)

    golden = nl.static_cast(rmsnorm_linear_golden(lhs, None, None, rhs, eps), data_type)

    numeric_func = baremetal(blocked_fused_rms_norm_linear)
    # TODO: add tests for num blocks and buffer sizes
    nki_out = nl.static_cast(numeric_func(lhs, rhs, 2, 1, 2, 1, 1, 1, nl.float32, eps), data_type)

    assert allclose(nki_out, golden, atol=atol, rtol=rtol, verbose=1)


@pytest.mark.xfail(reason="Optimized kernel not yet done")
@pytest.mark.parametrize("batch, seqlen, dim, d_head, eps", [(1, 1024, 4096, 256, 1e-6), (1, 2048, 1024, 512, 1e-3)])
def test_blocked_fused_rms_norm_linear_perf(batch, seqlen, dim, d_head, eps):
    dtype = nl.bfloat16
    hidden = nt.tensor[[batch, seqlen, dim], dtype]
    qkv_weights = nt.tensor[[dim, d_head], dtype]
    warmup = 10
    iters = 100
    baseline_p99 = profile_kernel(
        stack_allocated_fused_rms_norm_qkv, (hidden, qkv_weights, nl.float32, eps), warmup=warmup, iters=iters
    )
    optimized_p99 = profile_kernel(
        blocked_fused_rms_norm_linear, (hidden, qkv_weights, nl.float32, eps), warmup=warmup, iters=iters
    )
    print(f"Optimized version {optimized_p99}ms. stack_allocated_fused_rms_norm_qkv {baseline_p99}ms.")
    assert (
        optimized_p99 <= baseline_p99
    ), f"Optimized version {optimized_p99}ms should be faster than stack_allocated_fused_rms_norm_qkv {baseline_p99}ms"
