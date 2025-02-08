import pytest
import numpy as np

from src.weighted_rmsnorm import weighted_rmsnorm, allocated_weighted_rmsnorm
from src.fused_rmsnorm_linear import allocated_fused_rms_norm_qkv

import neuronxcc.nki.language as nl
from neuronxcc.starfish.support.util import allclose
from neuronxcc.nki import baremetal


def cpu_golden_result(hidden, gamma, qkv_weights, eps):
    rms = np.sqrt(np.mean(np.square(hidden), axis=-1, keepdims=True))
    output = hidden * np.reciprocal(rms + eps)
    if gamma is not None:
        output *= gamma
    if qkv_weights is not None:
        output = output @ qkv_weights
    return output


@pytest.mark.parametrize(
    "batch, seqlen, dim, eps",
    [
        (1, 1024, 4096, 1e-6),
        (1, 2048, 1024, 1e-3),
        (1, 4096, 2048, 1e-6),
    ],
)
def test_weighted_rmsnorm(batch, seqlen, dim, eps):
    hidden = np.random.random_sample((batch, seqlen, dim))
    gamma = np.random.random_sample((dim))
    golden_output = nl.static_cast(
        cpu_golden_result(hidden, gamma, None, eps), np.float32
    )

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
    [
        (1, 1024, 4096, 1, 1e-6),
        (1, 2048, 1024, 2, 1e-3),
        (1, 4096, 2048, 4, 1e-6),
    ],
)
def test_allocated_weighted_rmsnorm(batch, seqlen, dim, buffer_degree, eps):
    hidden = np.random.random_sample((batch, seqlen, dim))
    gamma = np.random.random_sample((dim))
    golden_output = nl.static_cast(
        cpu_golden_result(hidden, gamma, None, eps), np.float32
    )

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
    [
        (1, 1024, 4096, 256, 1, 1e-6),
        (1, 2048, 1024, 512, 2, 1e-3),
        (1, 4096, 2048, 128, 4, 1e-6),
    ],
)
def test_allocated_fused_rms_norm_qkv(batch, seqlen, dim, d_head, buffer_degree, eps):
    hidden = np.random.random_sample((batch, seqlen, dim))
    qkv_weights = np.random.random_sample((dim, d_head))
    golden_output = nl.static_cast(
        cpu_golden_result(hidden, None, qkv_weights, eps), np.float32
    )

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
