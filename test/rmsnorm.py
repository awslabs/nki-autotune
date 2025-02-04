# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np

from src.autotune_kernel import Autotune
from src.benchmark import test_kernel
from src.allocated_kernels import allocated_fused_rms_norm_qkv, allocated_rms_norm
from src.kernels import nki_rmsnorm_kernel
from neuronxcc.starfish.support.util import allclose
from neuronxcc.nki import baremetal


def get_autotune_configs():
    configs = [
        {"hidden_buffer_degree": 1},
        {"hidden_buffer_degree": 2},
        {"hidden_buffer_degree": 4},
        {"hidden_buffer_degree": 8},
    ]
    return configs


def cpu_golden_result(hidden, dtype, gamma=None, qkv_weights=None, eps=1e-6):
    rms = np.sqrt(np.mean(np.square(hidden), axis=-1, keepdims=True))
    output = hidden * np.reciprocal(rms + eps)
    if gamma is not None:
        output *= gamma
    if qkv_weights is not None:
        output = output @ qkv_weights
    output = output.astype(dtype)
    return output


def verify(batch, seqlen, dim, d_head, hidden_buffer_degree):
    dtype = np.float16
    atol = 1e-2
    rtol = 1e-3

    hidden = np.random.random_sample((batch, seqlen, dim))
    gamma = np.random.random_sample((dim))
    qkv_weights = np.random.random_sample((dim, d_head))
    hidden_dev = nl.static_cast(hidden, dtype)
    gamma_dev = nl.static_cast(gamma, dtype)
    qkv_weights_dev = nl.static_cast(qkv_weights, dtype)

    golden_res = nl.static_cast(
        cpu_golden_result(hidden, dtype, qkv_weights=qkv_weights), np.float32
    )
    numeric_func = baremetal(allocated_fused_rms_norm_qkv)
    allocated_out = numeric_func(hidden_dev, qkv_weights_dev, hidden_buffer_degree)
    allocated_out = nl.static_cast(allocated_out, np.float32)
    match = allclose(allocated_out, golden_res, atol=atol, rtol=rtol, verbose=1)
    assert (
        match
    ), f"allocated_fused_rms_norm_qkv match for hidden_buffer_degree {hidden_buffer_degree}: {match}"

    golden_res = nl.static_cast(cpu_golden_result(hidden, dtype), np.float32)
    numeric_func = baremetal(allocated_rms_norm)
    allocated_out = numeric_func(hidden_dev, hidden_buffer_degree)
    allocated_out = nl.static_cast(allocated_out, np.float32)
    match = allclose(allocated_out, golden_res, atol=atol, rtol=rtol, verbose=1)
    assert (
        match
    ), f"allocated_rms_norm match for hidden_buffer_degree {hidden_buffer_degree}: {match}"

    golden_res = nl.static_cast(cpu_golden_result(hidden, dtype), np.float32)
    numeric_func = baremetal(nki_rmsnorm_kernel)
    nki_out = numeric_func(hidden_dev)
    nki_out = nl.static_cast(nki_out, np.float32)
    match = allclose(nki_out, golden_res, atol=atol, rtol=rtol, verbose=1)
    assert match, f"nki_rmsnorm_kernel match: {match} {golden_res.shape}"


if __name__ == "__main__":
    batch, seqlen, dim, d_head = 1, 2048, 4096, 512
    configs = get_autotune_configs()
    for config in configs:
        verify(batch, seqlen, dim, d_head, config["hidden_buffer_degree"])
    hidden = nt.tensor[[batch, seqlen, dim], nl.bfloat16]
    weights = nt.tensor[[dim, d_head], nl.bfloat16]

    # tuner = Autotune(
    #     allocated_fused_rms_norm_qkv,
    #     configs=configs,
    #     warmup=10,
    #     iters=100,
    #     max_workers=1,
    #     cache_dir="private",
    # )
    # tuner(hidden, weights)

    tuner = Autotune(
        allocated_rms_norm,
        configs=configs,
        warmup=10,
        iters=100,
        max_workers=1,
        cache_dir="private",
    )
    tuner(hidden)

    p99 = test_kernel(nki_rmsnorm_kernel, [hidden], warmup=10, iters=100)
    print(f"nki_rmsnorm_kernel p99 = {p99} us.")
