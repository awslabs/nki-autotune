# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki as nki
import numpy as np
from neuronxcc.nki import baremetal
from neuronxcc.starfish.support.util import allclose

from src.autotune_kernel import Autotune
from src.benchmark import profile_kernel
from src.kernels.rmsnorm_linear import (
    stack_allocated_fused_rms_norm_qkv,
    allocated_fused_rms_norm_qkv,
    optimized_fused_rms_norm_qkv,
)


def silu(x):
    return x / (1 + np.exp(-x))


def cpu_golden_result(hidden, gate, gamma, qkv_weights, eps):
    if gate is not None:
        hidden = hidden * silu(gate.astype(np.float32))
    rms = np.sqrt(np.mean(np.square(hidden), axis=-1, keepdims=True) + eps)
    output = hidden * np.reciprocal(rms)
    if gamma is not None:
        output *= gamma
    if qkv_weights is not None:
        output = output @ qkv_weights
    return output


if __name__ == "__main__":
    batch, seqlen, dim, d_head = 1, 2048, 4096, 512
    hidden = nt.tensor[[batch, seqlen, dim], nl.bfloat16]
    weights = nt.tensor[[dim, d_head], nl.bfloat16]
    z_tensor = nt.tensor[[batch, seqlen, dim], nl.float16]
    gamma = nt.tensor[[dim], nl.bfloat16]
    n_groups = 1
    eps = 1e-6

    # p99 = profile_kernel(
    #     stack_allocated_fused_rms_norm_qkv, (hidden, weights, nl.float32, eps), {}, warmup=10, iters=100
    # )
    # print(f"stack_allocated_fused_rms_norm_qkv p99 = {p99} us.")

    hidden = np.random.random_sample((batch, seqlen, dim))
    qkv_weights = np.random.random_sample((dim, d_head))
    data_type = np.float16
    hidden_dev = nl.static_cast(hidden, data_type)
    qkv_weights_dev = nl.static_cast(qkv_weights, data_type)
    numeric_func = baremetal(optimized_fused_rms_norm_qkv)
    nki_out = numeric_func(hidden_dev, qkv_weights_dev, 1, 1, 1, 1, 1, 1, nl.float32, eps)
    golden_output = nl.static_cast(cpu_golden_result(hidden, None, None, qkv_weights, eps), np.float32)
    atol = 1e-2
    rtol = 1e-3
    print(nki_out.shape, nki_out)
    print(golden_output.shape, golden_output)
    assert allclose(nki_out, golden_output, atol=atol, rtol=rtol, verbose=1)
