# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
from itertools import product
import torch
import numpy as np

from src.autotune_kernel import Autotune
from src.allocated_kernels import allocated_fused_rms_norm_qkv
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


def rms_norm(hidden, gamma, eps=1e-6):
    rms = np.sqrt(np.mean(np.square(hidden), axis=-1, keepdims=True))
    norm = hidden * np.reciprocal(rms + eps)
    if gamma is not None:
        norm *= gamma
    return norm


def cpu_golden_result(hidden, gamma, qkv_weights, dtype, do_norm=True):
    if do_norm:
        hidden = rms_norm(hidden, gamma)
    qkv_out = (hidden @ qkv_weights).astype(dtype)
    return qkv_out


def verify(batch, seqlen, dim, d_head, hidden_buffer_degree):
    dtype = np.float16
    atol = 1e-2
    rtol = 1e-3

    hidden = np.random.random_sample((batch, seqlen, dim))
    weights = np.random.random_sample((dim, d_head))
    golden_res = nl.static_cast(
        cpu_golden_result(hidden, None, weights, dtype, do_norm=True), np.float32
    )

    hidden_dev = nl.static_cast(hidden, dtype)
    weights_dev = nl.static_cast(weights, dtype)

    numeric_func = baremetal(allocated_fused_rms_norm_qkv)
    allocated_out = numeric_func(hidden_dev, weights_dev, hidden_buffer_degree)
    allocated_out = nl.static_cast(allocated_out, np.float32)
    match = allclose(allocated_out, golden_res, atol=atol, rtol=rtol, verbose=1)
    print(
        f"Allocated kernel match for hidden_buffer_degree {hidden_buffer_degree}: {match}"
    )


if __name__ == "__main__":
    batch, seqlen, dim, d_head = 1, 4096, 2048, 256
    configs = get_autotune_configs()
    for config in configs:
        print(config)
        verify(batch, seqlen, dim, d_head, config["hidden_buffer_degree"])

    hidden = nt.tensor[[batch, seqlen, dim], nl.bfloat16]
    weights = nt.tensor[[dim, d_head], nl.bfloat16]
    tuner = Autotune(
        allocated_fused_rms_norm_qkv,
        configs=get_autotune_configs(),
        warmup=2,
        iters=10,
        max_workers=1,
        show_compiler_tb=True,
    )
    tuner(hidden, weights)
