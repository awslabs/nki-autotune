# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
from itertools import product
import torch
import numpy as np

from src.autotune_kernel import AutotuneKernel
from src.allocated_kernels import allocated_fused_rms_norm_qkv
from neuronxcc.starfish.support.util import allclose
from neuronxcc.nki import benchmark, baremetal, simulate_kernel


def get_autotune_configs():
    configs = [
        {"multi_buffer": 1},
        {"multi_buffer": 2},
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


def verify():
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
    allocated_out = numeric_func(hidden_dev, weights_dev)
    allocated_out = nl.static_cast(allocated_out, np.float32)
    match = allclose(allocated_out, golden_res, atol=atol, rtol=rtol, verbose=1)
    print(f"Allocated kernel match: {match}")


if __name__ == "__main__":
    batch, seqlen, dim, d_head = 1, 1024, 2048, 64

    hidden = nt.tensor[[batch, seqlen, dim], nl.bfloat16]
    weights = nt.tensor[[dim, d_head], nl.bfloat16]
    tuner = AutotuneKernel.trace(
        allocated_fused_rms_norm_qkv,
        iters=10,
        configs=get_autotune_configs(),
        max_workers=1,
        show_compiler_tb=True,
    )
    tuner(hidden, weights)
