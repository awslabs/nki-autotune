# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np

from src.autotune_kernel import Autotune
from src.benchmark import profile_kernel
from src.kernels.fused_rmsnorm_linear import stack_allocated_fused_rms_norm_qkv, allocated_fused_rms_norm_qkv


def get_autotune_configs():
    configs = [
        {"hidden_buffer_degree": 1},
        {"hidden_buffer_degree": 2},
        {"hidden_buffer_degree": 4},
        {"hidden_buffer_degree": 8},
    ]
    return configs


if __name__ == "__main__":
    batch, seqlen, dim, d_head = 1, 2048, 4096, 512
    configs = get_autotune_configs()
    hidden = nt.tensor[[batch, seqlen, dim], nl.bfloat16]
    weights = nt.tensor[[dim, d_head], nl.bfloat16]
    z_tensor = nt.tensor[[batch, seqlen, dim], nl.float16]
    gamma = nt.tensor[[dim], nl.bfloat16]
    n_groups = 1
    eps = 1e-6

    p99 = profile_kernel(stack_allocated_fused_rms_norm_qkv, [hidden, weights, nl.float32, eps], warmup=10, iters=100)
    print(f"stack_allocated_fused_rms_norm_qkv p99 = {p99} us.")
