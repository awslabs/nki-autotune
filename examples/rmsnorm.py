# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np

from src.autotune_kernel import Autotune
from src.benchmark import test_kernel
from src.weighted_rmsnorm import allocated_weighted_rmsnorm


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
    gamma = nt.tensor[[dim], nl.bfloat16]
    eps = 1e-6

    tuner = Autotune(
        allocated_weighted_rmsnorm,
        configs=configs,
        warmup=10,
        iters=100,
        max_workers=1,
        cache_dir="private",
    )
    tuner(hidden=hidden, gamma=gamma, eps=eps)

    p99 = test_kernel(
        allocated_weighted_rmsnorm, [hidden, gamma, 1, eps], warmup=10, iters=100
    )
    print(f"allocated_weighted_rmsnorm p99 = {p99} us.")
