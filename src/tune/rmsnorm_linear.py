# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import random
from itertools import permutations, product
from typing import Dict, List

import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt

from src.kernels.matmul import MatMulCompatibility
from src.kernels.rmsnorm_linear import blocked_fused_rms_norm_linear
from src.tune.autotune_kernel import Autotune


def get_autotune_configs() -> List[Dict]:
    """
    Define a list of configuration dictionaries representing the specific design choices for autotuning.

    Returns:
        List[Dict]: A list of dictionaries, each containing configuration parameters for
        NUM_BLOCK_M, NUM_BLOCK_N, BUFFER_M, BUFFER_N
    """
    NUM_BLOCK_M_options = [1, 2, 4, 8]
    NUM_BLOCK_N_options = [1, 2, 4, 8]
    BUFFER_M_options = [1, 2, 4, 8]
    BUFFER_N_options = [1, 2, 4, 8]
    params = list(product(NUM_BLOCK_M_options, NUM_BLOCK_N_options, BUFFER_M_options, BUFFER_N_options))
    configs = []
    for NUM_BLOCK_M, NUM_BLOCK_N, BUFFER_M, BUFFER_N in params:
        config = {"NUM_BLOCK_M": NUM_BLOCK_M, "NUM_BLOCK_N": NUM_BLOCK_N, "BUFFER_M": BUFFER_M, "BUFFER_N": BUFFER_N}
        configs.append(config)
    random.shuffle(configs)
    return configs


def profile():
    batch = 1
    M = 8192
    N = 512
    K = 4096
    dtype = nl.bfloat16
    lhs = nt.tensor[[batch, M, K], dtype]
    rhs = nt.tensor[[K, N], dtype]
    tuner = Autotune(
        kernel=blocked_fused_rms_norm_linear,
        kernel_args=(lhs, rhs),
        configs=get_autotune_configs(),
        max_configs=127,
        warmup=10,
        iters=100,
        pruning_func=MatMulCompatibility,
        trace=True,
    )
    tuner()


if __name__ == "__main__":
    os.environ["NEURON_CC_FLAGS"] = "--framework=XLA --target=trn1 --auto-cast=none"
    profile()
