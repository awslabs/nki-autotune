# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import random
import shutil
import time
from itertools import permutations, product
from typing import Dict, List

import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt

from src.cache.directories import TUNED_NKI_CACHE_DIR
from src.kernels.matmul import MatMulCompatibility, matmul_main
from src.tune.autotune_kernel import Autotune


def get_autotune_configs() -> List[Dict]:
    """
    Define a list of configuration dictionaries representing the specific design choices for autotuning.

    Returns:
        list: A list of dictionaries, each containing configuration parameters for NUM_BLOCK_M,
                NUM_BLOCK_N, and NUM_BLOCK_K.
    """
    NUM_BLOCK_M_options = [1, 2, 4, 8, 16, 32, 64]
    NUM_BLOCK_N_options = [1, 2, 4, 8, 16, 32, 64]
    NUM_BLOCK_K_options = [1, 2, 4, 8, 16, 32, 64]
    BUFFER_M_options = [1, 2, 4, 8]
    BUFFER_N_options = [1, 2, 4, 8]
    BUFFER_K_options = [1, 2, 4, 8]
    loop_orders = ["".join(p) for p in permutations("MNK")]
    params = list(
        product(
            NUM_BLOCK_M_options,
            NUM_BLOCK_N_options,
            NUM_BLOCK_K_options,
            BUFFER_M_options,
            BUFFER_N_options,
            BUFFER_K_options,
            loop_orders,
        )
    )
    configs = []
    for NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K, loop_order in params:
        config = {
            "NUM_BLOCK_M": NUM_BLOCK_M,
            "NUM_BLOCK_N": NUM_BLOCK_N,
            "NUM_BLOCK_K": NUM_BLOCK_K,
            "BUFFER_M": BUFFER_M,
            "BUFFER_N": BUFFER_N,
            "BUFFER_K": BUFFER_K,
            "loop_order": loop_order,
        }
        configs.append(config)
    random.shuffle(configs)
    return configs


def profile():
    cache_root = TUNED_NKI_CACHE_DIR
    if os.path.exists(cache_root):
        shutil.rmtree(cache_root)
    os.makedirs(cache_root)
    shapes = [4096, 8192]
    dtype = nl.bfloat16
    MNK = list(product(shapes, shapes, shapes))
    start = time.perf_counter()
    for M, N, K in MNK:
        lhsT = nt.tensor[[K, M], dtype]
        rhs = nt.tensor[[K, N], dtype]
        tuner = Autotune(
            kernel=matmul_main,
            kernel_args=(lhsT, rhs),
            configs=get_autotune_configs(),
            max_configs=127,
            pruning_func=MatMulCompatibility,
            trace=True,
        )
        tuner()
        break
    latency = time.perf_counter() - start
    print(f"latency = {latency}")


if __name__ == "__main__":
    os.environ["NEURON_CC_FLAGS"] = "--framework=XLA --target=trn1 --auto-cast=none"
    profile()
