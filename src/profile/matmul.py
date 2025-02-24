# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import random, os, shutil
from typing import List, Dict
from itertools import product, permutations

from src.autotune_kernel import Autotune
from src.matmul import matmul_main, MatMulCompatibility


def get_autotune_configs() -> List[Dict]:
    """
    Define a list of configuration dictionaries representing the specific design choices for autotuning.

    Returns:
        list: A list of dictionaries, each containing configuration parameters for TILES_IN_BLOCK_M,
                TILES_IN_BLOCK_N, and TILES_IN_BLOCK_K.
    """
    TILES_IN_BLOCK_M_options = [1, 2, 4, 8, 16, 32, 64]
    TILES_IN_BLOCK_N_options = [1, 2, 4, 8, 16, 32, 64]
    TILES_IN_BLOCK_K_options = [1, 2, 4, 8, 16, 32, 64]
    BUFFER_M_options = [1, 2, 4, 8]
    BUFFER_N_options = [1, 2, 4, 8]
    BUFFER_K_options = [1, 2, 4, 8]
    loop_orders = ["".join(p) for p in permutations("MNK")]
    params = list(
        product(
            TILES_IN_BLOCK_M_options,
            TILES_IN_BLOCK_N_options,
            TILES_IN_BLOCK_K_options,
            BUFFER_M_options,
            BUFFER_N_options,
            BUFFER_K_options,
            loop_orders,
        )
    )
    configs = []
    for TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K, loop_order in params:
        config = {
            "TILES_IN_BLOCK_M": TILES_IN_BLOCK_M,
            "TILES_IN_BLOCK_N": TILES_IN_BLOCK_N,
            "TILES_IN_BLOCK_K": TILES_IN_BLOCK_K,
            "BUFFER_M": BUFFER_M,
            "BUFFER_N": BUFFER_N,
            "BUFFER_K": BUFFER_K,
            "loop_order": loop_order,
        }
        configs.append(config)
    random.shuffle(configs)
    return configs


def profile():
    cache_root = "/home/ubuntu/matmul-cache"
    if os.path.exists(cache_root):
        shutil.rmtree(cache_root)
    os.makedirs(cache_root)
    shapes = [4096, 8192]
    MNK = list(product(shapes, shapes, shapes))
    for M, N, K in MNK:
        lhsT = nt.tensor[[K, M], nl.float32]
        rhs = nt.tensor[[K, N], nl.float32]
        tuner = Autotune(
            kernel=matmul_main,
            max_configs=100,
            warmup=10,
            iters=100,
            pruning_func=MatMulCompatibility,
            cache_dir=f"{cache_root}/M{M}-N{N}-K{K}",
        )
        tuner(get_autotune_configs(), lhsT, rhs)


if __name__ == "__main__":
    profile()
