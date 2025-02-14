# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
from itertools import product
import random

from src.autotune_kernel import Autotune
from src.matmul import (
    matmul_KMN,
    matmul_MKN,
    matmul_KNM,
    matmul_MNK,
    matmul_NKM,
    matmul_NMK,
)


def get_autotune_configs():
    """
    Define a list of configuration dictionaries representing the specific design choices for autotuning.

    Returns:
        list: A list of dictionaries, each containing configuration parameters for TILES_IN_BLOCK_M,
                TILES_IN_BLOCK_N, and TILES_IN_BLOCK_K.
    """
    kernels = [matmul_KMN, matmul_KNM, matmul_MNK, matmul_NKM, matmul_NMK]
    TILES_IN_BLOCK_M_options = [1, 4, 8, 16]
    TILES_IN_BLOCK_N_options = [1, 4, 8, 16]
    TILES_IN_BLOCK_K_options = [1, 4, 8, 16]
    params = list(
        product(
            kernels,
            TILES_IN_BLOCK_M_options,
            TILES_IN_BLOCK_N_options,
            TILES_IN_BLOCK_K_options,
        )
    )
    configs = []
    for kernel, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K in params:
        config = {
            "kernel": kernel,
            "TILES_IN_BLOCK_M": TILES_IN_BLOCK_M,
            "TILES_IN_BLOCK_N": TILES_IN_BLOCK_N,
            "TILES_IN_BLOCK_K": TILES_IN_BLOCK_K,
        }
        configs.append(config)
    random.shuffle(configs)
    return configs


if __name__ == "__main__":
    for free_dim in [1024, 2048]:
        M = free_dim
        N = free_dim
        for K in [128, 256, 512, 1024]:
            lhsT = nt.tensor[[K, M], nl.float32]
            rhs = nt.tensor[[K, N], nl.float32]
            tuner = Autotune(
                configs=get_autotune_configs(),
                warmup=2,
                iters=10,
                max_workers=2,
                cache_dir=f"private/matmul-M{M}-N{N}-K{K}",
            )
            tuner(lhsT, rhs)
