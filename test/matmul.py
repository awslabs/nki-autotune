# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
from itertools import product
import random

from src.autotune_kernel import Autotune
from src.kernels import nki_matmul_fully_optimized_


def get_autotune_configs():
    """
    Define a list of configuration dictionaries representing the specific design choices for autotuning.

    Returns:
        list: A list of dictionaries, each containing configuration parameters for TILES_IN_BLOCK_M,
                TILES_IN_BLOCK_N, and TILES_IN_BLOCK_K.
    """
    TILES_IN_BLOCK_M_options = [4]
    TILES_IN_BLOCK_N_options = [2, 4]
    TILES_IN_BLOCK_K_options = [2, 4]
    params = list(
        product(
            TILES_IN_BLOCK_M_options,
            TILES_IN_BLOCK_N_options,
            TILES_IN_BLOCK_K_options,
        )
    )
    configs = []
    for TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K in params:
        config = {
            "TILES_IN_BLOCK_M": TILES_IN_BLOCK_M,
            "TILES_IN_BLOCK_N": TILES_IN_BLOCK_N,
            "TILES_IN_BLOCK_K": TILES_IN_BLOCK_K,
        }
        configs.append(config)
    random.shuffle(configs)
    return configs


if __name__ == "__main__":
    lhsT = nt.tensor[[8192, 4096], nl.bfloat16]
    rhs = nt.tensor[[8192, 8192], nl.bfloat16]

    tuner = Autotune(
        nki_matmul_fully_optimized_,
        configs=get_autotune_configs(),
        warmup=2,
        iters=10,
        max_workers=2,
        show_compiler_tb=True,
    )
    tuner(lhsT, rhs)
