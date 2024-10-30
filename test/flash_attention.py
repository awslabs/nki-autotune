# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np
from itertools import product
import random

from neuronxcc.nki.kernels import flash_fwd
from src.autotune_kernel import AutotuneKernel


def get_autotune_configs():
    """
    Define a list of configuration dictionaries representing the specific design choices for autotuning.

    Returns:
        list: A list of dictionaries, each containing configuration parameters for TILES_IN_BLOCK_M,
                TILES_IN_BLOCK_N, and TILES_IN_BLOCK_K.
    """
    space = [int(2**x) for x in range(1, 7)]
    TILES_IN_BLOCK_M_options = space
    TILES_IN_BLOCK_N_options = space
    TILES_IN_BLOCK_K_options = space
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
    bs = 1
    num_heads = 1
    head_dim = 128
    seq_len = 8192
    data = {}
    for parameter_name in ["Q", "K", "V"]:
        val = (np.random.random_sample([bs, num_heads, seq_len, head_dim]) - 0.5) * 2
        data[parameter_name] = val.astype(np.float32)
    print(flash_fwd)
    print(data)

    # tuner = AutotuneKernel.trace(
    #     nki_matmul_fully_optimized_,
    #     configs=get_autotune_configs(),
    #     show_compiler_tb=True,
    # )
    # tuner(lhsT, rhs, output)
