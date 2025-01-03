# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
from itertools import product
import random
import numpy as np

from src.autotune_kernel import AutotuneKernel
from src.kernels import stack_allocated_fused_rms_norm_qkv, allocated_fused_rms_norm_qkv


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
    batch = 1
    seqlen = 1024
    dim = 512
    d_head = 16

    hidden = np.random.rand(batch, seqlen, dim).astype(np.float32)
    weights = np.random.rand(dim, d_head).astype(np.float32)

    stack_allocated_output_nki = stack_allocated_fused_rms_norm_qkv(hidden, weights)
    print(
        f"stack_allocated_output_nki = {stack_allocated_output_nki} {stack_allocated_output_nki.shape}"
    )

    allocated_output_nki = allocated_fused_rms_norm_qkv(hidden, weights)
    print(f"allocated_output_nki = {allocated_output_nki} {allocated_output_nki.shape}")

    # numpy RMSNorm
    s_hidden = hidden * hidden
    ms_hidden = np.mean(s_hidden, axis=-1, keepdims=True)
    rms_hidden = np.sqrt(ms_hidden)
    rms_norm_hidden = hidden / rms_hidden
    output_np = np.matmul(rms_norm_hidden, weights)
    print(f"output_np = {output_np} {output_np.shape}")

    for output_nki in [stack_allocated_output_nki, allocated_output_nki]:
        allclose = np.allclose(output_np, output_nki, atol=1e-4, rtol=1e-2)
        if allclose:
            print("NKI and NumPy match")
        assert allclose
