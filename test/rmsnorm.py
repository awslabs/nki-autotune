# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
from itertools import product
import random
import numpy as np
import sys

sys.path.append("../")
from src.autotune_kernel import AutotuneKernel
from src.kernels import nki_rms_norm_, stack_allocated_fused_rms_norm_qkv, allocated_fused_rms_norm_qkv


def get_autotune_configs():
    """
    Define a list of configuration dictionaries representing the specific design choices for autotuning.

    Returns:
        list: A list of dictionaries, each containing configuration parameters for TILES_IN_BLOCK_M,
                TILES_IN_BLOCK_N, and TILES_IN_BLOCK_K.
    """
    PARTITION_DIM_options = [int(2**x) for x in range(6, 9)]
    params = list(
        product(
            PARTITION_DIM_options,
        )
    )
    configs = []
    for PARTITION_DIM in PARTITION_DIM_options:
        config = {
            "PARTITION_DIM": PARTITION_DIM,
        }
        configs.append(config)
    random.shuffle(configs)
    return configs


if __name__ == "__main__":
    batch = 1
    seqlen = 1024
    dim = 512
    d_head = 16

    hidden = np.random.rand(batch, seqlen, dim).astype(np.float16)
    #weights = np.random.rand(dim, d_head).astype(np.float16)
    weights = np.random.rand(dim).astype(np.float16)
    eps = 1e-6

    # numpy RMSNorm
    s_hidden = hidden * hidden
    ms_hidden = np.mean(s_hidden, axis=-1, keepdims=True)
    rms_hidden = np.sqrt(ms_hidden)
    rms_norm_hidden = hidden / rms_hidden
    output_np = rms_norm_hidden * weights
    #output_np = np.matmul(rms_norm_hidden, weights)
    print(f"output_np = {output_np} {output_np.shape}")

    output = nki_rms_norm_(hidden, weights, eps, 128)
    print(
        f"rms_norm_output_nki = {output} {output.shape}"
    )

    '''
    stack_allocated_output_nki = stack_allocated_fused_rms_norm_qkv(hidden, weights)
    print(
        f"stack_allocated_output_nki = {stack_allocated_output_nki} {stack_allocated_output_nki.shape}"
    )

    allocated_output_nki = allocated_fused_rms_norm_qkv(hidden, weights)
    print(f"allocated_output_nki = {allocated_output_nki} {allocated_output_nki.shape}")
    '''

    #for output_nki in [stack_allocated_output_nki, allocated_output_nki]:
    for output_nki in [output]:
        allclose = np.allclose(output_np, output_nki, atol=1e-4, rtol=1e-2)
        if allclose:
            print("NKI and NumPy match")
        assert allclose
    
    tuner = AutotuneKernel.trace(
        nki_rms_norm_,
        iters=10,
        configs=get_autotune_configs(),
        show_compiler_tb=True,
    )
    tuner(hidden, weights, eps)