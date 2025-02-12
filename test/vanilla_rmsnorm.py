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
from src.kernels import nki_rms_norm_

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

def rmsnorm_golden(hidden: np.ndarray, weights: np.ndarray, eps: np.float16):
    """Vanilla RMS Norm

    Args:
        hidden (np.ndarray): input tensor
        weights (np.ndarray): matmul weights
        eps: (np.float16): epsilon for numerical stability
    """
    s_hidden = hidden * hidden
    ms_hidden = np.mean(s_hidden, axis=-1, keepdims=True)
    rms_hidden = np.sqrt(ms_hidden)
    rms_norm_hidden = hidden / rms_hidden
    output_np = rms_norm_hidden, * weights
    print(f"output_np = {output_np} {output_np.shape}")
    return output_np

if __name__ == "__main__":
    batch = 1
    seqlen = 1024
    dim = 512
    d_head = 16

    hidden = np.random.rand(batch, seqlen, dim).astype(np.float16)
    weights = np.random.rand(dim).astype(np.float16)
    eps = 1e-6

    output_nki = nki_rms_norm_(hidden, weights, eps, 128)
    print(
        f"rms_norm_output_nki = {output_nki} {output_nki.shape}"
    )
    
    tuner = AutotuneKernel.trace(
        nki_rms_norm_,
        iters=10,
        configs=get_autotune_configs(),
        show_compiler_tb=True,
    )
    tuner(hidden, weights, eps)