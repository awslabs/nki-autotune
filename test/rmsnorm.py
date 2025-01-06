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
    configs = [
        {"multi_buffer": 1},
        {"multi_buffer": 2},
    ]
    return configs


def rmsnorm_golden(hidden: np.ndarray, weights: np.ndarray):
    """Fused RMS Norm and Matmul

    Args:
        hidden (np.ndarray): input tensor
        weights (np.ndarray): matmul weights
    """
    s_hidden = hidden * hidden
    ms_hidden = np.mean(s_hidden, axis=-1, keepdims=True)
    rms_hidden = np.sqrt(ms_hidden)
    rms_norm_hidden = hidden / rms_hidden
    output_np = np.matmul(rms_norm_hidden, weights)
    return output_np


if __name__ == "__main__":
    batch = 1
    seqlen = 1024
    dim = 512
    d_head = 16

    hidden = nt.tensor[[batch, seqlen, dim], nl.bfloat16]
    weights = nt.tensor[[dim, d_head], nl.bfloat16]
    tuner = AutotuneKernel.trace(
        allocated_fused_rms_norm_qkv,
        iters=10,
        configs=get_autotune_configs(),
        max_workers=1,
        show_compiler_tb=True,
    )
    tuner(hidden, weights)
