# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np
from neuronxcc.nki import baremetal
from neuronxcc.starfish.support.util import allclose

from src.golden.gemm import gemm_cpu_golden
from src.kernels.matmul import gemm_with_non_transposed_lhs_MN
from src.tune.autotune_kernel import Autotune
from src.tune.benchmark import profile_kernel

if __name__ == "__main__":
    batch, M, K, N = 1, 2048, 4096, 512
    data_type = np.float32
    atol, rtol = 1e-2, 1e-3
    lhs = np.random.random_sample((M, K)).astype(data_type)
    rhs = np.random.random_sample((K, N)).astype(data_type)

    golden = nl.static_cast(gemm_cpu_golden(lhs, rhs, False), data_type)
    print(f"golden: {golden.shape}\n{golden}")

    numeric_func = baremetal(gemm_with_non_transposed_lhs_MN)
    nki_out = numeric_func(lhs, rhs, 2, 1, 2, 1, 1, 1)
    print(f"nki_out: {nki_out.shape}\n{nki_out}")

    assert allclose(nki_out, golden, atol=atol, rtol=rtol, verbose=1)

    latency, _ = profile_kernel(
        gemm_with_non_transposed_lhs_MN,
        (lhs, rhs),
        cache_dir=".",
        configs={"NUM_BLOCK_M": 1, "NUM_BLOCK_N": 1, "NUM_BLOCK_K": 1, "BUFFER_M": 1, "BUFFER_N": 1, "BUFFER_K": 1},
    )
    print(latency)
