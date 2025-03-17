# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki as nki
import numpy as np
from neuronxcc.nki import baremetal
from neuronxcc.starfish.support.util import allclose

from src.autotune_kernel import Autotune
from src.benchmark import profile_kernel
from src.kernels.matmul import gemm_with_non_transposed_lhs_MN
from src.golden.gemm import gemm_core, gemm_cpu_golden


if __name__ == "__main__":
    batch, M, K, N = 1, 2048, 4096, 512
    data_type = np.float32
    atol, rtol = 1e-2, 1e-3
    lhs = np.random.random_sample((batch, M, K)).astype(data_type)
    rhs = np.random.random_sample((K, N)).astype(data_type)

    golden_1 = nl.static_cast(gemm_core(lhs[0], rhs, False), data_type)
    golden_2 = nl.static_cast(gemm_cpu_golden(lhs, rhs, False), data_type)
    assert allclose(golden_1, golden_2, atol=atol, rtol=rtol, verbose=1)

    numeric_func = baremetal(gemm_with_non_transposed_lhs_MN)
    nki_out = numeric_func(lhs[0], rhs, 2, 1, 2, 1, 1, 1)

    assert allclose(nki_out, golden_1, atol=atol, rtol=rtol, verbose=1)
    assert allclose(nki_out, golden_2, atol=atol, rtol=rtol, verbose=1)
