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
from src.kernels.rmsnorm_linear import blocked_fused_rms_norm_linear
from src.golden.rmsnorm_linear import cpu_golden_result
from src.golden.gemm import gemm_cpu_golden


if __name__ == "__main__":
    batch, M, K, N = 1, 2048, 4096, 512
    eps = 1e-6
    atol, rtol = 1e-2, 1e-3
    data_type = np.float32

    lhs = np.random.random_sample((batch, M, K)).astype(data_type)
    rhs = np.random.random_sample((K, N)).astype(data_type)

    # golden_output = nl.static_cast(cpu_golden_result(lhs, None, None, rhs, eps), data_type)
    golden_output = nl.static_cast(gemm_cpu_golden(lhs, rhs, False), data_type)
    print(golden_output.shape, golden_output)

    numeric_func = baremetal(blocked_fused_rms_norm_linear)
    nki_out = nl.static_cast(numeric_func(lhs, rhs, 2, 1, 2, 1, 1, 1, nl.float32, eps), data_type)
    print(nki_out.shape, nki_out)

    assert allclose(nki_out, golden_output, atol=atol, rtol=rtol, verbose=1)
