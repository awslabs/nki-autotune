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
from src.golden.rmsnorm_linear import rmsnorm_linear_golden


if __name__ == "__main__":
    batch, M, K, N = 1, 2048, 4096, 512
    data_type = np.float32
    atol, rtol = 1e-2, 1e-3
    lhs = np.random.random_sample((batch, M, K)).astype(data_type)
    rhs = np.random.random_sample((K, N)).astype(data_type)
    eps = 1e-6

    golden = nl.static_cast(rmsnorm_linear_golden(lhs, None, None, rhs, eps), data_type)
    print(f"golden: {golden.shape}\n{golden}")

    numeric_func = baremetal(blocked_fused_rms_norm_linear)
    nki_out = nl.static_cast(numeric_func(lhs, rhs, 2, 1, 2, 1, 1, 1, nl.float32, eps), data_type)
    print(f"nki_out: {nki_out.shape}\n{nki_out}")

    assert allclose(nki_out, golden, atol=atol, rtol=rtol, verbose=1)
