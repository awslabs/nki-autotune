"""CPU-sim AND Trn2-HW-profile every hand-written ``kernel_<id>`` in this file
against ``lhs_T.T @ rhs``.

Each ``kernel_<id>`` is a hand-stepped NKI rendering of the 2048³ bf16
``matmul_lhsT_rhs`` workload. ``_main`` discovers them by name, then:

1. CPU-sim — runs each through ``simulate_fp32`` (fp32 end-to-end) and compares
   against the numpy golden, printing the max abs error and PASS/FAIL per kernel.
2. HW profile — AST-extracts each into a standalone module, compiles it with the
   scheduler + linear-scan allocator OFF, and benchmarks it on real Trn2 hardware
   via ``autotune.runner.profile``, printing a per-kernel MFU table.

Add another ``kernel_<id>`` and it is picked up automatically — no edit to
``_main``. Some kernels are EXPECTED to fail HW compile/run (e.g. rungs that hold
a full-extent PSUM/SBUF buffer the BIR verifier rejects); those show up in the
profile failure summary and do NOT abort the run. Only a CPU-sim divergence makes
the run exit non-zero.

Usage (on a Trn2 box via the SSH transport — it appends ``--cache``)::

    transport/ssh_host.sh --host gym-1 --cmd "python examples/manual_transforms.py" \
        --cache /home/weittang/workplace/cache/manual_transforms
"""

import argparse
import ast
import os
import re
import shutil
from collections.abc import Callable

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

from autotune.runner.api import profile
from autotune.runner.types import KernelJob
from nkigym.synthesis import simulate_fp32

K, M, N = 2048, 2048, 2048
INPUT_SPECS: dict[str, tuple[tuple[int, ...], str]] = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}
SEED = 0
ATOL, RTOL = 5e-3, 5e-3
NEURON_PLATFORM_TARGET = "trn2"
"""Hand-placed PSUM: the neuronx-cc scheduler + linear-scan allocator must be OFF
(scheduler-on OOMs PSUM even with per-iteration alloc)."""
SCHEDULER_OFF_ARGS = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")
_KERNEL_NAME = re.compile(r"^kernel_\d+$")


@nki.jit
def kernel_0(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[0:128, i_d1_0, 0 : 0 + 2048], value=0.0)
    for i_d0_0 in range(16):
        for i_d1_0 in range(16):
            for i_d2_0 in range(4):
                nisa.nc_matmul(
                    stationary=sbuf_lhs_T[0:128, i_d0_0, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                    moving=sbuf_rhs[0:128, i_d0_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                    dst=psum_prod[0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                )
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d1_0 in range(16):
        nisa.tensor_copy(src=psum_prod[0:128, i_d1_0, 0 : 0 + 2048], dst=sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048], dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


@nki.jit
def kernel_1(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[0:128, i_d1_0, 0 : 0 + 2048], value=0.0)
    # Reorder
    for i_d2_0 in range(4):
        for i_d0_0 in range(16):
            for i_d1_0 in range(16):
                nisa.nc_matmul(
                    stationary=sbuf_lhs_T[0:128, i_d0_0, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                    moving=sbuf_rhs[0:128, i_d0_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                    dst=psum_prod[0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                )
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d1_0 in range(16):
        nisa.tensor_copy(src=psum_prod[0:128, i_d1_0, 0 : 0 + 2048], dst=sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048], dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


@nki.jit
def kernel_2(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[0:128, i_d1_0, 0 : 0 + 2048], value=0.0)
    for i_d2_0 in range(4):
        # Split
        for i_d0_0 in range(2):
            for i_d0_1 in range(8):
                for i_d1_0 in range(16):
                    nisa.nc_matmul(
                        stationary=sbuf_lhs_T[0:128, i_d0_0 * 8 + i_d0_1, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                        moving=sbuf_rhs[0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        dst=psum_prod[0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                    )
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d1_0 in range(16):
        nisa.tensor_copy(src=psum_prod[0:128, i_d1_0, 0 : 0 + 2048], dst=sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048], dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


@nki.jit
def kernel_3(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[0:128, i_d1_0, 0 : 0 + 2048], value=0.0)
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d0_1 in range(8):
                # Split
                for i_d1_0 in range(4):
                    for i_d1_1 in range(4):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[0:128, (i_d1_0 * 4 + i_d1_1), i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d1_0 in range(16):
        nisa.tensor_copy(src=psum_prod[0:128, i_d1_0, 0 : 0 + 2048], dst=sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048], dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


@nki.jit
def kernel_4(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[0:128, i_d1_0, 0 : 0 + 2048], value=0.0)
    # Reorder
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d0_1 in range(8):
                    for i_d1_1 in range(4):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[0:128, (i_d1_0 * 4 + i_d1_1), i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d1_0 in range(16):
        nisa.tensor_copy(src=psum_prod[0:128, i_d1_0, 0 : 0 + 2048], dst=sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048], dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


@nki.jit
def kernel_5(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[0:128, i_d1_0, 0 : 0 + 2048], value=0.0)
    # Reorder
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[0:128, (i_d1_0 * 4 + i_d1_1), i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d1_0 in range(16):
        nisa.tensor_copy(src=psum_prod[0:128, i_d1_0, 0 : 0 + 2048], dst=sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048], dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


@nki.jit
def kernel_6(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0:128, i_d0_0, 0 : 0 + 2048]
        )
    # Buffer layout
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0 : 0 + 2048], value=0.0)
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d1_0 in range(16):
        nisa.tensor_copy(src=psum_prod[i_d1_0][0:128, 0, 0 : 0 + 2048], dst=sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048], dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


@nki.jit
def kernel_7(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0 : 0 + 2048], value=0.0)
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    # Split
    for i_d1_0 in range(16):
        for i_d2_0 in range(4):
            nisa.tensor_copy(
                src=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048], dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


@nki.jit
def kernel_8(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0 : 0 + 2048], value=0.0)
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    # Reorder
    for i_d2_0 in range(4):
        for i_d1_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048], dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out

"""
code motion legality checks
source instruction = 
```
nisa.tensor_copy(
    src=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
    dst=sbuf_prod[0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
)
```
source enclosing loops = 
```
for i_d2_0 in range(4):
    for i_d1_0 in range(16):
```
target location = after "for i_d0_0 in range(2)" loop closure.
target enclosing loops = 
```
for i_d2_0 in range(4): --> relevant dimension loop
```
legality checks:
1. Are relevant dimension loops an identical prefix to source enclosing loops? Yes.
Counter examples that violate this check:
target enclosing loops = 
```
for i_d2_0 in range(2): --> trip count is not the same
```

```
for i_d2_0 in range(4):
    for i_d1_0 in range(8):  --> trip count is not the same
```

```
for i_d1_0 in range(16):
    for i_d2_0 in range(4): --> loop order is not the same
```

```
for i_d1_0 in range(16): --> not the same loop prefix, missing for i_d2_0 in range(4) in front of it
```

2. Producer-consumer data dependency respected? Yes
source instruction data access:
Read: psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512]
Write: sbuf_prod[0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512]
If moved to target location:
1. psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512] has been fully written by producers.
2. sbuf_prod[0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512] is after producers.
--> ok
"""
@nki.jit
def kernel_9(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0 : 0 + 2048], value=0.0)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        # Code motion
        for i_d1_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048], dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


@nki.jit
def kernel_10(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0 : 0 + 2048], value=0.0)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        for i_d1_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    # Split
    for i_d1_0 in range(16):
        for i_d2_0 in range(4):
            nisa.dma_copy(
                src=sbuf_prod[0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


@nki.jit
def kernel_11(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0 : 0 + 2048], value=0.0)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        for i_d1_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    # Reorder
    for i_d2_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


@nki.jit
def kernel_12(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0 : 0 + 2048], value=0.0)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        # Side effect: automatic buffer scope tightening and compaction
        sbuf_prod = nl.ndarray((128, 16, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        for i_d1_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512], dst=sbuf_prod[0:128, i_d1_0, 0:512]
            )
        # Code motion
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[0:128, i_d1_0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


@nki.jit
def kernel_13(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0 : 0 + 2048], value=0.0)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        # Buffer layout
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[i_d1_0][0:128, 0, 0:512],
            )
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d1_0][0:128, 0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


@nki.jit
def kernel_14(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    # Split
    for i_d1_0 in range(16):
        for i_d2_0 in range(4):
            nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512], value=0.0)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[i_d1_0][0:128, 0, 0:512],
            )
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d1_0][0:128, 0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


@nki.jit
def kernel_15(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    # Reorder
    for i_d2_0 in range(4):
        for i_d1_0 in range(16):
            nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512], value=0.0)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[i_d1_0][0:128, 0, 0:512],
            )
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d1_0][0:128, 0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


@nki.jit
def kernel_16(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0:128, i_d0_0, 0 : 0 + 2048]
        )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        # Side effect: automatic buffer scope tightening and compaction
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        # Code motion
        for i_d1_0 in range(16):
            nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0:512], value=0.0)
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.tensor_copy(src=psum_prod[i_d1_0][0:128, 0, 0:512], dst=sbuf_prod[i_d1_0][0:128, 0, 0:512])
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d1_0][0:128, 0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


@nki.jit
def kernel_17(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    # Split
    for i_d0_0 in range(2):
        for i_d0_1 in range(8):
            nisa.dma_copy(
                src=rhs[(i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128, 0 : 0 + 2048],
                dst=sbuf_rhs[0:128, (i_d0_0 * 8 + i_d0_1), 0 : 0 + 2048],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0:512], value=0.0)
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.tensor_copy(src=psum_prod[i_d1_0][0:128, 0, 0:512], dst=sbuf_prod[i_d1_0][0:128, 0, 0:512])
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d1_0][0:128, 0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


@nki.jit
def kernel_18(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    # Split
    for i_d0_0 in range(2):
        for i_d0_1 in range(8):
            for i_d2_0 in range(4):
                nisa.dma_copy(
                    src=rhs[
                        (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[0:128, (i_d0_0 * 8 + i_d0_1), i_d2_0 * 512 : i_d2_0 * 512 + 512],
                )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0:512], value=0.0)
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.tensor_copy(src=psum_prod[i_d1_0][0:128, 0, 0:512], dst=sbuf_prod[i_d1_0][0:128, 0, 0:512])
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d1_0][0:128, 0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


@nki.jit
def kernel_19(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    # Reorder
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d0_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[0:128, (i_d0_0 * 8 + i_d0_1), i_d2_0 * 512 : i_d2_0 * 512 + 512],
                )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0:512], value=0.0)
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.tensor_copy(src=psum_prod[i_d1_0][0:128, 0, 0:512], dst=sbuf_prod[i_d1_0][0:128, 0, 0:512])
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d1_0][0:128, 0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


@nki.jit
def kernel_20(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0 : 0 + 2048]
        )

    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0:512], value=0.0)
        for i_d0_0 in range(2):
            # Side effect: automatic buffer scope tightening and compaction
            sbuf_rhs = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            # Code motion
            for i_d0_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[0:128, i_d0_1, 0:512],
                )
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0:128, i_d0_1, 0:512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.tensor_copy(src=psum_prod[i_d1_0][0:128, 0, 0:512], dst=sbuf_prod[i_d1_0][0:128, 0, 0:512])
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d1_0][0:128, 0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


@nki.jit
def kernel_21(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0 : 0 + 2048]
        )

    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0:512], value=0.0)
        for i_d0_0 in range(2):
            # Buffer layout
            sbuf_rhs = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)]
            for i_d0_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                )
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.tensor_copy(src=psum_prod[i_d1_0][0:128, 0, 0:512], dst=sbuf_prod[i_d1_0][0:128, 0, 0:512])
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d1_0][0:128, 0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


@nki.jit
def kernel_22(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    # Split
    for i_d0_0 in range(16):
        for i_d1_0 in range(4):
            nisa.dma_copy(
                src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, i_d1_0 * 512 : i_d1_0 * 512 + 512],
                dst=sbuf_lhs_T[0:128, i_d0_0, i_d1_0 * 512 : i_d1_0 * 512 + 512],
            )

    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0:512], value=0.0)
        for i_d0_0 in range(2):
            sbuf_rhs = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)]
            for i_d0_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                )
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.tensor_copy(src=psum_prod[i_d1_0][0:128, 0, 0:512], dst=sbuf_prod[i_d1_0][0:128, 0, 0:512])
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d1_0][0:128, 0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


@nki.jit
def kernel_23(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    # Split
    for i_d0_0 in range(2):
        for i_d0_1 in range(8):
            for i_d1_0 in range(4):
                nisa.dma_copy(
                    src=lhs_T[
                        (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                        i_d1_0 * 512 : i_d1_0 * 512 + 512,
                    ],
                    dst=sbuf_lhs_T[0:128, (i_d0_0 * 8 + i_d0_1), i_d1_0 * 512 : i_d1_0 * 512 + 512],
                )

    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0:512], value=0.0)
        for i_d0_0 in range(2):
            sbuf_rhs = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)]
            for i_d0_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                )
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.tensor_copy(src=psum_prod[i_d1_0][0:128, 0, 0:512], dst=sbuf_prod[i_d1_0][0:128, 0, 0:512])
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d1_0][0:128, 0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


@nki.jit
def kernel_24(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    # Reorder
    for i_d0_0 in range(2):
        for i_d1_0 in range(4):
            for i_d0_1 in range(8):
                nisa.dma_copy(
                    src=lhs_T[
                        (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                        i_d1_0 * 512 : i_d1_0 * 512 + 512,
                    ],
                    dst=sbuf_lhs_T[0:128, (i_d0_0 * 8 + i_d0_1), i_d1_0 * 512 : i_d1_0 * 512 + 512],
                )

    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0:512], value=0.0)
        for i_d0_0 in range(2):
            sbuf_rhs = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)]
            for i_d0_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                )
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.tensor_copy(src=psum_prod[i_d1_0][0:128, 0, 0:512], dst=sbuf_prod[i_d1_0][0:128, 0, 0:512])
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d1_0][0:128, 0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


@nki.jit
def kernel_25(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0:512], value=0.0)
        for i_d0_0 in range(2):
            sbuf_rhs = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)]
            for i_d0_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                )
            for i_d1_0 in range(4):
                # Side effect: automatic buffer scope tightening and compaction
                sbuf_lhs_T = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                # Code motion
                for i_d0_1 in range(8):
                    nisa.dma_copy(
                        src=lhs_T[
                            (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                            i_d1_0 * 512 : i_d1_0 * 512 + 512,
                        ],
                        dst=sbuf_lhs_T[0:128, i_d0_1, 0:512],
                    )
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0:128, i_d0_1, i_d1_1 * 128 : i_d1_1 * 128 + 128],
                            moving=sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.tensor_copy(src=psum_prod[i_d1_0][0:128, 0, 0:512], dst=sbuf_prod[i_d1_0][0:128, 0, 0:512])
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d1_0][0:128, 0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


@nki.jit
def kernel_26(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        """init_one_stage"""
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0:512], value=0.0)
        for i_d0_0 in range(2):
            sbuf_rhs = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)]
            for i_d0_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                )
            for i_d1_0 in range(4):
                # Buffer layout
                sbuf_lhs_T = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)]
                for i_d0_1 in range(8):
                    nisa.dma_copy(
                        src=lhs_T[
                            (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                            i_d1_0 * 512 : i_d1_0 * 512 + 512,
                        ],
                        dst=sbuf_lhs_T[i_d0_1][0:128, 0, 0:512],
                    )
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[i_d0_1][0:128, 0, i_d1_1 * 128 : i_d1_1 * 128 + 128],
                            moving=sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        )
        """drain_one_stage"""
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.tensor_copy(src=psum_prod[i_d1_0][0:128, 0, 0:512], dst=sbuf_prod[i_d1_0][0:128, 0, 0:512])
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d1_0][0:128, 0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


# Apply RFactor
@nki.jit
def kernel_27(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_d2_0 in range(4):
        """init_two_stage_0"""
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.memset(dst=sbuf_prod[i_d1_0][0:128, 0, 0:512], value=0.0)
        for i_d0_0 in range(2):
            sbuf_rhs = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)]
            for i_d0_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                )
            for i_d1_0 in range(4):
                sbuf_lhs_T = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)]
                for i_d0_1 in range(8):
                    nisa.dma_copy(
                        src=lhs_T[
                            (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                            i_d1_0 * 512 : i_d1_0 * 512 + 512,
                        ],
                        dst=sbuf_lhs_T[i_d0_1][0:128, 0, 0:512],
                    )
                for i_d1_1 in range(4):
                    """init_two_stage_1"""
                    psum_prod = nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum)
                    nisa.memset(dst=psum_prod[0:128, 0, 0:512], value=0.0)
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[i_d0_1][0:128, 0, i_d1_1 * 128 : i_d1_1 * 128 + 128],
                            moving=sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                            dst=psum_prod[0:128, 0, 0:512],
                        )
                    """drain_two_stage_0"""
                    sbuf_rfactor = nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(src=psum_prod[0:128, 0, 0:512], dst=sbuf_rfactor[0:128, 0, 0:512])
                    nisa.tensor_tensor(
                        data1=sbuf_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        data2=sbuf_rfactor[0:128, 0, 0:512],
                        dst=sbuf_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        op=nl.add,
                    )
        """drain_two_stage_1: None"""
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d1_0][0:128, 0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


def _discover_kernels(namespace: dict[str, object]) -> list[tuple[str, Callable]]:
    """Module-level ``kernel_<id>`` callables in ``namespace``, ordered by numeric id."""
    found = [(name, obj) for name, obj in namespace.items() if _KERNEL_NAME.match(name) and callable(obj)]
    return sorted(found, key=lambda item: int(item[0].split("_", 1)[1]))


def _check_numerics(name: str, kernel: Callable, inputs: dict[str, np.ndarray], expected: np.ndarray) -> bool:
    """CPU-sim ``kernel`` in fp32, compare to ``expected``; print max abs error + PASS/FAIL."""
    actual = np.asarray(simulate_fp32(kernel)(**inputs))
    max_abs = float(np.abs(actual - expected).max())
    ok = bool(np.allclose(actual, expected, atol=ATOL, rtol=RTOL))
    print(f"[sim] {name}: max_abs={max_abs:.3e} pass={ok}")
    return ok


def _kernel_source(name: str) -> str:
    """Standalone NKI module string for the hand kernel ``name`` (AST-extracted).

    ``inspect.getsource`` does not work through the ``@nki.jit`` wrapper, so the
    single ``def {name}`` is pulled from THIS file by AST and the three ``nki``
    imports prepended, giving ``profile`` a module it can compile in isolation.
    The ``@nki.jit`` decorator is dropped by ``ast.get_source_segment`` — the
    compile path wraps the bare function in ``nki.compiler.Kernel`` itself.
    """
    this_src = open(os.path.abspath(__file__), encoding="utf-8").read()
    tree = ast.parse(this_src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    body = ast.get_source_segment(this_src, fn)
    if body is None:
        raise RuntimeError(f"could not extract source for {name}")
    return "import nki\nimport nki.isa as nisa\nimport nki.language as nl\n\n\n" + body + "\n"


def _profile_on_hw(names: list[str], cache_dir: str) -> None:
    """Compile + benchmark every kernel on real Trn2 HW; print a per-kernel MFU table.

    Each kernel is AST-extracted into a standalone module and submitted to
    ``autotune.runner.profile`` with the scheduler + linear-scan allocator OFF
    (hand-placed PSUM). Kernels that fail to compile or run on HW are reported in
    the profile failure summary and do NOT abort — only CPU-sim divergence does.
    """
    jobs: dict[str, KernelJob] = {
        name: KernelJob(
            source=_kernel_source(name),
            func_name=name,
            output_shape=(M, N),
            input_specs=INPUT_SPECS,
            neuronx_cc_args=SCHEDULER_OFF_ARGS,
        )
        for name in names
    }
    print(
        f"\n[hw] compiling + profiling {len(jobs)} kernel(s) on {NEURON_PLATFORM_TARGET} (scheduler + linear-scan OFF)\n"
    )
    output = profile(
        jobs,
        cache_dir=cache_dir,
        seed=SEED,
        neuron_platform_target=NEURON_PLATFORM_TARGET,
        collect_detailed_profile=False,
    )
    print(output)


def _main() -> None:
    """Discover every ``kernel_<id>``, CPU-sim each against ``lhs_T.T @ rhs``, then
    profile each on real Trn2 HW. Exits non-zero only on a CPU-sim divergence."""
    parser = argparse.ArgumentParser(description="CPU-sim + Trn2-HW-profile every kernel_<id> against lhs_T.T @ rhs.")
    parser.add_argument("--cache", required=True, help="absolute cache dir (the SSH transport appends this)")
    args = parser.parse_args()
    cache_dir = os.path.join(args.cache, "manual_transforms")
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)

    rng = np.random.default_rng(SEED)
    inputs = {nm: rng.standard_normal(shape).astype(np.float32) for nm, (shape, _d) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]

    kernels = _discover_kernels(globals())
    print(f"[sim] {len(kernels)} kernel(s): {', '.join(name for name, _ in kernels)}")
    results = [(name, _check_numerics(name, kernel, inputs, expected)) for name, kernel in kernels]

    summary = "\n".join(f"{name}: pass={ok}" for name, ok in results)
    with open(os.path.join(cache_dir, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(summary + "\n")

    _profile_on_hw([name for name, _ in kernels], cache_dir)

    failed = [name for name, ok in results if not ok]
    if failed:
        raise SystemExit(f"[sim] FAILED: {', '.join(failed)}")
    print(f"[sim] all {len(results)} kernel(s) PASS")


if __name__ == "__main__":
    _main()
