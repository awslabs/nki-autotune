from pathlib import Path

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

from nkigym.tune.verify import _rewrite_to_fp32

"""
d0: K
d1: M
d2: N
"""


@nki.jit
def kernel_0(lhs_T, rhs):
    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    psum_acc = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d0_0 in range(16):
        for i_d1_0 in range(1):
            nisa.dma_copy(
                dst=sbuf_lhs_T[0:128, i_d0_0, (i_d1_0) * 2048 : (i_d1_0) * 2048 + 2048],
                src=lhs_T[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d1_0) * 2048 : (i_d1_0) * 2048 + 2048],
            )
    for i_d0_0 in range(16):
        for i_d2_0 in range(1):
            nisa.dma_copy(
                dst=sbuf_rhs[0:128, i_d0_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                src=rhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.memset(psum_acc[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048], value=0.0)
    for i_d0_0 in range(16):
        for i_d1_0 in range(16):
            for i_d2_0 in range(4):
                nisa.nc_matmul(
                    dst=psum_acc[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                    stationary=sbuf_lhs_T[0:128, i_d0_0, (i_d1_0) * 128 : (i_d1_0) * 128 + 128],
                    moving=sbuf_rhs[0:128, i_d0_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.tensor_copy(
                sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                psum_acc[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.dma_copy(
                dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                src=sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    return hbm_out


@nki.jit
def kernel_1(lhs_T, rhs):
    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    psum_acc = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d0_0 in range(16):
        # split
        for i_d1_0 in range(16):
            nisa.dma_copy(
                dst=sbuf_lhs_T[0:128, i_d0_0, (i_d1_0) * 128 : (i_d1_0) * 128 + 128],
                src=lhs_T[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d1_0) * 128 : (i_d1_0) * 128 + 128],
            )
    for i_d0_0 in range(16):
        for i_d2_0 in range(1):
            nisa.dma_copy(
                dst=sbuf_rhs[0:128, i_d0_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                src=rhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.memset(psum_acc[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048], value=0.0)
    for i_d0_0 in range(16):
        for i_d1_0 in range(16):
            for i_d2_0 in range(4):
                nisa.nc_matmul(
                    dst=psum_acc[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                    stationary=sbuf_lhs_T[0:128, i_d0_0, (i_d1_0) * 128 : (i_d1_0) * 128 + 128],
                    moving=sbuf_rhs[0:128, i_d0_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.tensor_copy(
                sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                psum_acc[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.dma_copy(
                dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                src=sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    return hbm_out


@nki.jit
def kernel_2(lhs_T, rhs):
    sbuf_lhs_T = nl.ndarray((128, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    psum_acc = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d0_0 in range(16):
        for i_d2_0 in range(1):
            nisa.dma_copy(
                dst=sbuf_rhs[0:128, i_d0_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                src=rhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.memset(psum_acc[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048], value=0.0)
    for i_d0_0 in range(16):
        for i_d1_0 in range(16):
            # compute_at
            nisa.dma_copy(
                dst=sbuf_lhs_T[0:128, 0, 0:128],
                src=lhs_T[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d1_0) * 128 : (i_d1_0) * 128 + 128],
            )
            for i_d2_0 in range(4):
                nisa.nc_matmul(
                    dst=psum_acc[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                    stationary=sbuf_lhs_T[0:128, 0, 0:128],
                    moving=sbuf_rhs[0:128, i_d0_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.tensor_copy(
                sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                psum_acc[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.dma_copy(
                dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                src=sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    return hbm_out


@nki.jit
def kernel_3(lhs_T, rhs):
    sbuf_lhs_T = nl.ndarray((128, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    psum_acc = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d0_0 in range(16):
        # Split
        for i_d2_0 in range(4):
            nisa.dma_copy(
                dst=sbuf_rhs[0:128, i_d0_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                src=rhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
            )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.memset(psum_acc[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048], value=0.0)
    for i_d0_0 in range(16):
        for i_d1_0 in range(16):
            nisa.dma_copy(
                dst=sbuf_lhs_T[0:128, 0, 0:128],
                src=lhs_T[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d1_0) * 128 : (i_d1_0) * 128 + 128],
            )
            for i_d2_0 in range(4):
                nisa.nc_matmul(
                    dst=psum_acc[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                    stationary=sbuf_lhs_T[0:128, 0, 0:128],
                    moving=sbuf_rhs[0:128, i_d0_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.tensor_copy(
                sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                psum_acc[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.dma_copy(
                dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                src=sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    return hbm_out


@nki.jit
def kernel_4(lhs_T, rhs):
    sbuf_lhs_T = nl.ndarray((128, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    psum_acc = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.memset(psum_acc[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048], value=0.0)
    for i_d0_0 in range(16):
        for i_d1_0 in range(16):
            nisa.dma_copy(
                dst=sbuf_lhs_T[0:128, 0, 0:128],
                src=lhs_T[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d1_0) * 128 : (i_d1_0) * 128 + 128],
            )
            for i_d2_0 in range(4):
                # compute_at
                nisa.dma_copy(
                    dst=sbuf_rhs[0:128, 0, 0:512],
                    src=rhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                )
                nisa.nc_matmul(
                    dst=psum_acc[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                    stationary=sbuf_lhs_T[0:128, 0, 0:128],
                    moving=sbuf_rhs[0:128, 0, 0:512],
                )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.tensor_copy(
                sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                psum_acc[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.dma_copy(
                dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                src=sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    return hbm_out


@nki.jit
def kernel_5(lhs_T, rhs):
    sbuf_lhs_T = nl.ndarray((128, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    psum_acc = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        # Split
        for i_d2_0 in range(4):
            nisa.memset(psum_acc[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512], value=0.0)
    for i_d0_0 in range(16):
        for i_d1_0 in range(16):
            nisa.dma_copy(
                dst=sbuf_lhs_T[0:128, 0, 0:128],
                src=lhs_T[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d1_0) * 128 : (i_d1_0) * 128 + 128],
            )
            for i_d2_0 in range(4):
                nisa.dma_copy(
                    dst=sbuf_rhs[0:128, 0, 0:512],
                    src=rhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                )
                nisa.nc_matmul(
                    dst=psum_acc[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                    stationary=sbuf_lhs_T[0:128, 0, 0:128],
                    moving=sbuf_rhs[0:128, 0, 0:512],
                )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.tensor_copy(
                sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                psum_acc[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.dma_copy(
                dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                src=sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    return hbm_out


@nki.jit
def kernel_6(lhs_T, rhs):
    sbuf_lhs_T = nl.ndarray((128, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    psum_acc = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        for i_d2_0 in range(4):
            nisa.memset(psum_acc[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512], value=0.0)
    # reorder
    for i_d1_0 in range(16):
        for i_d0_0 in range(16):
            nisa.dma_copy(
                dst=sbuf_lhs_T[0:128, 0, 0:128],
                src=lhs_T[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d1_0) * 128 : (i_d1_0) * 128 + 128],
            )
            for i_d2_0 in range(4):
                nisa.dma_copy(
                    dst=sbuf_rhs[0:128, 0, 0:512],
                    src=rhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                )
                nisa.nc_matmul(
                    dst=psum_acc[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                    stationary=sbuf_lhs_T[0:128, 0, 0:128],
                    moving=sbuf_rhs[0:128, 0, 0:512],
                )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.tensor_copy(
                sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                psum_acc[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.dma_copy(
                dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                src=sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    return hbm_out


@nki.jit
def kernel_7(lhs_T, rhs):
    sbuf_lhs_T = nl.ndarray((128, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    psum_acc = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        # compute_at
        for i_d2_0 in range(4):
            nisa.memset(psum_acc[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512], value=0.0)
        for i_d0_0 in range(16):
            nisa.dma_copy(
                dst=sbuf_lhs_T[0:128, 0, 0:128],
                src=lhs_T[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d1_0) * 128 : (i_d1_0) * 128 + 128],
            )
            for i_d2_0 in range(4):
                nisa.dma_copy(
                    dst=sbuf_rhs[0:128, 0, 0:512],
                    src=rhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                )
                nisa.nc_matmul(
                    dst=psum_acc[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                    stationary=sbuf_lhs_T[0:128, 0, 0:128],
                    moving=sbuf_rhs[0:128, 0, 0:512],
                )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.tensor_copy(
                sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                psum_acc[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.dma_copy(
                dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                src=sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    return hbm_out


@nki.jit
def kernel_8(lhs_T, rhs):
    sbuf_lhs_T = nl.ndarray((128, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    psum_acc = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        for i_d2_0 in range(4):
            nisa.memset(psum_acc[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512], value=0.0)
        for i_d0_0 in range(16):
            for i_d2_0 in range(4):
                # compute_at
                nisa.dma_copy(
                    dst=sbuf_lhs_T[0:128, 0, 0:128],
                    src=lhs_T[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d1_0) * 128 : (i_d1_0) * 128 + 128],
                )
                nisa.dma_copy(
                    dst=sbuf_rhs[0:128, 0, 0:512],
                    src=rhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                )
                nisa.nc_matmul(
                    dst=psum_acc[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                    stationary=sbuf_lhs_T[0:128, 0, 0:128],
                    moving=sbuf_rhs[0:128, 0, 0:512],
                )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.tensor_copy(
                sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                psum_acc[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.dma_copy(
                dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                src=sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    return hbm_out


@nki.jit
def kernel_9(lhs_T, rhs):
    sbuf_lhs_T = nl.ndarray((128, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    psum_acc = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        for i_d2_0 in range(4):
            nisa.memset(psum_acc[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512], value=0.0)
        # reorder
        for i_d2_0 in range(4):
            for i_d0_0 in range(16):
                nisa.dma_copy(
                    dst=sbuf_lhs_T[0:128, 0, 0:128],
                    src=lhs_T[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d1_0) * 128 : (i_d1_0) * 128 + 128],
                )
                nisa.dma_copy(
                    dst=sbuf_rhs[0:128, 0, 0:512],
                    src=rhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                )
                nisa.nc_matmul(
                    dst=psum_acc[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                    stationary=sbuf_lhs_T[0:128, 0, 0:128],
                    moving=sbuf_rhs[0:128, 0, 0:512],
                )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.tensor_copy(
                sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                psum_acc[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.dma_copy(
                dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                src=sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    return hbm_out


@nki.jit
def kernel_10(lhs_T, rhs):
    sbuf_lhs_T = nl.ndarray((128, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    psum_acc = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        for i_d2_0 in range(4):
            # compute_at
            nisa.memset(psum_acc[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512], value=0.0)
            for i_d0_0 in range(16):
                nisa.dma_copy(
                    dst=sbuf_lhs_T[0:128, 0, 0:128],
                    src=lhs_T[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d1_0) * 128 : (i_d1_0) * 128 + 128],
                )
                nisa.dma_copy(
                    dst=sbuf_rhs[0:128, 0, 0:512],
                    src=rhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                )
                nisa.nc_matmul(
                    dst=psum_acc[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                    stationary=sbuf_lhs_T[0:128, 0, 0:128],
                    moving=sbuf_rhs[0:128, 0, 0:512],
                )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.tensor_copy(
                sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                psum_acc[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.dma_copy(
                dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                src=sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    return hbm_out


@nki.jit
def kernel_11(lhs_T, rhs):
    sbuf_lhs_T = nl.ndarray((128, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    psum_acc = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        for i_d2_0 in range(4):
            nisa.memset(psum_acc[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512], value=0.0)
            for i_d0_0 in range(16):
                nisa.dma_copy(
                    dst=sbuf_lhs_T[0:128, 0, 0:128],
                    src=lhs_T[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d1_0) * 128 : (i_d1_0) * 128 + 128],
                )
                nisa.dma_copy(
                    dst=sbuf_rhs[0:128, 0, 0:512],
                    src=rhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                )
                nisa.nc_matmul(
                    dst=psum_acc[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                    stationary=sbuf_lhs_T[0:128, 0, 0:128],
                    moving=sbuf_rhs[0:128, 0, 0:512],
                )
    for i_d1_0 in range(16):
        # split
        for i_d2_0 in range(4):
            nisa.tensor_copy(
                sbuf_prod[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                psum_acc[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
            )
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.dma_copy(
                dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                src=sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    return hbm_out


@nki.jit
def kernel_12(lhs_T, rhs):
    sbuf_lhs_T = nl.ndarray((128, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    psum_acc = nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        for i_d2_0 in range(4):
            nisa.memset(psum_acc[0:128, 0, 0:512], value=0.0)
            for i_d0_0 in range(16):
                nisa.dma_copy(
                    dst=sbuf_lhs_T[0:128, 0, 0:128],
                    src=lhs_T[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d1_0) * 128 : (i_d1_0) * 128 + 128],
                )
                nisa.dma_copy(
                    dst=sbuf_rhs[0:128, 0, 0:512],
                    src=rhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                )
                nisa.nc_matmul(
                    dst=psum_acc[0:128, 0, 0:512],
                    stationary=sbuf_lhs_T[0:128, 0, 0:128],
                    moving=sbuf_rhs[0:128, 0, 0:512],
                )
            # reverse_compute_at
            nisa.tensor_copy(sbuf_prod[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512], psum_acc[0:128, 0, 0:512])
    for i_d1_0 in range(16):
        for i_d2_0 in range(1):
            nisa.dma_copy(
                dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
                src=sbuf_prod[0:128, i_d1_0, (i_d2_0) * 2048 : (i_d2_0) * 2048 + 2048],
            )
    return hbm_out


@nki.jit
def kernel_13(lhs_T, rhs):
    sbuf_lhs_T = nl.ndarray((128, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    psum_acc = nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        for i_d2_0 in range(4):
            nisa.memset(psum_acc[0:128, 0, 0:512], value=0.0)
            for i_d0_0 in range(16):
                nisa.dma_copy(
                    dst=sbuf_lhs_T[0:128, 0, 0:128],
                    src=lhs_T[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d1_0) * 128 : (i_d1_0) * 128 + 128],
                )
                nisa.dma_copy(
                    dst=sbuf_rhs[0:128, 0, 0:512],
                    src=rhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                )
                nisa.nc_matmul(
                    dst=psum_acc[0:128, 0, 0:512],
                    stationary=sbuf_lhs_T[0:128, 0, 0:128],
                    moving=sbuf_rhs[0:128, 0, 0:512],
                )
            nisa.tensor_copy(sbuf_prod[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512], psum_acc[0:128, 0, 0:512])
    for i_d1_0 in range(16):
        # split
        for i_d2_0 in range(4):
            nisa.dma_copy(
                dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                src=sbuf_prod[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
            )
    return hbm_out


@nki.jit
def kernel_14(lhs_T, rhs):
    sbuf_lhs_T = nl.ndarray((128, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    psum_acc = nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum)
    sbuf_prod = nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        for i_d2_0 in range(4):
            nisa.memset(psum_acc[0:128, 0, 0:512], value=0.0)
            for i_d0_0 in range(16):
                nisa.dma_copy(
                    dst=sbuf_lhs_T[0:128, 0, 0:128],
                    src=lhs_T[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d1_0) * 128 : (i_d1_0) * 128 + 128],
                )
                nisa.dma_copy(
                    dst=sbuf_rhs[0:128, 0, 0:512],
                    src=rhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                )
                nisa.nc_matmul(
                    dst=psum_acc[0:128, 0, 0:512],
                    stationary=sbuf_lhs_T[0:128, 0, 0:128],
                    moving=sbuf_rhs[0:128, 0, 0:512],
                )
            nisa.tensor_copy(sbuf_prod[0:128, 0, 0:512], psum_acc[0:128, 0, 0:512])
            # reverse_compute_at
            nisa.dma_copy(
                dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                src=sbuf_prod[0:128, 0, 0:512],
            )
    return hbm_out


def _main() -> None:
    """CPU-sim every ``kernel_N`` against a single numpy matmul golden."""
    K, M, N = 2048, 2048, 2048
    rng = np.random.default_rng(0)
    lhs_T = rng.standard_normal((K, M)).astype(np.float32)
    rhs = rng.standard_normal((K, N)).astype(np.float32)
    expected = lhs_T.T @ rhs

    atol = rtol = 5e-3
    source = Path(__file__).read_text()
    sim_source = _rewrite_to_fp32(source)
    ns: dict = {}
    exec(sim_source, ns)

    kernel_names = [name for name in ns if name.startswith("kernel_")]
    for name in kernel_names[-3:]:
        actual = nki.simulate(ns[name])(lhs_T, rhs)
        if isinstance(actual, tuple):
            actual = actual[0]
        actual = np.asarray(actual)
        max_abs = float(np.abs(actual - expected).max())
        max_rel = float((np.abs(actual - expected) / (np.abs(expected) + atol)).max())
        ok = np.allclose(actual, expected, atol=atol, rtol=rtol)
        print(f"{name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} pass={ok}")


if __name__ == "__main__":
    _main()
