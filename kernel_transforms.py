import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

from nkigym.synthesis import simulate_fp32

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
        nisa.dma_copy(dst=sbuf_lhs_T[0:128, i_d0_0, 0:2048], src=lhs_T[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, 0:2048])
    for i_d0_0 in range(16):
        nisa.dma_copy(dst=sbuf_rhs[0:128, i_d0_0, 0:2048], src=rhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, 0:2048])
    for i_d1_0 in range(16):
        nisa.memset(psum_acc[0:128, i_d1_0, 0:2048], value=0.0)
    for i_d0_0 in range(16):
        for i_d1_0 in range(16):
            for i_d2_0 in range(4):
                nisa.nc_matmul(
                    dst=psum_acc[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                    stationary=sbuf_lhs_T[0:128, i_d0_0, (i_d1_0) * 128 : (i_d1_0) * 128 + 128],
                    moving=sbuf_rhs[0:128, i_d0_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                )
    for i_d1_0 in range(16):
        nisa.tensor_copy(sbuf_prod[0:128, i_d1_0, 0:2048], psum_acc[0:128, i_d1_0, 0:2048])
    for i_d1_0 in range(16):
        nisa.dma_copy(dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, 0:2048], src=sbuf_prod[0:128, i_d1_0, 0:2048])
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
        nisa.dma_copy(dst=sbuf_rhs[0:128, i_d0_0, 0:2048], src=rhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, 0:2048])
    for i_d1_0 in range(16):
        nisa.memset(psum_acc[0:128, i_d1_0, 0:2048], value=0.0)
    for i_d0_0 in range(16):
        for i_d1_0 in range(16):
            for i_d2_0 in range(4):
                nisa.nc_matmul(
                    dst=psum_acc[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                    stationary=sbuf_lhs_T[0:128, i_d0_0, (i_d1_0) * 128 : (i_d1_0) * 128 + 128],
                    moving=sbuf_rhs[0:128, i_d0_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                )
    for i_d1_0 in range(16):
        nisa.tensor_copy(sbuf_prod[0:128, i_d1_0, 0:2048], psum_acc[0:128, i_d1_0, 0:2048])
    for i_d1_0 in range(16):
        nisa.dma_copy(dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, 0:2048], src=sbuf_prod[0:128, i_d1_0, 0:2048])
    return hbm_out


@nki.jit
def kernel_2(lhs_T, rhs):
    sbuf_lhs_T = nl.ndarray((128, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    psum_acc = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d0_0 in range(16):
        nisa.dma_copy(dst=sbuf_rhs[0:128, i_d0_0, 0:2048], src=rhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, 0:2048])
    for i_d1_0 in range(16):
        nisa.memset(psum_acc[0:128, i_d1_0, 0:2048], value=0.0)
    for i_d0_0 in range(16):
        for i_d1_0 in range(16):
            # move
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
        nisa.tensor_copy(sbuf_prod[0:128, i_d1_0, 0:2048], psum_acc[0:128, i_d1_0, 0:2048])
    for i_d1_0 in range(16):
        nisa.dma_copy(dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, 0:2048], src=sbuf_prod[0:128, i_d1_0, 0:2048])
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
        nisa.memset(psum_acc[0:128, i_d1_0, 0:2048], value=0.0)
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
        nisa.tensor_copy(sbuf_prod[0:128, i_d1_0, 0:2048], psum_acc[0:128, i_d1_0, 0:2048])
    for i_d1_0 in range(16):
        nisa.dma_copy(dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, 0:2048], src=sbuf_prod[0:128, i_d1_0, 0:2048])
    return hbm_out


@nki.jit
def kernel_4(lhs_T, rhs):
    sbuf_lhs_T = nl.ndarray((128, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    psum_acc = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.memset(psum_acc[0:128, i_d1_0, 0:2048], value=0.0)
    for i_d0_0 in range(16):
        for i_d1_0 in range(16):
            nisa.dma_copy(
                dst=sbuf_lhs_T[0:128, 0, 0:128],
                src=lhs_T[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d1_0) * 128 : (i_d1_0) * 128 + 128],
            )
            for i_d2_0 in range(4):
                # move
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
        nisa.tensor_copy(sbuf_prod[0:128, i_d1_0, 0:2048], psum_acc[0:128, i_d1_0, 0:2048])
    for i_d1_0 in range(16):
        nisa.dma_copy(dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, 0:2048], src=sbuf_prod[0:128, i_d1_0, 0:2048])
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
        nisa.tensor_copy(sbuf_prod[0:128, i_d1_0, 0:2048], psum_acc[0:128, i_d1_0, 0:2048])
    for i_d1_0 in range(16):
        nisa.dma_copy(dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, 0:2048], src=sbuf_prod[0:128, i_d1_0, 0:2048])
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
        nisa.tensor_copy(sbuf_prod[0:128, i_d1_0, 0:2048], psum_acc[0:128, i_d1_0, 0:2048])
    for i_d1_0 in range(16):
        nisa.dma_copy(dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, 0:2048], src=sbuf_prod[0:128, i_d1_0, 0:2048])
    return hbm_out


@nki.jit
def kernel_7(lhs_T, rhs):
    sbuf_lhs_T = nl.ndarray((128, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    psum_acc = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        # move
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
        nisa.tensor_copy(sbuf_prod[0:128, i_d1_0, 0:2048], psum_acc[0:128, i_d1_0, 0:2048])
    for i_d1_0 in range(16):
        nisa.dma_copy(dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, 0:2048], src=sbuf_prod[0:128, i_d1_0, 0:2048])
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
                # move
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
        nisa.tensor_copy(sbuf_prod[0:128, i_d1_0, 0:2048], psum_acc[0:128, i_d1_0, 0:2048])
    for i_d1_0 in range(16):
        nisa.dma_copy(dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, 0:2048], src=sbuf_prod[0:128, i_d1_0, 0:2048])
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
        nisa.tensor_copy(sbuf_prod[0:128, i_d1_0, 0:2048], psum_acc[0:128, i_d1_0, 0:2048])
    for i_d1_0 in range(16):
        nisa.dma_copy(dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, 0:2048], src=sbuf_prod[0:128, i_d1_0, 0:2048])
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
            # move
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
        nisa.tensor_copy(sbuf_prod[0:128, i_d1_0, 0:2048], psum_acc[0:128, i_d1_0, 0:2048])
    for i_d1_0 in range(16):
        nisa.dma_copy(dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, 0:2048], src=sbuf_prod[0:128, i_d1_0, 0:2048])
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
        nisa.dma_copy(dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, 0:2048], src=sbuf_prod[0:128, i_d1_0, 0:2048])
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
            # reverse_move
            nisa.tensor_copy(sbuf_prod[0:128, i_d1_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512], psum_acc[0:128, 0, 0:512])
    for i_d1_0 in range(16):
        nisa.dma_copy(dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, 0:2048], src=sbuf_prod[0:128, i_d1_0, 0:2048])
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
            # reverse_move
            nisa.dma_copy(
                dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                src=sbuf_prod[0:128, 0, 0:512],
            )
    return hbm_out


@nki.jit
def kernel_15(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d0_0 in range(16):
        nisa.dma_copy(src=lhs_T[i_d0_0 * 128:i_d0_0 * 128 + 128, 0:0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0:0 + 2048])
    for i_d0_0 in range(16):
        nisa.dma_copy(src=rhs[i_d0_0 * 128:i_d0_0 * 128 + 128, 0:0 + 2048], dst=sbuf_rhs[0:128, i_d0_0, 0:0 + 2048])
    psum_prod = nl.ndarray((128, 2, 2048), dtype=nl.float32, buffer=nl.psum)
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[0:128, i_d1_0 % 2, 0:0 + 2048], value=0.0)
        for i_d2_0 in range(4):
            for i_d0_0 in range(16):
                nisa.nc_matmul(stationary=sbuf_lhs_T[0:128, i_d0_0, i_d1_0 * 128:i_d1_0 * 128 + 128], moving=sbuf_rhs[0:128, i_d0_0, i_d2_0 * 512:i_d2_0 * 512 + 512], dst=psum_prod[0:128, i_d1_0 % 2, i_d2_0 * 512:i_d2_0 * 512 + 512])
        nisa.tensor_copy(src=psum_prod[0:128, i_d1_0 % 2, 0:0 + 2048], dst=sbuf_prod[0:128, i_d1_0, 0:0 + 2048])
    for i_d1_0 in range(16):
        nisa.dma_copy(src=sbuf_prod[0:128, i_d1_0, 0:0 + 2048], dst=hbm_out[i_d1_0 * 128:i_d1_0 * 128 + 128, 0:0 + 2048])
    return hbm_out


@nki.jit
def kernel_partial(lhs_T, rhs):
    # partial-coverage byte oracle: lhs_T load (d1 range16) sunk under matmul
    # outer-d1 range(4); residual range(4) sweeps the 4 inner d1 tiles, M tile
    # = i_d1_0 * 4 + i_d1_1; sbuf_lhs_T holds those 4 tiles (128, 1, 512).
    sbuf_lhs_T = nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    psum_acc = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d0_0 in range(16):
        nisa.dma_copy(dst=sbuf_rhs[0:128, i_d0_0, 0:2048], src=rhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, 0:2048])
    for i_d1_0 in range(16):
        nisa.memset(psum_acc[0:128, i_d1_0, 0:2048], value=0.0)
    for i_d0_0 in range(16):
        for i_d1_0 in range(4):
            # move (partial cover: residual range(4) over the 4 inner d1 tiles)
            for i_d1_1 in range(4):
                nisa.dma_copy(
                    dst=sbuf_lhs_T[0:128, 0, (i_d1_1) * 128 : (i_d1_1) * 128 + 128],
                    src=lhs_T[
                        (i_d0_0) * 128 : (i_d0_0) * 128 + 128,
                        (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                    ],
                )
            for i_d1_1 in range(4):
                for i_d2_0 in range(4):
                    nisa.nc_matmul(
                        dst=psum_acc[0:128, i_d1_0 * 4 + i_d1_1, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                        stationary=sbuf_lhs_T[0:128, 0, (i_d1_1) * 128 : (i_d1_1) * 128 + 128],
                        moving=sbuf_rhs[0:128, i_d0_0, (i_d2_0) * 512 : (i_d2_0) * 512 + 512],
                    )
    for i_d1_0 in range(16):
        nisa.tensor_copy(sbuf_prod[0:128, i_d1_0, 0:2048], psum_acc[0:128, i_d1_0, 0:2048])
    for i_d1_0 in range(16):
        nisa.dma_copy(dst=hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, 0:2048], src=sbuf_prod[0:128, i_d1_0, 0:2048])
    return hbm_out


def _main() -> None:
    """CPU-sim every ``kernel_N`` against a single numpy matmul golden."""
    K, M, N = 2048, 2048, 2048
    rng = np.random.default_rng(0)
    lhs_T = rng.standard_normal((K, M)).astype(np.float32)
    rhs = rng.standard_normal((K, N)).astype(np.float32)
    expected = lhs_T.T @ rhs

    atol = rtol = 5e-3
    kernels = [(name, obj) for name, obj in globals().items() if name.startswith("kernel_") and callable(obj)]

    def _order(name: str) -> tuple[int, int, str]:
        """Sort numeric ``kernel_N`` ascending, then named kernels alphabetically."""
        suffix = name.split("_", 1)[1]
        return (0, int(suffix), "") if suffix.isdigit() else (1, 0, suffix)

    kernels.sort(key=lambda nv: _order(nv[0]))
    for name, kernel in kernels:
        actual = simulate_fp32(kernel)(lhs_T, rhs)
        if isinstance(actual, tuple):
            actual = actual[0]
        actual = np.asarray(actual)
        max_abs = float(np.abs(actual - expected).max())
        max_rel = float((np.abs(actual - expected) / (np.abs(expected) + atol)).max())
        ok = np.allclose(actual, expected, atol=atol, rtol=rtol)
        print(f"{name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} pass={ok}")


if __name__ == "__main__":
    _main()
