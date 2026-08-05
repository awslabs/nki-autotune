"""Explicit NKI ladder for ``lhs @ rhs`` with lhs transpose materialization.

The 32 hand-written kernels are test fixtures for byte-exact transform
verification. ``kernel_31`` is the 87.237% MFU endpoint.
"""

import nki
import nki.isa as nisa
import nki.language as nl


# dma_k00_canonical
@nki.jit
def kernel_0(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.psum) for _ in range(1)]
    for i_d0_0 in range(16):
        for i_d1_0 in range(16):
            nisa.nc_transpose(
                data=sbuf_lhs[0][0:128, i_d0_0, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=psum_lhs_T[0][0:128, i_d1_0, i_d0_0 * 128 : i_d0_0 * 128 + 128],
            )
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.tensor_copy(src=psum_lhs_T[0][0:128, i_d1_0, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d1_0, 0 : 0 + 2048])
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d1_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.memset(dst=psum_prod[0][0:128, i_d0_0, 0 : 0 + 2048], value=0.0)
    for i_d1_0 in range(16):
        for i_d0_0 in range(16):
            for i_d2_0 in range(4):
                nisa.nc_matmul(
                    stationary=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 128 : i_d0_0 * 128 + 128],
                    moving=sbuf_rhs[0][0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                    dst=psum_prod[0][0:128, i_d0_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                )
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.tensor_copy(src=psum_prod[0][0:128, i_d0_0, 0 : 0 + 2048], dst=sbuf_prod[0][0:128, i_d0_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d0_0, 0 : 0 + 2048], dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


# dma_k01_transpose_through_load
@nki.jit
def kernel_1(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d1_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.memset(dst=psum_prod[0][0:128, i_d0_0, 0 : 0 + 2048], value=0.0)
    for i_d1_0 in range(16):
        for i_d0_0 in range(16):
            for i_d2_0 in range(4):
                nisa.nc_matmul(
                    stationary=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 128 : i_d0_0 * 128 + 128],
                    moving=sbuf_rhs[0][0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                    dst=psum_prod[0][0:128, i_d0_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                )
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.tensor_copy(src=psum_prod[0][0:128, i_d0_0, 0 : 0 + 2048], dst=sbuf_prod[0][0:128, i_d0_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d0_0, 0 : 0 + 2048], dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


# dma_k02_matmul_reorder_mn
@nki.jit
def kernel_2(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d1_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.memset(dst=psum_prod[0][0:128, i_d0_0, 0 : 0 + 2048], value=0.0)
    for i_d1_0 in range(16):
        for i_d2_0 in range(4):
            for i_d0_0 in range(16):
                nisa.nc_matmul(
                    stationary=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 128 : i_d0_0 * 128 + 128],
                    moving=sbuf_rhs[0][0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                    dst=psum_prod[0][0:128, i_d0_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                )
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.tensor_copy(src=psum_prod[0][0:128, i_d0_0, 0 : 0 + 2048], dst=sbuf_prod[0][0:128, i_d0_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d0_0, 0 : 0 + 2048], dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


# dma_k03_matmul_reorder_kn
@nki.jit
def kernel_3(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d1_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.memset(dst=psum_prod[0][0:128, i_d0_0, 0 : 0 + 2048], value=0.0)
    for i_d2_0 in range(4):
        for i_d1_0 in range(16):
            for i_d0_0 in range(16):
                nisa.nc_matmul(
                    stationary=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 128 : i_d0_0 * 128 + 128],
                    moving=sbuf_rhs[0][0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                    dst=psum_prod[0][0:128, i_d0_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                )
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.tensor_copy(src=psum_prod[0][0:128, i_d0_0, 0 : 0 + 2048], dst=sbuf_prod[0][0:128, i_d0_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d0_0, 0 : 0 + 2048], dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


# dma_k04_matmul_split_k
@nki.jit
def kernel_4(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d1_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.memset(dst=psum_prod[0][0:128, i_d0_0, 0 : 0 + 2048], value=0.0)
    for i_d2_0 in range(4):
        for i_d1_0 in range(2):
            for i_d1_1 in range(8):
                for i_d0_0 in range(16):
                    nisa.nc_matmul(
                        stationary=sbuf_lhs_T[0][0:128, i_d1_0 * 8 + i_d1_1, i_d0_0 * 128 : i_d0_0 * 128 + 128],
                        moving=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        dst=psum_prod[0][0:128, i_d0_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                    )
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.tensor_copy(src=psum_prod[0][0:128, i_d0_0, 0 : 0 + 2048], dst=sbuf_prod[0][0:128, i_d0_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d0_0, 0 : 0 + 2048], dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


# dma_k05_matmul_split_m
@nki.jit
def kernel_5(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d1_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.memset(dst=psum_prod[0][0:128, i_d0_0, 0 : 0 + 2048], value=0.0)
    for i_d2_0 in range(4):
        for i_d1_0 in range(2):
            for i_d1_1 in range(8):
                for i_d0_0 in range(4):
                    for i_d0_1 in range(4):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[0][0:128, i_d0_0 * 4 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.tensor_copy(src=psum_prod[0][0:128, i_d0_0, 0 : 0 + 2048], dst=sbuf_prod[0][0:128, i_d0_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d0_0, 0 : 0 + 2048], dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


# dma_k06_matmul_reorder_ki_mo
@nki.jit
def kernel_6(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d1_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.memset(dst=psum_prod[0][0:128, i_d0_0, 0 : 0 + 2048], value=0.0)
    for i_d2_0 in range(4):
        for i_d1_0 in range(2):
            for i_d0_0 in range(4):
                for i_d1_1 in range(8):
                    for i_d0_1 in range(4):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[0][0:128, i_d0_0 * 4 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.tensor_copy(src=psum_prod[0][0:128, i_d0_0, 0 : 0 + 2048], dst=sbuf_prod[0][0:128, i_d0_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d0_0, 0 : 0 + 2048], dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


# dma_k07_matmul_reorder_ki_mi
@nki.jit
def kernel_7(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d1_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.memset(dst=psum_prod[0][0:128, i_d0_0, 0 : 0 + 2048], value=0.0)
    for i_d2_0 in range(4):
        for i_d1_0 in range(2):
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[0][0:128, i_d0_0 * 4 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.tensor_copy(src=psum_prod[0][0:128, i_d0_0, 0 : 0 + 2048], dst=sbuf_prod[0][0:128, i_d0_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d0_0, 0 : 0 + 2048], dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


# dma_k08_layout_psum
@nki.jit
def kernel_8(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d1_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d0_0 in range(16):
        nisa.memset(dst=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 2048], value=0.0)
    for i_d2_0 in range(4):
        for i_d1_0 in range(2):
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.tensor_copy(src=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 2048], dst=sbuf_prod[0][0:128, i_d0_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d0_0, 0 : 0 + 2048], dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


# dma_k09_split_product_drain_n
@nki.jit
def kernel_9(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d1_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d0_0 in range(16):
        nisa.memset(dst=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 2048], value=0.0)
    for i_d2_0 in range(4):
        for i_d1_0 in range(2):
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        for i_d2_0 in range(4):
            nisa.tensor_copy(
                src=psum_prod[i_d0_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[0][0:128, i_d0_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d0_0, 0 : 0 + 2048], dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


# dma_k10_reorder_product_drain
@nki.jit
def kernel_10(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d1_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d0_0 in range(16):
        nisa.memset(dst=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 2048], value=0.0)
    for i_d2_0 in range(4):
        for i_d1_0 in range(2):
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d2_0 in range(4):
        for i_d0_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d0_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[0][0:128, i_d0_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d0_0, 0 : 0 + 2048], dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


# dma_k11_sink_product_drain
@nki.jit
def kernel_11(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d1_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d0_0 in range(16):
        nisa.memset(dst=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 2048], value=0.0)
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d2_0 in range(4):
        for i_d1_0 in range(2):
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        for i_d0_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d0_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[0][0:128, i_d0_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d0_0, 0 : 0 + 2048], dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


# dma_k12_split_store_n
@nki.jit
def kernel_12(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d1_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d0_0 in range(16):
        nisa.memset(dst=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 2048], value=0.0)
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d2_0 in range(4):
        for i_d1_0 in range(2):
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        for i_d0_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d0_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[0][0:128, i_d0_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d0_0 in range(16):
        for i_d2_0 in range(4):
            nisa.dma_copy(
                src=sbuf_prod[0][0:128, i_d0_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


# dma_k13_reorder_store
@nki.jit
def kernel_13(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d1_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d0_0 in range(16):
        nisa.memset(dst=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 2048], value=0.0)
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d2_0 in range(4):
        for i_d1_0 in range(2):
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        for i_d0_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d0_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[0][0:128, i_d0_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        for i_d0_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[0][0:128, i_d0_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


# dma_k14_sink_store
@nki.jit
def kernel_14(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d1_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d0_0 in range(16):
        nisa.memset(dst=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 2048], value=0.0)
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        for i_d1_0 in range(2):
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        for i_d0_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d0_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[0][0:128, i_d0_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
        for i_d0_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[0][0:128, i_d0_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


# dma_k15_compact_product_sbuf
@nki.jit
def kernel_15(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d1_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d0_0 in range(16):
        nisa.memset(dst=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 2048], value=0.0)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        for i_d1_0 in range(2):
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        sbuf_prod = [nl.ndarray((128, 16, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
        for i_d0_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d0_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[0][0:128, i_d0_0, 0 : 0 + 512],
            )
        for i_d0_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[0][0:128, i_d0_0, 0 : 0 + 512],
                dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


# dma_k16_layout_product_sbuf
@nki.jit
def kernel_16(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d1_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d0_0 in range(16):
        nisa.memset(dst=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 2048], value=0.0)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        for i_d1_0 in range(2):
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d0_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512],
            )
        for i_d0_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512],
                dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


# dma_k17_split_product_memset_n
@nki.jit
def kernel_17(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d1_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d0_0 in range(16):
        for i_d2_0 in range(4):
            nisa.memset(dst=psum_prod[i_d0_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512], value=0.0)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        for i_d1_0 in range(2):
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d0_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512],
            )
        for i_d0_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512],
                dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


# dma_k18_reorder_product_memset
@nki.jit
def kernel_18(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d1_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d2_0 in range(4):
        for i_d0_0 in range(16):
            nisa.memset(dst=psum_prod[i_d0_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512], value=0.0)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        for i_d1_0 in range(2):
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d0_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512],
            )
        for i_d0_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512],
                dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


# dma_k19_sink_product_memset
@nki.jit
def kernel_19(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d1_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        for i_d0_0 in range(16):
            nisa.memset(dst=psum_prod[i_d0_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512], value=0.0)
        for i_d1_0 in range(2):
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d0_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512],
            )
        for i_d0_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512],
                dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


# dma_k20_compact_product_psum
@nki.jit
def kernel_20(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d1_0, 0 : 0 + 2048]
        )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.memset(dst=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 512], value=0.0)
        for i_d1_0 in range(2):
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, 0 : 0 + 512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.tensor_copy(src=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 512], dst=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512])
        for i_d0_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512],
                dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


# dma_k21_split_rhs_k
@nki.jit
def kernel_21(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(2):
        for i_d1_1 in range(8):
            nisa.dma_copy(
                src=rhs[i_d1_0 * 1024 + i_d1_1 * 128 : i_d1_0 * 1024 + i_d1_1 * 128 + 128, 0 : 0 + 2048],
                dst=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, 0 : 0 + 2048],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.memset(dst=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 512], value=0.0)
        for i_d1_0 in range(2):
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, 0 : 0 + 512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.tensor_copy(src=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 512], dst=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512])
        for i_d0_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512],
                dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


# dma_k22_split_rhs_n
@nki.jit
def kernel_22(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(2):
        for i_d1_1 in range(8):
            for i_d2_0 in range(4):
                nisa.dma_copy(
                    src=rhs[
                        i_d1_0 * 1024 + i_d1_1 * 128 : i_d1_0 * 1024 + i_d1_1 * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.memset(dst=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 512], value=0.0)
        for i_d1_0 in range(2):
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, 0 : 0 + 512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.tensor_copy(src=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 512], dst=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512])
        for i_d0_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512],
                dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


# dma_k23_reorder_rhs_ki_n
@nki.jit
def kernel_23(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(2):
        for i_d2_0 in range(4):
            for i_d1_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        i_d1_0 * 1024 + i_d1_1 * 128 : i_d1_0 * 1024 + i_d1_1 * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.memset(dst=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 512], value=0.0)
        for i_d1_0 in range(2):
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, 0 : 0 + 512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.tensor_copy(src=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 512], dst=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512])
        for i_d0_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512],
                dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


# dma_k24_reorder_rhs_ko_n
@nki.jit
def kernel_24(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d2_0 in range(4):
        for i_d1_0 in range(2):
            for i_d1_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        i_d1_0 * 1024 + i_d1_1 * 128 : i_d1_0 * 1024 + i_d1_1 * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.memset(dst=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 512], value=0.0)
        for i_d1_0 in range(2):
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, 0 : 0 + 512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.tensor_copy(src=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 512], dst=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512])
        for i_d0_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512],
                dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


# dma_k25_sink_rhs
@nki.jit
def kernel_25(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.memset(dst=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 512], value=0.0)
        for i_d1_0 in range(2):
            for i_d1_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        i_d1_0 * 1024 + i_d1_1 * 128 : i_d1_0 * 1024 + i_d1_1 * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                )
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d1_0 * 8 + i_d1_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, 0 : 0 + 512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.tensor_copy(src=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 512], dst=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512])
        for i_d0_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512],
                dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


# dma_k26_compact_rhs
@nki.jit
def kernel_26(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.memset(dst=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 512], value=0.0)
        for i_d1_0 in range(2):
            sbuf_rhs = [nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
            for i_d1_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        i_d1_0 * 1024 + i_d1_1 * 128 : i_d1_0 * 1024 + i_d1_1 * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[0][0:128, i_d1_1, 0 : 0 + 512],
                )
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d1_1, 0 : 0 + 512],
                            dst=psum_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, 0 : 0 + 512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.tensor_copy(src=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 512], dst=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512])
        for i_d0_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512],
                dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


# dma_k27_layout_rhs
@nki.jit
def kernel_27(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.memset(dst=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 512], value=0.0)
        for i_d1_0 in range(2):
            sbuf_rhs = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)]
            for i_d1_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        i_d1_0 * 1024 + i_d1_1 * 128 : i_d1_0 * 1024 + i_d1_1 * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[i_d1_1][0:128, 0, 0 : 0 + 512],
                )
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[i_d1_1][0:128, 0, 0 : 0 + 512],
                            dst=psum_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, 0 : 0 + 512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.tensor_copy(src=psum_prod[i_d0_0][0:128, 0, 0 : 0 + 512], dst=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512])
        for i_d0_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512],
                dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


# dma_k28_rfactor_k
@nki.jit
def kernel_28(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.memset(dst=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512], value=0.0)
        for i_d1_0 in range(2):
            sbuf_rhs = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)]
            for i_d1_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        i_d1_0 * 1024 + i_d1_1 * 128 : i_d1_0 * 1024 + i_d1_1 * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[i_d1_1][0:128, 0, 0 : 0 + 512],
                )
            psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
            sbuf_rfactor = [nl.ndarray((128, 16, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    nisa.memset(dst=psum_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, 0 : 0 + 512], value=0.0)
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[i_d1_1][0:128, 0, 0 : 0 + 512],
                            dst=psum_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, 0 : 0 + 512],
                        )
                    nisa.tensor_copy(
                        src=psum_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, 0 : 0 + 512],
                        dst=sbuf_rfactor[0][0:128, i_d0_0 * 4 + i_d0_1, 0 : 0 + 512],
                    )
                    nisa.tensor_tensor(
                        data1=sbuf_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, 0 : 0 + 512],
                        data2=sbuf_rfactor[0][0:128, i_d0_0 * 4 + i_d0_1, 0 : 0 + 512],
                        dst=sbuf_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, 0 : 0 + 512],
                        op=nl.add,
                    )
        for i_d0_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512],
                dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


# dma_k29_compact_rfactor_psum
@nki.jit
def kernel_29(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.memset(dst=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512], value=0.0)
        for i_d1_0 in range(2):
            sbuf_rhs = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)]
            for i_d1_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        i_d1_0 * 1024 + i_d1_1 * 128 : i_d1_0 * 1024 + i_d1_1 * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[i_d1_1][0:128, 0, 0 : 0 + 512],
                )
            sbuf_rfactor = [nl.ndarray((128, 16, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]
                    nisa.memset(dst=psum_prod[0][0:128, 0, 0 : 0 + 512], value=0.0)
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[i_d1_1][0:128, 0, 0 : 0 + 512],
                            dst=psum_prod[0][0:128, 0, 0 : 0 + 512],
                        )
                    nisa.tensor_copy(
                        src=psum_prod[0][0:128, 0, 0 : 0 + 512],
                        dst=sbuf_rfactor[0][0:128, i_d0_0 * 4 + i_d0_1, 0 : 0 + 512],
                    )
                    nisa.tensor_tensor(
                        data1=sbuf_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, 0 : 0 + 512],
                        data2=sbuf_rfactor[0][0:128, i_d0_0 * 4 + i_d0_1, 0 : 0 + 512],
                        dst=sbuf_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, 0 : 0 + 512],
                        op=nl.add,
                    )
        for i_d0_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512],
                dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


# dma_k30_compact_rfactor_sbuf
@nki.jit
def kernel_30(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[0][0:128, i_d1_0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.memset(dst=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512], value=0.0)
        for i_d1_0 in range(2):
            sbuf_rhs = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)]
            for i_d1_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        i_d1_0 * 1024 + i_d1_1 * 128 : i_d1_0 * 1024 + i_d1_1 * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[i_d1_1][0:128, 0, 0 : 0 + 512],
                )
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]
                    nisa.memset(dst=psum_prod[0][0:128, 0, 0 : 0 + 512], value=0.0)
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d1_0 * 8 + i_d1_1,
                                i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128,
                            ],
                            moving=sbuf_rhs[i_d1_1][0:128, 0, 0 : 0 + 512],
                            dst=psum_prod[0][0:128, 0, 0 : 0 + 512],
                        )
                    sbuf_rfactor = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
                    nisa.tensor_copy(
                        src=psum_prod[0][0:128, 0, 0 : 0 + 512], dst=sbuf_rfactor[0][0:128, 0, 0 : 0 + 512]
                    )
                    nisa.tensor_tensor(
                        data1=sbuf_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, 0 : 0 + 512],
                        data2=sbuf_rfactor[0][0:128, 0, 0 : 0 + 512],
                        dst=sbuf_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, 0 : 0 + 512],
                        op=nl.add,
                    )
        for i_d0_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512],
                dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


# dma_k31_layout_dma
@nki.jit
def kernel_31(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = [nl.ndarray((128, 1, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
    for i_d0_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_transpose(
                src=lhs[i_d0_0 * 512 : i_d0_0 * 512 + 512, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                dst=sbuf_lhs_T[i_d1_0][0:128, 0, i_d0_0 * 512 : i_d0_0 * 512 + 512],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d0_0 in range(16):
            nisa.memset(dst=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512], value=0.0)
        for i_d1_0 in range(2):
            sbuf_rhs = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)]
            for i_d1_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        i_d1_0 * 1024 + i_d1_1 * 128 : i_d1_0 * 1024 + i_d1_1 * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[i_d1_1][0:128, 0, 0 : 0 + 512],
                )
            for i_d0_0 in range(4):
                for i_d0_1 in range(4):
                    psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]
                    nisa.memset(dst=psum_prod[0][0:128, 0, 0 : 0 + 512], value=0.0)
                    for i_d1_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[i_d1_0 * 8 + i_d1_1][
                                0:128, 0, i_d0_0 * 512 + i_d0_1 * 128 : i_d0_0 * 512 + i_d0_1 * 128 + 128
                            ],
                            moving=sbuf_rhs[i_d1_1][0:128, 0, 0 : 0 + 512],
                            dst=psum_prod[0][0:128, 0, 0 : 0 + 512],
                        )
                    sbuf_rfactor = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
                    nisa.tensor_copy(
                        src=psum_prod[0][0:128, 0, 0 : 0 + 512], dst=sbuf_rfactor[0][0:128, 0, 0 : 0 + 512]
                    )
                    nisa.tensor_tensor(
                        data1=sbuf_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, 0 : 0 + 512],
                        data2=sbuf_rfactor[0][0:128, 0, 0 : 0 + 512],
                        dst=sbuf_prod[i_d0_0 * 4 + i_d0_1][0:128, 0, 0 : 0 + 512],
                        op=nl.add,
                    )
        for i_d0_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d0_0][0:128, 0, 0 : 0 + 512],
                dst=hbm_out[i_d0_0 * 128 : i_d0_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out
