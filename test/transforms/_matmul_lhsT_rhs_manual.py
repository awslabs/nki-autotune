"""Explicit NKI ladder for the canonical ``lhs_T.T @ rhs`` workload.

The 36 hand-written kernels are test fixtures for byte-exact transform
verification. ``kernel_35`` is the 90.92% MFU endpoint.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def kernel_0(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[0][0:128, i_d1_0, 0 : 0 + 2048], value=0.0)
    for i_d0_0 in range(16):
        for i_d1_0 in range(16):
            for i_d2_0 in range(4):
                nisa.nc_matmul(
                    stationary=sbuf_lhs_T[0][0:128, i_d0_0, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                    moving=sbuf_rhs[0][0:128, i_d0_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                    dst=psum_prod[0][0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                )
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.tensor_copy(src=psum_prod[0][0:128, i_d1_0, 0 : 0 + 2048], dst=sbuf_prod[0][0:128, i_d1_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d1_0, 0 : 0 + 2048], dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


@nki.jit
def kernel_1(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[0][0:128, i_d1_0, 0 : 0 + 2048], value=0.0)
    # Reorder
    for i_d0_0 in range(16):
        for i_d2_0 in range(4):
            for i_d1_0 in range(16):
                nisa.nc_matmul(
                    stationary=sbuf_lhs_T[0][0:128, i_d0_0, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                    moving=sbuf_rhs[0][0:128, i_d0_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                    dst=psum_prod[0][0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                )
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.tensor_copy(src=psum_prod[0][0:128, i_d1_0, 0 : 0 + 2048], dst=sbuf_prod[0][0:128, i_d1_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d1_0, 0 : 0 + 2048], dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


@nki.jit
def kernel_2(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[0][0:128, i_d1_0, 0 : 0 + 2048], value=0.0)
    # Reorder
    for i_d2_0 in range(4):
        for i_d0_0 in range(16):
            for i_d1_0 in range(16):
                nisa.nc_matmul(
                    stationary=sbuf_lhs_T[0][0:128, i_d0_0, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                    moving=sbuf_rhs[0][0:128, i_d0_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                    dst=psum_prod[0][0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                )
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.tensor_copy(src=psum_prod[0][0:128, i_d1_0, 0 : 0 + 2048], dst=sbuf_prod[0][0:128, i_d1_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d1_0, 0 : 0 + 2048], dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


@nki.jit
def kernel_3(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[0][0:128, i_d1_0, 0 : 0 + 2048], value=0.0)
    for i_d2_0 in range(4):
        # Split
        for i_d0_0 in range(2):
            for i_d0_1 in range(8):
                for i_d1_0 in range(16):
                    nisa.nc_matmul(
                        stationary=sbuf_lhs_T[0][0:128, i_d0_0 * 8 + i_d0_1, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                        moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        dst=psum_prod[0][0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                    )
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.tensor_copy(src=psum_prod[0][0:128, i_d1_0, 0 : 0 + 2048], dst=sbuf_prod[0][0:128, i_d1_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d1_0, 0 : 0 + 2048], dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


@nki.jit
def kernel_4(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[0][0:128, i_d1_0, 0 : 0 + 2048], value=0.0)
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d0_1 in range(8):
                # Split
                for i_d1_0 in range(4):
                    for i_d1_1 in range(4):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[0][0:128, (i_d1_0 * 4 + i_d1_1), i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.tensor_copy(src=psum_prod[0][0:128, i_d1_0, 0 : 0 + 2048], dst=sbuf_prod[0][0:128, i_d1_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d1_0, 0 : 0 + 2048], dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


@nki.jit
def kernel_5(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[0][0:128, i_d1_0, 0 : 0 + 2048], value=0.0)
    # Reorder
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d0_1 in range(8):
                    for i_d1_1 in range(4):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[0][0:128, (i_d1_0 * 4 + i_d1_1), i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.tensor_copy(src=psum_prod[0][0:128, i_d1_0, 0 : 0 + 2048], dst=sbuf_prod[0][0:128, i_d1_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d1_0, 0 : 0 + 2048], dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


@nki.jit
def kernel_6(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[0][0:128, i_d1_0, 0 : 0 + 2048], value=0.0)
    # Reorder
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[0][0:128, (i_d1_0 * 4 + i_d1_1), i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.tensor_copy(src=psum_prod[0][0:128, i_d1_0, 0 : 0 + 2048], dst=sbuf_prod[0][0:128, i_d1_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d1_0, 0 : 0 + 2048], dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


@nki.jit
def kernel_7(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0 : 0 + 2048]
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
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.tensor_copy(src=psum_prod[i_d1_0][0:128, 0, 0 : 0 + 2048], dst=sbuf_prod[0][0:128, i_d1_0, 0 : 0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d1_0, 0 : 0 + 2048], dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


@nki.jit
def kernel_8(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0 : 0 + 2048]
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
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    # Split
    for i_d1_0 in range(16):
        for i_d2_0 in range(4):
            nisa.tensor_copy(
                src=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[0][0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d1_0, 0 : 0 + 2048], dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


@nki.jit
def kernel_9(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0 : 0 + 2048]
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
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    # Reorder
    for i_d2_0 in range(4):
        for i_d1_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[0][0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d1_0, 0 : 0 + 2048], dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


@nki.jit
def kernel_10(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0 : 0 + 2048], value=0.0)
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        # Code motion
        for i_d1_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[0][0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d1_0, 0 : 0 + 2048], dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out


@nki.jit
def kernel_11(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0 : 0 + 2048], value=0.0)
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        for i_d1_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[0][0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    # Split
    for i_d1_0 in range(16):
        for i_d2_0 in range(4):
            nisa.dma_copy(
                src=sbuf_prod[0][0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


@nki.jit
def kernel_12(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0 : 0 + 2048], value=0.0)
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        for i_d1_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[0][0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    # Reorder
    for i_d2_0 in range(4):
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[0][0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


@nki.jit
def kernel_13(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0 : 0 + 2048], value=0.0)
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        for i_d1_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[0][0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
        # Code motion
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[0][0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


@nki.jit
def kernel_14(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0 : 0 + 2048]
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
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        # Buffer compaction
        sbuf_prod = [nl.ndarray((128, 16, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
        for i_d1_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[0][0:128, i_d1_0, 0:512],
            )
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[0][0:128, i_d1_0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


@nki.jit
def kernel_15(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0 : 0 + 2048]
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
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
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
def kernel_16(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0 : 0 + 2048]
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
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
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
def kernel_17(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0 : 0 + 2048]
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
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
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
def kernel_18(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        # Code motion
        for i_d1_0 in range(16):
            nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512], value=0.0)
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
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
def kernel_19(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        # Buffer compaction
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0:512], value=0.0)
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
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

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    # Split
    for i_d0_0 in range(2):
        for i_d0_1 in range(8):
            nisa.dma_copy(
                src=rhs[(i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128, 0 : 0 + 2048],
                dst=sbuf_rhs[0][0:128, (i_d0_0 * 8 + i_d0_1), 0 : 0 + 2048],
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
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
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

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    # Split
    for i_d0_0 in range(2):
        for i_d0_1 in range(8):
            for i_d2_0 in range(4):
                nisa.dma_copy(
                    src=rhs[
                        (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[0][0:128, (i_d0_0 * 8 + i_d0_1), i_d2_0 * 512 : i_d2_0 * 512 + 512],
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
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
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

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    # Reorder
    for i_d0_0 in range(2):
        for i_d2_0 in range(4):
            for i_d0_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[0][0:128, (i_d0_0 * 8 + i_d0_1), i_d2_0 * 512 : i_d2_0 * 512 + 512],
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
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
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

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    # Reorder
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d0_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[0][0:128, (i_d0_0 * 8 + i_d0_1), i_d2_0 * 512 : i_d2_0 * 512 + 512],
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
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
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

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0:512], value=0.0)
        for i_d0_0 in range(2):
            # Code motion
            for i_d0_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[0][0:128, (i_d0_0 * 8 + i_d0_1), i_d2_0 * 512 : i_d2_0 * 512 + 512],
                )
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
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

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )

    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0:512], value=0.0)
        for i_d0_0 in range(2):
            # Buffer compaction
            sbuf_rhs = [nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
            for i_d0_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[0][0:128, i_d0_1, 0:512],
                )
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_1, 0:512],
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

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
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
                            stationary=sbuf_lhs_T[0][
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
def kernel_27(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    # Split
    for i_d0_0 in range(16):
        for i_d1_0 in range(4):
            nisa.dma_copy(
                src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, i_d1_0 * 512 : i_d1_0 * 512 + 512],
                dst=sbuf_lhs_T[0][0:128, i_d0_0, i_d1_0 * 512 : i_d1_0 * 512 + 512],
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
                            stationary=sbuf_lhs_T[0][
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
def kernel_28(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    # Split
    for i_d0_0 in range(2):
        for i_d0_1 in range(8):
            for i_d1_0 in range(4):
                nisa.dma_copy(
                    src=lhs_T[
                        (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                        i_d1_0 * 512 : i_d1_0 * 512 + 512,
                    ],
                    dst=sbuf_lhs_T[0][0:128, (i_d0_0 * 8 + i_d0_1), i_d1_0 * 512 : i_d1_0 * 512 + 512],
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
                            stationary=sbuf_lhs_T[0][
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
def kernel_29(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    # Reorder
    for i_d0_0 in range(2):
        for i_d1_0 in range(4):
            for i_d0_1 in range(8):
                nisa.dma_copy(
                    src=lhs_T[
                        (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                        i_d1_0 * 512 : i_d1_0 * 512 + 512,
                    ],
                    dst=sbuf_lhs_T[0][0:128, (i_d0_0 * 8 + i_d0_1), i_d1_0 * 512 : i_d1_0 * 512 + 512],
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
                            stationary=sbuf_lhs_T[0][
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
def kernel_30(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
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
                # Code motion
                for i_d0_1 in range(8):
                    nisa.dma_copy(
                        src=lhs_T[
                            (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                            i_d1_0 * 512 : i_d1_0 * 512 + 512,
                        ],
                        dst=sbuf_lhs_T[0][0:128, (i_d0_0 * 8 + i_d0_1), i_d1_0 * 512 : i_d1_0 * 512 + 512],
                    )
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
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
def kernel_31(lhs_T, rhs):
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
                # Buffer compaction
                sbuf_lhs_T = [nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
                for i_d0_1 in range(8):
                    nisa.dma_copy(
                        src=lhs_T[
                            (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                            i_d1_0 * 512 : i_d1_0 * 512 + 512,
                        ],
                        dst=sbuf_lhs_T[0][0:128, i_d0_1, 0:512],
                    )
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][0:128, i_d0_1, i_d1_1 * 128 : i_d1_1 * 128 + 128],
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
def kernel_32(lhs_T, rhs):
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
def kernel_33(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_d2_0 in range(4):
        # init_two_stage_0
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
            psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
            sbuf_rfactor = [nl.ndarray((128, 16, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
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
                    # init_two_stage_1
                    nisa.memset(dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512], value=0.0)
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[i_d0_1][0:128, 0, i_d1_1 * 128 : i_d1_1 * 128 + 128],
                            moving=sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        )
                    # drain_two_stage_0
                    nisa.tensor_copy(
                        src=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        dst=sbuf_rfactor[0][0:128, i_d1_0 * 4 + i_d1_1, 0:512],
                    )
                    nisa.tensor_tensor(
                        data1=sbuf_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        data2=sbuf_rfactor[0][0:128, i_d1_0 * 4 + i_d1_1, 0:512],
                        dst=sbuf_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        op=nl.add,
                    )
        # drain_two_stage_1: None
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d1_0][0:128, 0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


# Apply BufferCompaction to psum_prod
@nki.jit
def kernel_34(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_d2_0 in range(4):
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
            sbuf_rfactor = [nl.ndarray((128, 16, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
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
                    psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]
                    nisa.memset(dst=psum_prod[0][0:128, 0, 0:512], value=0.0)
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[i_d0_1][0:128, 0, i_d1_1 * 128 : i_d1_1 * 128 + 128],
                            moving=sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                            dst=psum_prod[0][0:128, 0, 0:512],
                        )
                    nisa.tensor_copy(
                        src=psum_prod[0][0:128, 0, 0:512], dst=sbuf_rfactor[0][0:128, i_d1_0 * 4 + i_d1_1, 0:512]
                    )
                    nisa.tensor_tensor(
                        data1=sbuf_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        data2=sbuf_rfactor[0][0:128, i_d1_0 * 4 + i_d1_1, 0:512],
                        dst=sbuf_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        op=nl.add,
                    )
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d1_0][0:128, 0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out


# Apply BufferCompaction to sbuf_rfactor
@nki.jit
def kernel_35(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_d2_0 in range(4):
        # init_two_stage_0
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
                    # init_two_stage_1
                    psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]
                    nisa.memset(dst=psum_prod[0][0:128, 0, 0:512], value=0.0)
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[i_d0_1][0:128, 0, i_d1_1 * 128 : i_d1_1 * 128 + 128],
                            moving=sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                            dst=psum_prod[0][0:128, 0, 0:512],
                        )
                    # drain_two_stage_0
                    sbuf_rfactor = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
                    nisa.tensor_copy(src=psum_prod[0][0:128, 0, 0:512], dst=sbuf_rfactor[0][0:128, 0, 0:512])
                    nisa.tensor_tensor(
                        data1=sbuf_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        data2=sbuf_rfactor[0][0:128, 0, 0:512],
                        dst=sbuf_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        op=nl.add,
                    )
        # drain_two_stage_1: None
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d1_0][0:128, 0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out
