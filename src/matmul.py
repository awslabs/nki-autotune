# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

import logging, math
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.compiler as ncc
from neuronxcc.nki.language import par_dim


def matmul(
    A_DRAM, B_DRAM, Z_DRAM, TILES_IN_BLOCK_K=8, TILES_IN_BLOCK_M=4, TILES_IN_BLOCK_N=4
):
    """
    Optimized matrix multiplication kernel

     Args:

        A_DRAM: an input tensor of shape [K, M], where K is a multiple of 1024
        and M is a multiple of 512.  It is the left-hand-side argument of the
        matrix multiplication, delivered transposed for optimal performance.

        B_DRAM: an input tensor of shape [K, N],  where K is a multiple of 1024
          and N is a multiple of 2048.  It is the right-hand-side argument of
          the matrix multiplication.

        Z_DRAM: the resulting output tensor of shape [M, N]

    """
    K, M = A_DRAM.shape
    _, N = B_DRAM.shape

    TILE_K = nl.tile_size.pmax
    TILE_M = nl.tile_size.gemm_stationary_fmax
    TILE_N = nl.tile_size.gemm_moving_fmax

    NUM_BLOCK_K = K // (TILES_IN_BLOCK_K * TILE_K)
    NUM_BLOCK_M = M // (TILES_IN_BLOCK_M * TILE_M)
    NUM_BLOCK_N = N // (TILES_IN_BLOCK_N * TILE_N)

    assert NUM_BLOCK_K * TILES_IN_BLOCK_K * TILE_K == K
    assert NUM_BLOCK_M * TILES_IN_BLOCK_M * TILE_M == M
    assert NUM_BLOCK_N * TILES_IN_BLOCK_N * TILE_N == N

    for n2 in nl.affine_range(NUM_BLOCK_N):
        for m2 in nl.affine_range(NUM_BLOCK_M):

            # Partition Z and then ensure that we are Z-block stationary
            # This way, no matter how large K, M, and N are, Z is never spilled/loaded
            # We only need to store once
            Z_SBUF = nl.zeros(
                (TILES_IN_BLOCK_M, nl.par_dim(TILE_M), TILES_IN_BLOCK_N * TILE_N),
                dtype=Z_DRAM.dtype,
                buffer=nl.sbuf,
            )

            for k2 in nl.affine_range(NUM_BLOCK_K):
                A_SBUF = nl.ndarray(
                    (TILES_IN_BLOCK_K, nl.par_dim(TILE_K), TILES_IN_BLOCK_M * TILE_M),
                    dtype=A_DRAM.dtype,
                    buffer=nl.sbuf,
                )
                B_SBUF = nl.ndarray(
                    (TILES_IN_BLOCK_K, nl.par_dim(TILE_K), TILES_IN_BLOCK_N * TILE_N),
                    dtype=B_DRAM.dtype,
                    buffer=nl.sbuf,
                )

                # Load in a block of A and a block of B
                for k1 in nl.affine_range(TILES_IN_BLOCK_K):
                    k_start = k2 * TILES_IN_BLOCK_K * TILE_K + k1 * TILE_K
                    k_end = k_start + TILE_K

                    m_start = m2 * TILES_IN_BLOCK_M * TILE_M
                    m_end = m_start + TILES_IN_BLOCK_M * TILE_M

                    n_start = n2 * TILES_IN_BLOCK_N * TILE_N
                    n_end = n_start + TILES_IN_BLOCK_N * TILE_N

                    # We coalesce memory accesses by loading TILES_IN_BLOCK_M * TILE_M
                    # values of A at a time. We cannot coalesce across K because K gets
                    # split across the partition dimension
                    A_SBUF[k1] = nl.load(A_DRAM[k_start:k_end, m_start:m_end])

                    # We coalesce memory accesses by loading TILES_IN_BLOCK_N * TILE_N
                    # values of B at a time. We cannot coalesce across K because K gets
                    # split across the partition dimension
                    B_SBUF[k1] = nl.load(B_DRAM[k_start:k_end, n_start:n_end])

                for m1 in nl.affine_range(TILES_IN_BLOCK_M):
                    for n1 in nl.affine_range(TILES_IN_BLOCK_N):
                        # Keep the tile of Z stationary in the PSUM buffer to minimize the
                        # number of calls to nl.loop_reduce
                        Z_PSUM = nl.zeros(
                            (TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum
                        )

                        m_start = m1 * TILE_M
                        m_end = m_start + TILE_M

                        n_start = n1 * TILE_N
                        n_end = n_start + TILE_N

                        for k1 in nl.affine_range(TILES_IN_BLOCK_K):
                            Z_PSUM += ni.nc_matmul(
                                A_SBUF[k1, :, m_start:m_end],
                                B_SBUF[k1, :, n_start:n_end],
                            )

                        Z_SBUF[m1, :, n_start:n_end] = nl.loop_reduce(
                            Z_PSUM, op=np.add, loop_indices=[k2], dtype=Z_DRAM.dtype
                        )

            for m1 in nl.affine_range(TILES_IN_BLOCK_M):
                m_start = m2 * TILES_IN_BLOCK_M * TILE_M + m1 * TILE_M
                m_end = m_start + TILE_M

                n_start = n2 * TILES_IN_BLOCK_N * TILE_N
                n_end = n_start + TILES_IN_BLOCK_N * TILE_N

                # We coalesce memory accesses by storing TILES_IN_BLOCK_N * TILE_N
                # values of Z at a time. We cannot coalesce across M because M gets
                # split across the partition dimension
                nl.store(Z_DRAM[m_start:m_end, n_start:n_end], value=Z_SBUF[m1])


# This is taken from the open source NKI samples repo
# https://github.com/aws-neuron/nki-samples/blob/main/src/tutorials/matrix_multiplication/matrix_multiplication_nki_kernels.py#L247
@nki.jit
def nki_matmul_fully_optimized_(
    lhsT,
    rhs,
    # Meta-parameters
    TILES_IN_BLOCK_M=16,
    TILES_IN_BLOCK_N=2,
    TILES_IN_BLOCK_K=8,
):
    """NKI kernel to compute a large matrix multiplication efficiently by
       blocking all dimensions and doing layout optimization.

    Args:
        lhsT: an input tensor of shape [K,M], where K is a multiple of 128 *
          TILES_IN_BLOCK_K and M is a multiple of 128 * TILES_IN_BLOCK_M.  It is the
          left-hand-side argument of the matrix multiplication, delivered transposed
          for optimal performance.
        rhs: an input tensor of shape [K,N],  where K is a multiple of 128 *
          TILES_IN_BLOCK_K and N is a multiple of 512 * TILES_IN_BLOCK_N.  It is
          the right-hand-side argument of the matrix multiplication.
        result: the resulting output tensor of shape [M,N]
        TILES_IN_BLOCK_*: meta parameters to control blocking dimensions
    """

    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, "lhsT and rhs must have the same contraction dimension"

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_K = nl.tile_size.pmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    BLOCK_M = TILE_M * TILES_IN_BLOCK_M
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N
    BLOCK_K = TILE_K * TILES_IN_BLOCK_K

    # the size has to be multiple of block size
    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

    NUM_BLOCK_M = M // BLOCK_M
    NUM_BLOCK_N = N // BLOCK_N
    NUM_BLOCK_K = K // BLOCK_K

    # Blocking N dimension (the RHS free dimension)
    for n in nl.affine_range(NUM_BLOCK_N):
        result_tiles = nl.zeros(
            (
                NUM_BLOCK_M,
                TILES_IN_BLOCK_M,
                TILES_IN_BLOCK_N,
                nl.par_dim(TILE_M),
                TILE_N,
            ),
            dtype=lhsT.dtype,
            buffer=nl.sbuf,
        )

        # Blocking K dimension (the contraction dimension)
        # Use `sequential_range` because we do not want the compiler to change this loop by,
        # for example, vectorizing it
        for k in nl.sequential_range(NUM_BLOCK_K):
            # Loading tiles from rhs
            # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
            i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
            rhs_tiles = nl.ndarray(
                (TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                dtype=rhs.dtype,
                buffer=nl.sbuf,
            )

            for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
                rhs_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
                    rhs[
                        (TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i_rhs.p,
                        BLOCK_N * n + i_rhs.x,
                    ]
                )

            # Blocking M dimension (the LHS free dimension)
            for m in nl.affine_range(NUM_BLOCK_M):
                # Loading tiles from lhsT
                i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
                lhsT_tiles = nl.ndarray(
                    (TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                    dtype=lhsT.dtype,
                    buffer=nl.sbuf,
                )
                for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
                    lhsT_tiles[bk_l, i_lhsT.p, i_lhsT.x] = nl.load(
                        lhsT[
                            (TILES_IN_BLOCK_K * k + bk_l) * TILE_K + i_lhsT.p,
                            BLOCK_M * m + i_lhsT.x,
                        ]
                    )

                # Do matmul with all tiles in the blocks
                i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M]
                i_rhs_mm = nl.mgrid[0:TILE_K, 0:TILE_N]
                i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_N]
                for bn in nl.affine_range(TILES_IN_BLOCK_N):
                    for bm in nl.affine_range(TILES_IN_BLOCK_M):
                        res_tile = nl.zeros(
                            (TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum
                        )

                        for bk in nl.affine_range(TILES_IN_BLOCK_K):
                            res_tile[...] += nisa.nc_matmul(
                                lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                                rhs_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x],
                            )

                        # Accumulate on corresponding SBUF tile
                        result_tiles[m, bm, bn, i_res_mm.p, i_res_mm.x] += res_tile[
                            i_res_mm.p, i_res_mm.x
                        ]

        # Copying the result from SBUF to HBM
        for m in nl.affine_range(NUM_BLOCK_M):
            for bm in nl.affine_range(TILES_IN_BLOCK_M):
                i_res = nl.mgrid[0:TILE_K, 0:TILE_N]
                i_res_packed = nl.mgrid[0:TILE_K, 0:BLOCK_N]
                result_packed = nl.ndarray(
                    (TILE_K, BLOCK_N), dtype=result_tiles.dtype, buffer=nl.sbuf
                )

                # coalesce result tiles for better DMA performance
                for bn in nl.affine_range(TILES_IN_BLOCK_N):
                    result_packed[i_res.p, bn * TILE_N + i_res.x] = nl.copy(
                        result_tiles[m, bm, bn, i_res.p, i_res.x]
                    )
                nl.store(
                    result[
                        (TILES_IN_BLOCK_M * m + bm) * TILE_K + i_res_packed.p,
                        BLOCK_N * n + i_res_packed.x,
                    ],
                    value=result_packed[i_res_packed.p, i_res_packed.x],
                )
    return result
