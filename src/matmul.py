# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

import logging, math
import neuronxcc.nki as nki
import neuronxcc.nki.compiler as ncc
from neuronxcc.nki.language import par_dim
import numpy as np


def common_head(lhsT, rhs, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N):
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert (
        K == K_
    ), f"lhsT and rhs contraction dimension mismatch, got {lhsT.shape} and {rhs.shape}"

    # Tile sizes
    TILE_K = nl.tile_size.pmax  # 128
    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    # Block sizes
    BLOCK_K = TILE_K * TILES_IN_BLOCK_K
    BLOCK_M = TILE_M * TILES_IN_BLOCK_M
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N

    # Number of blocks
    NUM_BLOCK_K = K // BLOCK_K
    NUM_BLOCK_M = M // BLOCK_M
    NUM_BLOCK_N = N // BLOCK_N

    assert NUM_BLOCK_K * TILES_IN_BLOCK_K * TILE_K == K
    assert NUM_BLOCK_M * TILES_IN_BLOCK_M * TILE_M == M
    assert NUM_BLOCK_N * TILES_IN_BLOCK_N * TILE_N == N

    return (
        M,
        N,
        K,
        TILE_M,
        TILE_N,
        TILE_K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        NUM_BLOCK_M,
        NUM_BLOCK_N,
        NUM_BLOCK_K,
    )


@nki.jit
def matmul_NMK(lhsT, rhs, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N):
    """
    Optimized matrix multiplication kernel

     Args:

        lhsT: an input tensor of shape [K, M], where K is a multiple of 1024
        and M is a multiple of 512.  It is the left-hand-side argument of the
        matrix multiplication, delivered transposed for optimal performance.

        rhs: an input tensor of shape [K, N],  where K is a multiple of 1024
          and N is a multiple of 2048.  It is the right-hand-side argument of
          the matrix multiplication.

        result: the resulting output tensor of shape [M, N]

    """
    (
        M,
        N,
        K,
        TILE_M,
        TILE_N,
        TILE_K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        NUM_BLOCK_M,
        NUM_BLOCK_N,
        NUM_BLOCK_K,
    ) = common_head(lhsT, rhs, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N)

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    for block_id_N in nl.affine_range(NUM_BLOCK_N):
        for block_id_M in nl.affine_range(NUM_BLOCK_M):
            # BLOCK_M * BLOCK_N result block
            result_block = nl.zeros(
                (TILES_IN_BLOCK_M, nl.par_dim(TILE_M), BLOCK_N),
                dtype=result.dtype,
                buffer=nl.sbuf,
            )

            for block_id_K in nl.affine_range(NUM_BLOCK_K):
                # BLOCK_K * BLOCK_M lhsT block
                lhsT_block = nl.ndarray(
                    (TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                    dtype=lhsT.dtype,
                    buffer=nl.sbuf,
                )
                # BLOCK_K * BLOCK_N rhs block
                rhs_block = nl.ndarray(
                    (TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                    dtype=rhs.dtype,
                    buffer=nl.sbuf,
                )

                # Load the lhsT and the rhs block by tiles
                i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
                i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
                for tile_id_K in nl.affine_range(TILES_IN_BLOCK_K):
                    k_start = block_id_K * BLOCK_K + tile_id_K * TILE_K
                    m_start = block_id_M * BLOCK_M
                    n_start = block_id_N * BLOCK_N

                    lhsT_block[tile_id_K] = nl.load(
                        lhsT[k_start + i_lhsT.p, m_start + i_lhsT.x]
                    )
                    rhs_block[tile_id_K] = nl.load(
                        rhs[k_start + i_rhs.p, n_start + i_rhs.x]
                    )

                for tile_id_M in nl.affine_range(TILES_IN_BLOCK_M):
                    for tile_id_N in nl.affine_range(TILES_IN_BLOCK_N):
                        # Keep the tile of Z stationary in the PSUM buffer to minimize the
                        # number of calls to nl.loop_reduce
                        res_tile = nl.zeros(
                            (TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum
                        )

                        m_start = tile_id_M * TILE_M
                        m_end = m_start + TILE_M

                        n_start = tile_id_N * TILE_N
                        n_end = n_start + TILE_N

                        # Accumulate the K tiles within the K block
                        for tile_id_K in nl.affine_range(TILES_IN_BLOCK_K):
                            res_tile += nisa.nc_matmul(
                                lhsT_block[tile_id_K, :, m_start:m_end],
                                rhs_block[tile_id_K, :, n_start:n_end],
                            )

                        # Accumulate the K blocks
                        result_block[tile_id_M, :, n_start:n_end] = nl.loop_reduce(
                            res_tile,
                            op=np.add,
                            loop_indices=[block_id_K],
                            dtype=result.dtype,
                        )

            for tile_id_M in nl.affine_range(TILES_IN_BLOCK_M):
                m_start = block_id_M * BLOCK_M + tile_id_M * TILE_M
                m_end = m_start + TILE_M

                n_start = block_id_N * BLOCK_N
                n_end = n_start + BLOCK_N

                # We coalesce memory accesses by storing BLOCK_N
                # values of Z at a time. We cannot coalesce across M because M gets
                # split across the partition dimension
                nl.store(
                    result[m_start:m_end, n_start:n_end], value=result_block[tile_id_M]
                )
    return result


# This is taken from the open source NKI samples repo
# https://github.com/aws-neuron/nki-samples/blob/main/src/tutorials/matrix_multiplication/matrix_multiplication_nki_kernels.py#L247
@nki.jit
def matmul_NKM(lhsT, rhs, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N):
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

    (
        M,
        N,
        K,
        TILE_M,
        TILE_N,
        TILE_K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        NUM_BLOCK_M,
        NUM_BLOCK_N,
        NUM_BLOCK_K,
    ) = common_head(lhsT, rhs, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N)

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

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
