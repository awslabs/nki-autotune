# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

import logging
import math
from functools import partial
from typing import Optional
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.compiler as ncc
from neuronxcc.nki.language import par_dim
import numpy as np


# This is taken from the open source NKI samples repo
# https://github.com/aws-neuron/nki-samples/blob/main/src/tutorials/matrix_multiplication/matrix_multiplication_nki_kernels.py#L247
def nki_matmul_fully_optimized_(
    lhsT,
    rhs,
    result,
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


@nki.jit
def nki_rmsnorm_kernel(a_tensor, g_tensor):
    # Calculate out_tensor = a_tensor/RMS(a_tensor) * g_tensor
    # Where RMS(a_tensor) = sqrt((1/N) * sum(a_tensor * a_tensor))
    # and N = a_tensor.shape[1]
    # Reduction (mean) is performed in the free (2nd) dimension
    out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

    # Make sure shapes match
    assert (
        a_tensor.shape[1] == g_tensor.shape[0]
    ), f"a_tensor {a_tensor.shape} g_tensor {g_tensor.shape}"

    # Generate tensor indices to index input tensor
    ix = nl.arange(128)[:, None]
    iw = nl.arange(1)[:, None]
    iy = nl.arange(a_tensor.shape[1])[None, :]

    num_rows = a_tensor.shape[0]

    # Load RMSNorm weight once, reused by rows/tiles of a_tensor
    g_tile = nl.load(g_tensor.reshape((1, g_tensor.shape[0]))[iw, iy])

    # Process 128 rows at a time due to 128-partition tile size limitation
    # Since we're not reducing across the first dimension
    # Tiles can be processed independently
    for i in nl.affine_range(math.ceil(a_tensor.shape[0] / 128)):

        # Load input data from external memory to on-chip memory
        a_tile = nl.load(a_tensor[i * 128 + ix, iy], mask=(i * 128 + ix < num_rows))

        # Compute element-wise square of a_tensor
        in_square = nl.square(a_tile)

        # Calculate sum of squared elements, along last dimension
        square_sum = nl.sum(in_square, axis=[1])

        # Scale and get a reciprocal
        mean = square_sum / a_tensor.shape[1]

        # Take square root of mean and then reciprocal with
        # rsqrt API (one ISA instruction)
        rms_reciprocal = nl.rsqrt(mean)

        # Scale the input tensor
        out_tile = nl.multiply(a_tile, rms_reciprocal)

        # Broadcast weight along first axis to match tensor shape
        # num_rows_active = min(num_rows - i * 128, 128)
        g_bcast = g_tile.broadcast_to((128, g_tensor.shape[0]))

        # Multiply with the RMSNorm weight
        out_tile[...] = nl.multiply(out_tile, g_bcast, mask=(i * 128 + ix < num_rows))

        # store the addition results back to external memory (out_tensor)
        nl.store(
            out_tensor[i * 128 + ix, iy], value=out_tile, mask=(i * 128 + ix < num_rows)
        )

    return out_tensor
