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
def allocated_fused_rms_norm_qkv(
    hidden, weights, multi_buffer, norm_dtype=nl.float32, eps=1e-6
):
    """
    Allocated kernel that computes RMSNorm(hidden) @ wQKV. This kernel is designed to only handle fp16/bf16 tensor types.
    Internally, normalizations are cast to fp32 to avoid NaN errors.

    Args:
        hidden (_type_): Input tensor of the attention block in BSH layout
        weights (_type_): Fused QKV linear weights, assumed to be eltwise-multiplied with RMS norm weight vector (gamma)
        out_tensor (_type_): Output tensor
        norm_dtype (_type_, optional): Data type for RMS norm, should be f32 to avoid NaN. Defaults to nl.float32.
        eps (_type_, optional): RMS norm epsilon term. Defaults to 1e-6.
    """
    # Hidden should be in BSH layout.
    batch, batchless_shape = hidden.shape[0], hidden.shape[1:]
    seqlen, dim = batchless_shape
    _dim, head_dim = weights.shape

    assert dim <= 8192 and dim & 128 == 0, "Unsupported hidden dimension"
    assert _dim == dim, "Reduction dimension must match"
    assert head_dim <= 512, "Head dimension must be 512 or less"

    out_tensor = nl.ndarray(
        (batch, seqlen, head_dim), dtype=hidden.dtype, buffer=nl.shared_hbm
    )

    pmax, fmax = nl.tile_size.pmax, nl.tile_size.psum_fmax  # 128, 512
    ix, iy = nl.mgrid[0:pmax, 0:dim]
    i_lhs = nl.mgrid[0:pmax, 0:pmax]
    i_rhs = nl.mgrid[0:pmax, 0:fmax]
    i_res = nl.mgrid[0:pmax, 0:fmax]
    M = math.ceil(dim / pmax)
    NUM_TRANSP_TILES = math.ceil(dim / fmax)
    NUM_TILES = math.ceil(seqlen / pmax)
    TILES_INT = math.ceil(NUM_TILES / 2)
    scale = 1 / dim

    iden_x, iden_y = nl.mgrid[0:pmax, 0:128]

    identity_a = nl.shared_constant(
        np.identity(n=128, dtype=np.int8), dtype=hidden.dtype
    )
    identity_tensor = nl.ndarray(
        (par_dim(pmax), 128),
        dtype=weights.dtype,
        buffer=ncc.sbuf.mod_alloc(base_addr=0),
    )
    identity_tensor[iden_x, iden_y] = nl.load(identity_a, dtype=weights.dtype)
    bias_placeholder = nl.ndarray(
        (par_dim(pmax), 1),
        dtype=np.float32,
        buffer=ncc.sbuf.mod_alloc(base_addr=128 * 2),
    )
    bias_placeholder[...] = 0

    for b in nl.affine_range(batch):
        weights_buffer = nl.ndarray(
            (M, par_dim(pmax), fmax),
            dtype=weights.dtype,
            buffer=ncc.sbuf.mod_alloc(
                base_addr=260 + (3 * dim + fmax) * 2 + (dim + 1) * 4,
                num_free_tiles=(M,),
            ),
        )
        # Preload the entire weights tensor. everything fits in SBUF for LLaMA 3.1 70B
        for m in nl.affine_range(M):
            weights_buffer[m, i_rhs.p, i_rhs.x] = nl.load(
                weights[m * pmax + i_rhs.p, i_rhs.x],
                mask=(m * pmax + i_rhs.p < dim) & (i_rhs.x < head_dim),
            )
        for i in nl.affine_range(TILES_INT):
            # Double buffer the input tensor
            in_bufs = nl.ndarray(
                (2, par_dim(pmax), dim),
                dtype=hidden.dtype,
                buffer=ncc.sbuf.mod_alloc(
                    base_addr=260, num_free_tiles=(multi_buffer,)
                ),
            )
            for i_interleave_grp in nl.affine_range(2):
                in_bufs[i_interleave_grp] = nl.load(
                    hidden[b, (2 * i + i_interleave_grp) * pmax + ix, iy],
                    mask=(2 * i + i_interleave_grp) * pmax + ix < seqlen,
                )
                act = nl.ndarray(
                    (par_dim(pmax), dim),
                    dtype=norm_dtype,
                    buffer=ncc.sbuf.mod_alloc(base_addr=260 + (2 * dim) * 2),
                )

                # Write the RMS and RMS Reciprocal tensors back out here, in-place
                square_sum = nl.ndarray(
                    (par_dim(pmax), 1),
                    dtype=norm_dtype,
                    buffer=ncc.sbuf.mod_alloc(
                        base_addr=260 + (2 * dim) * 2 + (dim) * 4
                    ),
                )

                # Write the output of RMS and RMS^T (in-place) out to here
                out_tile = nl.ndarray(
                    (par_dim(pmax), dim),
                    dtype=weights.dtype,
                    buffer=ncc.sbuf.mod_alloc(
                        base_addr=260 + (2 * dim) * 2 + (dim + 1) * 4
                    ),
                )

                # Store the final output tiles to here before sending back to DRAM
                output_sbuf = nl.ndarray(
                    (par_dim(pmax), fmax),
                    dtype=weights.dtype,
                    buffer=ncc.sbuf.mod_alloc(
                        base_addr=260 + (3 * dim) * 2 + (dim + 1) * 4
                    ),
                )

                act[...] = nisa.activation_reduce(
                    op=nl.square,
                    data=in_bufs[i_interleave_grp],
                    reduce_op=np.add,
                    reduce_res=square_sum[...],
                    bias=bias_placeholder[...],
                )
                square_sum[...] = nisa.tensor_scalar(
                    square_sum[...], np.multiply, scale, op1=np.add, operand1=eps
                )
                square_sum[...] = nisa.activation(
                    op=nl.rsqrt, data=square_sum[...], bias=bias_placeholder[...]
                )

                # all PE array ops must output to FP32 on trn1 but must match input dtype in trn2
                if nisa.get_nc_version() == nisa.nc_version.gen3:
                    transpose_res_psum = nl.ndarray(
                        (NUM_TRANSP_TILES, par_dim(pmax), 4 * pmax),
                        dtype=weights.dtype,
                        buffer=ncc.psum.mod_alloc(base_bank=0, num_bank_tiles=(1,)),
                    )  # FIXME: perf is better when all tiles are on bank 0?
                else:
                    transpose_res_psum = nl.ndarray(
                        (NUM_TRANSP_TILES, par_dim(pmax), 4 * pmax),
                        dtype=np.float32,
                        buffer=ncc.psum.mod_alloc(base_bank=0, num_bank_tiles=(1,)),
                    )  # FIXME: perf is better when all tiles are on bank 0?

                for m in nl.affine_range(NUM_TRANSP_TILES):
                    # Perform (hidden .* RMS Reciprocal)^T in tiles of fmax (512)
                    out_tile[i_rhs.p, m * fmax + i_rhs.x] = nl.multiply(
                        in_bufs[i_interleave_grp, i_rhs.p, m * fmax + i_rhs.x],
                        square_sum[...],
                        dtype=weights.dtype,
                    )
                    for j in nl.affine_range(4):
                        transpose_res_psum[m, i_lhs.p, j * pmax + i_lhs.x] = (
                            nisa.nc_matmul(
                                out_tile[i_lhs.p, (m * 4 + j) * pmax + i_lhs.x],
                                identity_tensor[...],
                                is_transpose=True,
                            )
                        )
                    out_tile[i_rhs.p, m * 4 * pmax + i_rhs.x] = nl.copy(
                        transpose_res_psum[m], dtype=hidden.dtype
                    )

                # perform (RMSNorm(hidden)^T)^T @ wQKV
                res_psum = nl.ndarray(
                    (1, par_dim(pmax), fmax),
                    dtype=nl.float32,
                    buffer=ncc.psum.mod_alloc(base_bank=7, num_bank_tiles=(1,)),
                )
                for m in nl.affine_range(M):
                    res_psum[0] += nisa.nc_matmul(
                        out_tile[i_lhs.p, m * pmax + i_lhs.x],
                        weights_buffer[m, i_rhs.p, i_rhs.x],
                    )

                output_sbuf[...] = nl.copy(res_psum[0], dtype=out_tensor.dtype)
                nl.store(
                    out_tensor[b, (2 * i + i_interleave_grp) * pmax + i_res.p, i_res.x],
                    value=output_sbuf,
                    mask=((2 * i + i_interleave_grp) * pmax + i_res.p < seqlen)
                    & (i_res.x < head_dim),
                )
    return out_tensor
