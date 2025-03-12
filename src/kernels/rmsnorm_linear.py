"""
Copyright (c) 2024, Amazon.com. All Rights Reserved

kernels - Fused normalization with linear layers

"""

import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.compiler as ncc
import math
import numpy as np
from typing import Tuple
from neuronxcc import nki
from neuronxcc.nki.language import par_dim

from src.allocation.utils import update_base_addr
from src.kernels.utils import load_tensor


class RMSNormLinearCompatibility:
    """
    Inputs compatibility checks
    """

    def __init__(
        self,
        lhs_shape: Tuple,
        rhs_shape: Tuple,
        NUM_BLOCK_M: int,
        NUM_BLOCK_N: int,
        NUM_BLOCK_K: int,
        BUFFER_M: int,
        BUFFER_N: int,
        BUFFER_K: int,
    ) -> None:
        # Input sizes
        self.M, self.K = lhs_shape
        K_, self.N = rhs_shape

        # Single tile sizes
        self.TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
        self.TILE_N = nl.tile_size.gemm_moving_fmax  # 512
        self.TILE_K = nl.tile_size.pmax  # 128

        # Number of blocks
        self.NUM_BLOCK_M = NUM_BLOCK_M
        self.NUM_BLOCK_N = NUM_BLOCK_N
        self.NUM_BLOCK_K = NUM_BLOCK_K

        # Tiles in a block
        self.TILES_IN_BLOCK_M = self.M // self.NUM_BLOCK_M // self.TILE_M
        self.TILES_IN_BLOCK_N = self.N // self.NUM_BLOCK_N // self.TILE_N
        self.TILES_IN_BLOCK_K = self.K // self.NUM_BLOCK_K // self.TILE_K

        # Block sizes
        self.BLOCK_M = self.TILE_M * self.TILES_IN_BLOCK_M
        self.BLOCK_N = self.TILE_N * self.TILES_IN_BLOCK_N
        self.BLOCK_K = self.TILE_K * self.TILES_IN_BLOCK_K

        # Buffer degrees
        self.BUFFER_K = BUFFER_K
        self.BUFFER_M = BUFFER_M
        self.BUFFER_N = BUFFER_N

        self._check(K_)

    def _check(self, K_):
        pmax, fmax = nl.tile_size.pmax, nl.tile_size.psum_fmax  # 128, 512
        assert self.K <= 8192 and self.K & pmax == 0, "Unsupported hidden dimension"
        assert K_ == self.K, "Reduction dimension must match"
        assert self.N <= fmax, "Head dimension must be fmax (512) or less"

        assert self.K == K_, f"lhs and rhs contraction dimension mismatch, got {self.K} and {K_}"
        assert (
            self.NUM_BLOCK_K * self.TILES_IN_BLOCK_K * self.TILE_K == self.K
        ), f"NUM_BLOCK_K {self.NUM_BLOCK_K} * TILES_IN_BLOCK_K {self.TILES_IN_BLOCK_K} * TILE_K {self.TILE_K} != K {self.K}"
        assert (
            self.NUM_BLOCK_M * self.TILES_IN_BLOCK_M * self.TILE_M == self.M
        ), f"NUM_BLOCK_M {self.NUM_BLOCK_M} * TILES_IN_BLOCK_M {self.TILES_IN_BLOCK_M} * TILE_M {self.TILE_M} != M {self.M}"
        assert (
            self.NUM_BLOCK_N * self.TILES_IN_BLOCK_N * self.TILE_N == self.N
        ), f"NUM_BLOCK_N {self.NUM_BLOCK_N} * TILES_IN_BLOCK_N {self.TILES_IN_BLOCK_N} * TILE_N {self.TILE_N} != N {self.N}"

        assert (
            self.BUFFER_M <= self.NUM_BLOCK_M
        ), f"M buffer degree {self.BUFFER_M} cannot be larger than number of blocks {self.NUM_BLOCK_M}"
        assert (
            self.BUFFER_N <= self.NUM_BLOCK_N
        ), f"N buffer degree {self.BUFFER_N} cannot be larger than number of blocks {self.NUM_BLOCK_N}"
        assert (
            self.BUFFER_K <= self.NUM_BLOCK_K
        ), f"K buffer degree {self.BUFFER_K} cannot be larger than number of blocks {self.NUM_BLOCK_K}"


def compatibility_checks(hidden_shape, weights_shape):
    batch, seqlen, dim = hidden_shape
    _dim, head_dim = weights_shape
    pmax, fmax = nl.tile_size.pmax, nl.tile_size.psum_fmax  # 128, 512
    assert dim <= 8192 and dim & pmax == 0, "Unsupported hidden dimension"
    assert _dim == dim, "Reduction dimension must match"
    assert head_dim <= fmax, "Head dimension must be fmax (512) or less"


@nki.compiler.skip_middle_end_transformations
@nki.jit
def allocated_fused_rms_norm_qkv(hidden, weights, hidden_buffer_degree, eps):
    """
    Allocated kernel that computes RMSNorm(hidden) @ wQKV. This kernel is designed to only handle fp16/bf16 tensor types.
    Internally, normalizations are cast to fp32 to avoid NaN errors.

    Args:
        hidden (_type_): Input tensor of the attention block in BSH layout
        weights (_type_): Fused QKV linear weights, assumed to be eltwise-multiplied with RMS norm weight vector (gamma)
        eps (_type_, optional): RMS norm epsilon term. Defaults to 1e-6.
    """
    compatibility_checks(hidden.shape, weights.shape)
    # Hidden should be in BSH layout.
    batch, batchless_shape = hidden.shape[0], hidden.shape[1:]
    seqlen, dim = batchless_shape
    _dim, head_dim = weights.shape

    norm_dtype = nl.float32

    out_tensor = nl.ndarray((batch, seqlen, head_dim), dtype=hidden.dtype, buffer=nl.shared_hbm)

    pmax, fmax = nl.tile_size.pmax, nl.tile_size.psum_fmax  # 128, 512
    ix, iy = nl.mgrid[0:pmax, 0:dim]
    i_lhs = nl.mgrid[0:pmax, 0:pmax]
    i_rhs = nl.mgrid[0:pmax, 0:fmax]
    i_res = nl.mgrid[0:pmax, 0:fmax]
    M = math.ceil(dim / pmax)
    NUM_TRANSP_TILES = math.ceil(dim / fmax)
    NUM_TILES = math.ceil(seqlen / pmax)
    TILES_INT = math.ceil(NUM_TILES / hidden_buffer_degree)
    scale = 1 / dim
    sbuf_base_addr = 0

    iden_x, iden_y = nl.mgrid[0:pmax, 0:128]

    identity_a = nl.shared_constant(np.identity(n=128, dtype=np.int8), dtype=hidden.dtype)
    identity_tensor = nl.ndarray(
        (par_dim(pmax), 128), dtype=weights.dtype, buffer=ncc.sbuf.mod_alloc(base_addr=sbuf_base_addr)
    )
    sbuf_base_addr = update_base_addr(sbuf_base_addr, identity_tensor, True)
    identity_tensor[iden_x, iden_y] = nl.load(identity_a, dtype=weights.dtype)
    bias_placeholder = nl.ndarray(
        (par_dim(pmax), 1), dtype=np.float32, buffer=ncc.sbuf.mod_alloc(base_addr=sbuf_base_addr)
    )
    sbuf_base_addr = update_base_addr(sbuf_base_addr, bias_placeholder, True)
    bias_placeholder[...] = 0

    for b in nl.affine_range(batch):
        weights_buffer = nl.ndarray(
            (M, par_dim(pmax), fmax),
            dtype=weights.dtype,
            buffer=ncc.sbuf.mod_alloc(base_addr=sbuf_base_addr, num_free_tiles=(M,)),
        )
        sbuf_base_addr = update_base_addr(sbuf_base_addr, weights_buffer, True)
        # Preload the entire weights tensor. everything fits in SBUF for LLaMA 3.1 70B
        for m in nl.affine_range(M):
            weights_buffer[m, i_rhs.p, i_rhs.x] = nl.load(
                weights[m * pmax + i_rhs.p, i_rhs.x], mask=(m * pmax + i_rhs.p < dim) & (i_rhs.x < head_dim)
            )
        for i in nl.affine_range(TILES_INT):
            # Double buffer the input tensor
            in_bufs = nl.ndarray(
                (hidden_buffer_degree, par_dim(pmax), dim),
                dtype=hidden.dtype,
                buffer=ncc.sbuf.mod_alloc(base_addr=sbuf_base_addr, num_free_tiles=(hidden_buffer_degree,)),
            )
            sbuf_base_addr = update_base_addr(sbuf_base_addr, in_bufs, True)
            for i_interleave_grp in nl.affine_range(hidden_buffer_degree):
                in_bufs[i_interleave_grp] = nl.load(
                    hidden[b, (hidden_buffer_degree * i + i_interleave_grp) * pmax + ix, iy],
                    mask=(hidden_buffer_degree * i + i_interleave_grp) * pmax + ix < seqlen,
                )
                act = nl.ndarray(
                    (par_dim(pmax), dim), dtype=norm_dtype, buffer=ncc.sbuf.mod_alloc(base_addr=sbuf_base_addr)
                )
                sbuf_base_addr = update_base_addr(sbuf_base_addr, act, True)

                # Write the RMS and RMS Reciprocal tensors back out here, in-place
                square_sum = nl.ndarray(
                    (par_dim(pmax), 1), dtype=norm_dtype, buffer=ncc.sbuf.mod_alloc(base_addr=sbuf_base_addr)
                )
                sbuf_base_addr = update_base_addr(sbuf_base_addr, square_sum, True)

                # Write the output of RMS and RMS^T (in-place) out to here
                out_tile = nl.ndarray(
                    (par_dim(pmax), dim), dtype=weights.dtype, buffer=ncc.sbuf.mod_alloc(base_addr=sbuf_base_addr)
                )
                sbuf_base_addr = update_base_addr(sbuf_base_addr, out_tile, True)

                # Store the final output tiles to here before sending back to DRAM
                output_sbuf = nl.ndarray(
                    (par_dim(pmax), fmax), dtype=weights.dtype, buffer=ncc.sbuf.mod_alloc(base_addr=sbuf_base_addr)
                )
                sbuf_base_addr = update_base_addr(sbuf_base_addr, output_sbuf, True)

                act[...] = nisa.activation_reduce(
                    op=nl.square,
                    data=in_bufs[i_interleave_grp],
                    reduce_op=np.add,
                    reduce_res=square_sum[...],
                    bias=bias_placeholder[...],
                )
                square_sum[...] = nisa.tensor_scalar(square_sum[...], np.multiply, scale, op1=np.add, operand1=eps)
                square_sum[...] = nisa.activation(op=nl.rsqrt, data=square_sum[...], bias=bias_placeholder[...])

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
                        in_bufs[i_interleave_grp, i_rhs.p, m * fmax + i_rhs.x], square_sum[...], dtype=weights.dtype
                    )
                    TILES_IN_BLOCK = fmax // pmax
                    for j in nl.affine_range(TILES_IN_BLOCK):
                        transpose_res_psum[m, i_lhs.p, j * pmax + i_lhs.x] = nisa.nc_matmul(
                            out_tile[i_lhs.p, (m * TILES_IN_BLOCK + j) * pmax + i_lhs.x],
                            identity_tensor[...],
                            is_transpose=True,
                        )
                    out_tile[i_rhs.p, m * TILES_IN_BLOCK * pmax + i_rhs.x] = nl.copy(
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
                        out_tile[i_lhs.p, m * pmax + i_lhs.x], weights_buffer[m, i_rhs.p, i_rhs.x]
                    )

                output_sbuf[...] = nl.copy(res_psum[0], dtype=out_tensor.dtype)
                nl.store(
                    out_tensor[b, (hidden_buffer_degree * i + i_interleave_grp) * pmax + i_res.p, i_res.x],
                    value=output_sbuf,
                    mask=((hidden_buffer_degree * i + i_interleave_grp) * pmax + i_res.p < seqlen)
                    & (i_res.x < head_dim),
                )
                sbuf_base_addr = update_base_addr(sbuf_base_addr, output_sbuf, False)
                sbuf_base_addr = update_base_addr(sbuf_base_addr, out_tile, False)
                sbuf_base_addr = update_base_addr(sbuf_base_addr, square_sum, False)
                sbuf_base_addr = update_base_addr(sbuf_base_addr, act, False)
            sbuf_base_addr = update_base_addr(sbuf_base_addr, in_bufs, False)
        sbuf_base_addr = update_base_addr(sbuf_base_addr, weights_buffer, False)
    sbuf_base_addr = update_base_addr(sbuf_base_addr, bias_placeholder, False)
    sbuf_base_addr = update_base_addr(sbuf_base_addr, identity_tensor, False)
    return out_tensor


@nki.compiler.enable_stack_allocator()
@nki.compiler.skip_middle_end_transformations
@nki.jit
def stack_allocated_fused_rms_norm_qkv(hidden, weights, norm_dtype=nl.float32, eps=1e-6):
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
    compatibility_checks(hidden.shape, weights.shape)
    # Hidden should be in BSH layout.
    batch, seqlen, dim = hidden.shape
    _dim, head_dim = weights.shape

    out_tensor = nl.ndarray((batch, seqlen, head_dim), dtype=hidden.dtype, buffer=nl.shared_hbm)

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

    for b in nl.affine_range(batch):
        weights_buffer = nl.ndarray((M, par_dim(pmax), fmax), dtype=weights.dtype, buffer=nl.sbuf)
        # Preload the entire weights tensor. everything fits in SBUF for LLaMA 3.1 70B
        for m in nl.affine_range(M):
            weights_buffer[m, i_rhs.p, i_rhs.x] = nl.load(
                weights[m * pmax + i_rhs.p, i_rhs.x], mask=(m * pmax + i_rhs.p < dim) & (i_rhs.x < head_dim)
            )
        for i in nl.affine_range(TILES_INT):
            # Double buffer the input tensor
            in_bufs = nl.ndarray((2, par_dim(pmax), dim), dtype=hidden.dtype, buffer=nl.sbuf)
            for i_interleave_grp in nl.affine_range(2):
                in_bufs[i_interleave_grp] = nl.load(
                    hidden[b, (2 * i + i_interleave_grp) * pmax + ix, iy],
                    mask=(2 * i + i_interleave_grp) * pmax + ix < seqlen,
                )
                act = nl.ndarray((par_dim(pmax), dim), dtype=norm_dtype, buffer=nl.sbuf)

                # Write the RMS and RMS Reciprocal tensors back out here, in-place
                square_sum = nl.ndarray((par_dim(pmax), 1), dtype=norm_dtype, buffer=nl.sbuf)

                # Write the output of RMS and RMS^T (in-place) out to here
                out_tile = nl.ndarray((par_dim(pmax), dim), dtype=weights.dtype, buffer=nl.sbuf)

                # Store the final output tiles to here before sending back to DRAM
                output_sbuf = nl.ndarray((par_dim(pmax), fmax), dtype=weights.dtype, buffer=nl.sbuf)
                # Allocate the psum early, such that it will not share address with
                # transpose_res_psum, this avoid anti-dependencies on instuctions trying
                # to read two of them
                res_psum = nl.ndarray((par_dim(pmax), fmax), dtype=nl.float32, buffer=nl.psum)

                act[...] = nisa.activation_reduce(
                    op=nl.square, data=in_bufs[i_interleave_grp], reduce_op=np.add, reduce_res=square_sum[...]
                )
                square_sum[...] = nisa.tensor_scalar(square_sum[...], np.multiply, scale, op1=np.add, operand1=eps)
                square_sum[...] = nisa.activation(op=nl.rsqrt, data=square_sum[...])

                # perform (RMSNorm(hidden)^T)^T @ wQKV

                for m in nl.affine_range(NUM_TRANSP_TILES):
                    # all PE array ops must output to FP32 on trn1 but must match input dtype in trn2
                    if nisa.get_nc_version() == nisa.nc_version.gen3:
                        transpose_res_psum_dtype = weights.dtype
                    else:
                        transpose_res_psum_dtype = np.float32

                    transpose_res_psum = nl.ndarray(
                        (par_dim(pmax), 4 * pmax), dtype=transpose_res_psum_dtype, buffer=nl.psum
                    )

                    # Perform (hidden .* RMS Reciprocal)^T in tiles of fmax (512)
                    out_tile[i_rhs.p, m * fmax + i_rhs.x] = nl.multiply(
                        in_bufs[i_interleave_grp, i_rhs.p, m * fmax + i_rhs.x], square_sum[...], dtype=weights.dtype
                    )
                    for j in nl.affine_range(4):
                        transpose_res_psum[i_lhs.p, j * pmax + i_lhs.x] = nisa.nc_transpose(
                            out_tile[i_lhs.p, (m * 4 + j) * pmax + i_lhs.x]
                        )
                    out_tile[i_rhs.p, m * 4 * pmax + i_rhs.x] = nl.copy(transpose_res_psum, dtype=hidden.dtype)

                for m in nl.affine_range(M):
                    res_psum += nisa.nc_matmul(
                        out_tile[i_lhs.p, m * pmax + i_lhs.x], weights_buffer[m, i_rhs.p, i_rhs.x]
                    )

                output_sbuf[...] = nl.copy(res_psum, dtype=out_tensor.dtype)
                nl.store(
                    out_tensor[b, (2 * i + i_interleave_grp) * pmax + i_res.p, i_res.x],
                    value=output_sbuf,
                    mask=((2 * i + i_interleave_grp) * pmax + i_res.p < seqlen) & (i_res.x < head_dim),
                )
    return out_tensor


@nki.compiler.enable_stack_allocator()
@nki.compiler.skip_middle_end_transformations
@nki.jit
def optimized_fused_rms_norm_qkv(
    hidden,
    weights,
    NUM_BLOCK_M: int,
    NUM_BLOCK_N: int,
    NUM_BLOCK_K: int,
    BUFFER_M: int,
    BUFFER_N: int,
    BUFFER_K: int,
    norm_dtype=nl.float32,
    eps=1e-6,
):
    """
    Optimized kernel that computes RMSNorm(hidden) @ wQKV. This kernel is designed to only handle fp16/bf16 tensor types.
    Internally, normalizations are cast to fp32 to avoid NaN errors.

    Args:
        hidden (_type_): Input tensor of the attention block in BSH layout
        weights (_type_): Fused QKV linear weights, assumed to be eltwise-multiplied with RMS norm weight vector (gamma)
        out_tensor (_type_): Output tensor
        norm_dtype (_type_, optional): Data type for RMS norm, should be f32 to avoid NaN. Defaults to nl.float32.
        eps (_type_, optional): RMS norm epsilon term. Defaults to 1e-6.
    """
    checker = RMSNormLinearCompatibility(
        hidden.shape[1:], weights.shape, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K
    )
    batch, seqlen, dim = hidden.shape
    _, head_dim = weights.shape

    pmax, fmax = nl.tile_size.pmax, nl.tile_size.psum_fmax  # 128, 512
    ix, iy = nl.mgrid[0:pmax, 0:dim]
    i_lhs = nl.mgrid[0:pmax, 0:pmax]
    i_rhs = nl.mgrid[0:pmax, 0:fmax]
    i_res = nl.mgrid[0:pmax, 0:fmax]
    NUM_SEQ_TILES = math.ceil(seqlen / pmax)
    NUM_DIM_TILES = math.ceil(dim / pmax)

    out_tensor = nl.ndarray((batch, seqlen, head_dim), dtype=hidden.dtype, buffer=nl.shared_hbm)

    """
    Preload the entire weights tensor.
    Everything fits in SBUF for LLaMA 3.1 70B
    (dim, fmax) covers the entire weights tensor as head_dim<=fmax
    """
    weights_buffer = load_tensor(weights, par_ofs=0, free_ofs=0, load_shape=(NUM_DIM_TILES, pmax, fmax))

    for batch_id in nl.affine_range(batch):
        for seq_tile_id in nl.affine_range(NUM_SEQ_TILES, multi_buffer=2):
            seq_offset = seq_tile_id * pmax
            in_bufs = nl.ndarray((par_dim(pmax), dim), dtype=hidden.dtype, buffer=nl.sbuf)
            in_bufs = nl.load(hidden[batch_id, seq_offset + ix, iy], mask=seq_offset + ix < seqlen)

            rmsnormT_tile = rmsnorm_tile(in_bufs=in_bufs, eps=eps, norm_dtype=norm_dtype, output_dtype=weights.dtype)

            """
            Perform (RMSNorm(hidden)^T)^T @ wQKV
            """
            res_psum = nl.ndarray((par_dim(pmax), fmax), dtype=nl.float32, buffer=nl.psum)
            for dim_tile_id in nl.affine_range(NUM_DIM_TILES):
                res_psum += nisa.nc_matmul(
                    rmsnormT_tile[i_lhs.p, dim_tile_id * pmax + i_lhs.x], weights_buffer[dim_tile_id, i_rhs.p, i_rhs.x]
                )

            """
            Store the final output tiles to output_sbuf before sending back to DRAM
            """
            output_sbuf = nl.ndarray((par_dim(pmax), fmax), dtype=weights.dtype, buffer=nl.sbuf)
            output_sbuf[...] = nl.copy(res_psum, dtype=out_tensor.dtype)
            nl.store(
                out_tensor[batch_id, seq_offset + i_res.p, i_res.x],
                value=output_sbuf,
                mask=(seq_offset + i_res.p < seqlen) & (i_res.x < head_dim),
            )
    return out_tensor


def rmsnorm_tile(in_bufs, eps, norm_dtype, output_dtype):
    """
    Compute the RMSNorm(hidden)^T tile for the in_bufs
    Args:
        in_bufs: 2D input tensor tile
        eps: RMS norm epsilon term
        norm_dtype: Data type for RMS norm, should be f32 to avoid NaN
        output_dtype: Data type for output tensor
    """
    pmax, fmax = nl.tile_size.pmax, nl.tile_size.psum_fmax  # 128, 512
    _, dim = in_bufs.shape
    i_lhs = nl.mgrid[0:pmax, 0:pmax]
    i_rhs = nl.mgrid[0:pmax, 0:fmax]
    """
    all PE array ops must output to FP32 on trn1 but must match input dtype in trn2
    """
    if nisa.get_nc_version() == nisa.nc_version.gen3:
        transpose_res_psum_dtype = output_dtype
    else:
        transpose_res_psum_dtype = np.float32
    """
    Allocate the psum early, such that it will not share address with transpose_res_psum.
    This avoid anti-dependencies on instuctions trying to read two of them.
    """
    square_sum = nl.ndarray((par_dim(pmax), 1), dtype=norm_dtype, buffer=nl.sbuf)

    """
    Write the RMS and RMS Reciprocal tensors back to square_sum, in-place
    """
    nisa.activation_reduce(op=nl.square, data=in_bufs, reduce_op=np.add, reduce_res=square_sum[...])
    scale = 1 / dim
    square_sum[...] = nisa.tensor_scalar(square_sum[...], np.multiply, scale, op1=np.add, operand1=eps)
    square_sum[...] = nisa.activation(op=nl.rsqrt, data=square_sum[...])

    """
    Write the output of RMS and RMS^T (in-place) to rmsnorm_t_tile
    Perform (hidden .* RMS Reciprocal)^T in tiles of fmax (512)
    """
    NUM_TRANSP_TILES = math.ceil(dim / fmax)
    rmsnorm_t_tile = nl.ndarray((par_dim(pmax), dim), dtype=output_dtype, buffer=nl.sbuf)
    for transpose_tile_id in nl.affine_range(NUM_TRANSP_TILES):
        dim_offset = transpose_tile_id * fmax
        rmsnorm_t_tile[i_rhs.p, dim_offset + i_rhs.x] = nl.multiply(
            in_bufs[i_rhs.p, dim_offset + i_rhs.x], square_sum[...], dtype=output_dtype
        )
        """
        nisa.nc_transpose only handles tiles of (pmax, pmax)
        In order to transpose rmsnorm_t_tile (pmax, fmax), need to parallel by fmax//pmax
        """
        transpose_res_psum = nl.ndarray((par_dim(pmax), fmax), dtype=transpose_res_psum_dtype, buffer=nl.psum)
        for j in nl.affine_range(fmax // pmax):
            transpose_res_psum[i_lhs.p, j * pmax + i_lhs.x] = nisa.nc_transpose(
                rmsnorm_t_tile[i_lhs.p, dim_offset + j * pmax + i_lhs.x]
            )
        rmsnorm_t_tile[i_rhs.p, dim_offset + i_rhs.x] = nl.copy(transpose_res_psum, dtype=in_bufs.dtype)
    return rmsnorm_t_tile
