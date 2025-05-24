"""
Copyright (c) 2024, Amazon.com. All Rights Reserved

kernels - Fused normalization with linear layers

"""

import math

import neuronxcc.nki.compiler as ncc
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
from neuronxcc import nki
from neuronxcc.nki.language import par_dim
from neuronxcc.nki.typing import tensor

from autotune.allocation.utils import update_base_addr
from autotune.core.dma import load_tensor_block, save_result_block, save_result_dma
from autotune.core.layout import transpose_tile, transpose_tiles_in_block
from autotune.core.utils import GEMMCompatibility


@nki.compiler.skip_middle_end_transformations
@nki.jit
def allocated_fused_rms_norm_qkv(hidden, weights, hidden_buffer_degree: int, eps: float):
    """
    Allocated kernel that computes RMSNorm(hidden) @ wQKV. This kernel is designed to only handle fp16/bf16 tensor types.
    Internally, normalizations are cast to fp32 to avoid NaN errors.

    Args:
        hidden (_type_): Input tensor of the attention block in BSH layout
        weights (_type_): Fused QKV linear weights, assumed to be eltwise-multiplied with RMS norm weight vector (gamma)
        eps (_type_, optional): RMS norm epsilon term. Defaults to 1e-6.
    """
    # Hidden should be in BSH layout.
    batch, batchless_shape = hidden.shape[0], hidden.shape[1:]
    seqlen, dim = batchless_shape
    _dim, head_dim = weights.shape
    pmax, fmax = nl.tile_size.pmax, nl.tile_size.psum_fmax  # 128, 512
    assert head_dim <= fmax, f"Head dimension must be fmax (512) or less. Received {head_dim}."
    assert dim <= 8192 and dim & pmax == 0, f"Unsupported hidden dimension {dim}."

    norm_dtype = nl.float32

    out_tensor = nl.ndarray((batch, seqlen, head_dim), dtype=hidden.dtype, buffer=nl.shared_hbm)

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
    # Hidden should be in BSH layout.
    batch, seqlen, dim = hidden.shape
    _dim, head_dim = weights.shape
    pmax, fmax = nl.tile_size.pmax, nl.tile_size.psum_fmax  # 128, 512
    assert head_dim <= fmax, f"Head dimension must be fmax (512) or less. Received {head_dim}."
    assert dim <= 8192 and dim & pmax == 0, f"Unsupported hidden dimension {dim}."

    out_tensor = nl.ndarray((batch, seqlen, head_dim), dtype=hidden.dtype, buffer=nl.shared_hbm)

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


@nki.jit
def blocked_fused_rms_norm_linear(lhs, rhs, NUM_BLOCK_M: int, NUM_BLOCK_N: int, norm_dtype=nl.float32, eps=1e-6):
    """
    Optimized kernel that computes RMSNorm(hidden) @ wQKV. This kernel is designed to only handle fp16/bf16 tensor types.
    Internally, normalizations are cast to fp32 to avoid NaN errors.

    Args:
        lhs (_type_): Input tensor in (batch, M, K) layout
        rhs (_type_): Input tensor in (K, N) layout
        result (_type_): Output tensor
        norm_dtype (_type_, optional): Data type for RMS norm, should be f32 to avoid NaN. Defaults to nl.float32.
        eps (_type_, optional): RMS norm epsilon term. Defaults to 1e-6.
    """
    assert len(lhs.shape) == 3, f"Expecting (batch, M, K) in LHS. Received {lhs.shape}."
    mm = GEMMCompatibility(transposed_lhs=False)
    mm((lhs, rhs), {"NUM_BLOCK_M": NUM_BLOCK_M, "NUM_BLOCK_N": NUM_BLOCK_N})
    batch_size = lhs.shape[0]
    result = nl.ndarray((batch_size, mm.M, mm.N), dtype=lhs.dtype, buffer=nl.shared_hbm)
    for batch_id in nl.affine_range(batch_size):
        for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
            lhs_block = load_tensor_block(
                input_tensor=lhs[batch_id],
                ofs=(block_id_M * mm.BLOCK_M, 0),
                load_shape=(mm.TILE_M, mm.TILES_IN_BLOCK_M, 1, mm.K),
            )
            compute_RMSNormT(lhs_block, mm, eps, norm_dtype)
            for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
                result_block = nl.zeros(
                    (nl.par_dim(mm.TILE_M), mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                    dtype=lhs.dtype,
                    buffer=nl.sbuf,
                )
                rhs_block = load_tensor_block(
                    input_tensor=rhs,
                    ofs=(0, block_id_N * mm.BLOCK_N),
                    load_shape=(mm.TILE_K, mm.TILES_IN_K, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                )
                # matmul_blocks_tile_transposed_lhs(lhs_block, rhs_block, result_block)
                save_result_block(
                    result[batch_id], result_block, m_ofs=block_id_M * mm.BLOCK_M, n_ofs=block_id_N * mm.BLOCK_N
                )
    return result


def compute_RMSNormT(in_block, mm: GEMMCompatibility, eps, norm_dtype):
    """
    Compute the RMSNormT(hidden) block for the in_block
    Args:
        in_block: 3D input tensor tile (TILES_IN_BLOCK_M, TILE_M, K)
        eps: RMS norm epsilon term
        norm_dtype: Data type for RMS norm, should be f32 to avoid NaN
        output_dtype: Data type for output tensor
    """
    TILE_M, TILES_IN_BLOCK_M, _, K = in_block.shape
    assert in_block.shape == (mm.TILE_M, mm.TILES_IN_BLOCK_M, 1, mm.K)
    scale = 1 / K
    i_lhs = nl.mgrid[0:TILE_M, 0:K]
    for tile_id_M in nl.affine_range(TILES_IN_BLOCK_M):
        square_sum = nl.ndarray((par_dim(TILE_M), 1), dtype=norm_dtype, buffer=nl.sbuf)

        """
        Write the RMS and RMS Reciprocal tensors back to square_sum, in-place
        """
        nisa.activation_reduce(
            op=nl.square, data=in_block[i_lhs.p, tile_id_M, 0, i_lhs.x], reduce_op=np.add, reduce_res=square_sum[...]
        )
        square_sum[...] = nisa.tensor_scalar(square_sum[...], np.multiply, scale, op1=np.add, operand1=eps)
        square_sum[...] = nisa.activation(op=nl.rsqrt, data=square_sum[...])

        """
        Write the output of RMS (in-place)
        Perform (hidden .* RMS Reciprocal) in tiles of fmax (512)
        """
        rmsnorm_block = nl.ndarray((par_dim(TILE_M), K), dtype=in_block.dtype, buffer=nl.sbuf)
        rmsnorm_block[...] = nl.multiply(
            in_block[i_lhs.p, tile_id_M, 0, i_lhs.x], square_sum[...], dtype=in_block.dtype
        )
        transpose_tile(rmsnorm_block)
        in_block[i_lhs.p, tile_id_M, 0, i_lhs.x] = nl.copy(rmsnorm_block, dtype=in_block.dtype)


@nki.jit()
def online_rmsnorm_linear_MKN(
    lhs: tensor,
    rhs: tensor,
    NUM_BLOCK_M: int,
    NUM_BLOCK_N: int,
    NUM_BLOCK_K: int,
    norm_dtype=nl.float32,
    eps: float = 1e-6,
):
    assert len(lhs.shape) == 3, f"Expecting (batch, M, K) in LHS. Received {lhs.shape}."
    mm = GEMMCompatibility(transposed_lhs=False)
    mm((lhs, rhs), {"NUM_BLOCK_M": NUM_BLOCK_M, "NUM_BLOCK_N": NUM_BLOCK_N, "NUM_BLOCK_K": NUM_BLOCK_K})
    batch_size = lhs.shape[0]
    result = nl.ndarray((batch_size, mm.M, mm.N), dtype=lhs.dtype, buffer=nl.shared_hbm)
    for batch_id in nl.affine_range(batch_size):
        for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
            result_blocks = nl.zeros(
                (nl.par_dim(mm.TILE_M), mm.NUM_BLOCK_N, mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                dtype=result.dtype,
                buffer=nl.sbuf,
            )
            square_sums = nl.zeros(
                (nl.par_dim(mm.TILE_M), mm.TILES_IN_BLOCK_M, 1), dtype=norm_dtype, buffer=nl.sbuf
            )  # m_i
            rms_factors = nl.zeros(square_sums.shape, dtype=square_sums.dtype, buffer=nl.sbuf)
            prev_rms_factors = nl.zeros(square_sums.shape, dtype=square_sums.dtype, buffer=nl.sbuf)
            for block_id_K in nl.sequential_range(mm.NUM_BLOCK_K):
                # FIXME: need to develop a replaceable tensor load/access function
                lhs_block = load_tensor_block(
                    input_tensor=lhs[batch_id],
                    ofs=(block_id_M * mm.BLOCK_M, block_id_K * mm.BLOCK_K),
                    load_shape=(mm.TILE_M, mm.TILES_IN_BLOCK_M, 1, mm.BLOCK_K),
                )
                prev_rms_factors[...] = nl.copy(rms_factors[...], dtype=rms_factors.dtype)
                update_square_sums(lhs_block, square_sums)
                calculate_rms_factors(rms_factors, square_sums, scale=1 / mm.K, eps=eps)
                if block_id_K > 0:
                    result_blocks[...] = scale_prev_results(result_blocks, rms_factors, prev_rms_factors)
                scale_lhs(lhs_block, rms_factors)
                transpose_tiles_in_block(lhs_block)
                for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
                    rhs_block = load_tensor_block(
                        input_tensor=rhs,
                        ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                        load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                    )
                    matmul_blocks_tile_transposed_lhs(lhs_block, rhs_block, result_blocks, block_id_N)

            save_result_dma(result[batch_id], result_blocks, block_id_M)

    return result


def update_square_sums(lhs, square_sums):
    """
    Update the square sums for the lhs_block
    Args:
        lhs_block: 3D input tensor tile (TILE_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_K, TILE_K)
        square_sums: 3D tensor to store the square sums
    """
    TILE_M, TILES_IN_M, unity, K = lhs.shape
    assert unity == 1
    assert square_sums.shape == (TILE_M, TILES_IN_M, 1)
    i_lhs = nl.mgrid[0:TILE_M, 0:K]
    i_square_sums = nl.mgrid[0:TILE_M, 0:1]
    for tile_id_M in nl.affine_range(TILES_IN_M):
        tmp_square_sums = nl.ndarray((nl.par_dim(TILE_M), 1), dtype=square_sums.dtype, buffer=nl.sbuf)
        nisa.activation_reduce(
            op=nl.square, data=lhs[i_lhs.p, tile_id_M, 0, i_lhs.x], reduce_op=np.add, reduce_res=tmp_square_sums[...]
        )
        square_sums[i_square_sums.p, tile_id_M, i_square_sums.x] += tmp_square_sums[...]


def calculate_rms_factors(norm_factors, square_sums, scale: float, eps: float):
    """
    Calculate the norm factors for the square sums
    Args:
        square_sums: 3D tensor to store the square sums
        eps: RMS norm epsilon term
    """
    TILE_M, TILES_IN_BLOCK_M, unity = norm_factors.shape
    assert square_sums.shape == norm_factors.shape
    assert unity == 1
    i_norm = nl.mgrid[0:TILE_M, 0:1]
    for tile_id_M in nl.affine_range(TILES_IN_BLOCK_M):
        norm_factors[i_norm.p, tile_id_M, i_norm.x] = nisa.tensor_scalar(
            square_sums[i_norm.p, tile_id_M, i_norm.x], np.multiply, scale, op1=np.add, operand1=eps
        )
        norm_factors[i_norm.p, tile_id_M, i_norm.x] = nisa.activation(
            op=nl.sqrt, data=norm_factors[i_norm.p, tile_id_M, i_norm.x]
        )


def scale_lhs(lhs, rms_factors):
    TILE_M, TILES_IN_BLOCK_M, unity, BLOCK_K = lhs.shape
    assert rms_factors.shape == lhs.shape[:-1], f"rms_factors {rms_factors.shape} lhs {lhs.shape}"
    assert unity == 1
    i_lhs = nl.mgrid[0:TILE_M, 0:BLOCK_K]
    i_rms_factors = nl.mgrid[0:TILE_M, 0:1]
    for tile_id_M in nl.affine_range(TILES_IN_BLOCK_M):
        lhs[i_lhs.p, tile_id_M, 0, i_lhs.x] = nl.divide(
            lhs[i_lhs.p, tile_id_M, 0, i_lhs.x], rms_factors[i_rms_factors.p, tile_id_M, i_rms_factors.x]
        )


def scale_prev_results(result_blocks, rms_factors, prev_rms_factors):
    TILE_M, NUM_BLOCK_N, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILE_N = result_blocks.shape
    _TILE_M, _TILES_IN_BLOCK_M, unity = rms_factors.shape
    assert prev_rms_factors.shape == rms_factors.shape
    assert unity == 1
    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    i_rms = nl.mgrid[0:TILE_M, 0:1]
    tmp_scaling_factors = nl.ndarray((nl.par_dim(TILE_M), TILES_IN_BLOCK_M, 1), dtype=rms_factors.dtype, buffer=nl.sbuf)
    for tile_id_M in nl.affine_range(TILES_IN_BLOCK_M):
        tmp_scaling_factors[i_rms.p, tile_id_M, i_rms.x] = nl.divide(
            prev_rms_factors[i_rms.p, tile_id_M, i_rms.x], rms_factors[i_rms.p, tile_id_M, i_rms.x]
        )
    scaled_result_blocks = nl.ndarray(result_blocks.shape, dtype=result_blocks.dtype, buffer=nl.sbuf)
    for block_id_N in nl.affine_range(NUM_BLOCK_N):
        for tile_id_M in nl.affine_range(TILES_IN_BLOCK_M):
            for tile_id_N in nl.affine_range(TILES_IN_BLOCK_N):
                scaled_result_blocks[idx_res.p, block_id_N, tile_id_M, tile_id_N, idx_res.x] = nl.multiply(
                    result_blocks[idx_res.p, block_id_N, tile_id_M, tile_id_N, idx_res.x],
                    tmp_scaling_factors[i_rms.p, tile_id_M, i_rms.x],
                )
    return scaled_result_blocks


def matmul_blocks_tile_transposed_lhs(tileT_lhs_block, rhs_block, result_blocks, block_id_N):
    """
    Accumulate matmul result tiles between lhs and rhs into result_block
    LHS is transposed at the tile level.
    Note that this is not the same as lhsT.
    'matmul_block' module computes lhsT @ rhs.

    Args:
    tileT_lhs_block: TILE_M, TILE_K, TILES_IN_M, TILES_IN_K
    rhs_block: TILE_K, TILE_N, TILES_IN_K, TILES_IN_N
    result_block : TILE_M, TILE_N, TILES_IN_M, TILES_IN_N
    """
    TILE_M, TILES_IN_BLOCK_M, unity, BLOCK_K = tileT_lhs_block.shape
    TILE_K, TILES_IN_BLOCK_K, TILES_IN_BLOCK_N, TILE_N = rhs_block.shape
    _TILE_M, NUM_BLOCK_N, _TILES_IN_BLOCK_M, _TILES_IN_BLOCK_N, _TILE_N = result_blocks.shape
    assert unity == 1
    assert (
        TILE_K * TILES_IN_BLOCK_K == BLOCK_K
    ), f"K dimension mismatch: tileT_lhs_block {tileT_lhs_block.shape}. rhs_block {rhs_block.shape}."
    assert (
        TILE_M == _TILE_M and TILES_IN_BLOCK_M == _TILES_IN_BLOCK_M
    ), f"LHS and result shape mismatch: tileT_lhs_block {tileT_lhs_block.shape}. result_blocks {result_blocks.shape}."
    assert (
        TILE_N == _TILE_N and TILES_IN_BLOCK_N == _TILES_IN_BLOCK_N
    ), f"RHS and result shape mismatch: rhs_block {rhs_block.shape}. result_blocks {result_blocks.shape}."

    idx_lhs = nl.mgrid[0:TILE_M, 0:TILE_K]
    idx_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    for tile_id_M in nl.affine_range(TILES_IN_BLOCK_M):
        for tile_id_N in nl.affine_range(TILES_IN_BLOCK_N):
            result_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            """
            Use PSUM buffer to accumulate into a single hardware tile
            """
            for tile_id_K in nl.affine_range(TILES_IN_BLOCK_K):
                k_ofs = tile_id_K * TILE_K
                result_tile += nisa.nc_matmul(
                    tileT_lhs_block[idx_lhs.p, tile_id_M, 0, k_ofs + idx_lhs.x],
                    rhs_block[idx_rhs.p, tile_id_K, tile_id_N, idx_rhs.x],
                )
            # result_blocks[idx_res.p, block_id_N, tile_id_M, tile_id_N, idx_res.x] += result_tile[idx_res.p, idx_res.x]
            result_blocks[idx_res.p, block_id_N, tile_id_M, tile_id_N, idx_res.x] = nl.add(
                result_blocks[idx_res.p, block_id_N, tile_id_M, tile_id_N, idx_res.x], result_tile[idx_res.p, idx_res.x]
            )
