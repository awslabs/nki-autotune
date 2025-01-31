"""
Copyright (c) 2024, Amazon.com. All Rights Reserved

kernels - Fused normalization with linear layers

"""

import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.compiler as ncc
import math
import numpy as np
from neuronxcc import nki
from neuronxcc.nki.language import par_dim


def update_base_addr(base_addr, tensor, advance: bool):
    if tensor.ndim == 2:
        # pardim, fdim
        buf_size = tensor.shape[1] * tensor.itemsize
    elif tensor.ndim == 3:
        # block_dim, pardim, fdim
        buf_size = tensor.shape[0] * tensor.shape[2] * tensor.itemsize
    else:
        raise NotImplementedError(
            f"Buffer size for tensor shape {tensor.shape} is unknown"
        )
    if advance:
        next_base_addr = base_addr + buf_size
        print(f"Allocate buf @ {base_addr} -> {next_base_addr}")
    else:
        next_base_addr = base_addr - buf_size
        # print(f"Restore buf @ {base_addr} -> {next_base_addr}")
    return next_base_addr


@nki.jit
def allocated_fused_rms_norm_qkv(
    hidden, weights, hidden_buffer_degree, norm_dtype=nl.float32, eps=1e-6
):
    """
    Allocated kernel that computes RMSNorm(hidden) @ wQKV. This kernel is designed to only handle fp16/bf16 tensor types.
    Internally, normalizations are cast to fp32 to avoid NaN errors.

    Args:
        hidden (_type_): Input tensor of the attention block in BSH layout
        weights (_type_): Fused QKV linear weights, assumed to be eltwise-multiplied with RMS norm weight vector (gamma)
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
    print(f"hidden_buffer_degree = {hidden_buffer_degree}")
    TILES_INT = math.ceil(NUM_TILES / hidden_buffer_degree)
    scale = 1 / dim
    sbuf_base_addr = 0

    iden_x, iden_y = nl.mgrid[0:pmax, 0:128]

    identity_a = nl.shared_constant(
        np.identity(n=128, dtype=np.int8), dtype=hidden.dtype
    )
    identity_tensor = nl.ndarray(
        (par_dim(pmax), 128),
        dtype=weights.dtype,
        buffer=ncc.sbuf.mod_alloc(base_addr=sbuf_base_addr),
    )
    sbuf_base_addr = update_base_addr(sbuf_base_addr, identity_tensor, True)
    identity_tensor[iden_x, iden_y] = nl.load(identity_a, dtype=weights.dtype)
    bias_placeholder = nl.ndarray(
        (par_dim(pmax), 1),
        dtype=np.float32,
        buffer=ncc.sbuf.mod_alloc(base_addr=sbuf_base_addr),
    )
    sbuf_base_addr = update_base_addr(sbuf_base_addr, bias_placeholder, True)
    bias_placeholder[...] = 0

    for b in nl.affine_range(batch):
        weights_buffer = nl.ndarray(
            (M, par_dim(pmax), fmax),
            dtype=weights.dtype,
            buffer=ncc.sbuf.mod_alloc(
                base_addr=sbuf_base_addr,
                num_free_tiles=(M,),
            ),
        )
        sbuf_base_addr = update_base_addr(sbuf_base_addr, weights_buffer, True)
        # Preload the entire weights tensor. everything fits in SBUF for LLaMA 3.1 70B
        for m in nl.affine_range(M):
            weights_buffer[m, i_rhs.p, i_rhs.x] = nl.load(
                weights[m * pmax + i_rhs.p, i_rhs.x],
                mask=(m * pmax + i_rhs.p < dim) & (i_rhs.x < head_dim),
            )
        for i in nl.affine_range(TILES_INT):
            # Double buffer the input tensor
            in_bufs = nl.ndarray(
                (hidden_buffer_degree, par_dim(pmax), dim),
                dtype=hidden.dtype,
                buffer=ncc.sbuf.mod_alloc(
                    base_addr=sbuf_base_addr, num_free_tiles=(hidden_buffer_degree,)
                ),
            )
            sbuf_base_addr = update_base_addr(sbuf_base_addr, in_bufs, True)
            for i_interleave_grp in nl.affine_range(hidden_buffer_degree):
                in_bufs[i_interleave_grp] = nl.load(
                    hidden[
                        b, (hidden_buffer_degree * i + i_interleave_grp) * pmax + ix, iy
                    ],
                    mask=(hidden_buffer_degree * i + i_interleave_grp) * pmax + ix
                    < seqlen,
                )
                act = nl.ndarray(
                    (par_dim(pmax), dim),
                    dtype=norm_dtype,
                    buffer=ncc.sbuf.mod_alloc(base_addr=sbuf_base_addr),
                )
                sbuf_base_addr = update_base_addr(sbuf_base_addr, act, True)

                # Write the RMS and RMS Reciprocal tensors back out here, in-place
                square_sum = nl.ndarray(
                    (par_dim(pmax), 1),
                    dtype=norm_dtype,
                    buffer=ncc.sbuf.mod_alloc(base_addr=sbuf_base_addr),
                )
                sbuf_base_addr = update_base_addr(sbuf_base_addr, square_sum, True)

                # Write the output of RMS and RMS^T (in-place) out to here
                out_tile = nl.ndarray(
                    (par_dim(pmax), dim),
                    dtype=weights.dtype,
                    buffer=ncc.sbuf.mod_alloc(base_addr=sbuf_base_addr),
                )
                sbuf_base_addr = update_base_addr(sbuf_base_addr, out_tile, True)

                # Store the final output tiles to here before sending back to DRAM
                output_sbuf = nl.ndarray(
                    (par_dim(pmax), fmax),
                    dtype=weights.dtype,
                    buffer=ncc.sbuf.mod_alloc(base_addr=sbuf_base_addr),
                )
                sbuf_base_addr = update_base_addr(sbuf_base_addr, output_sbuf, True)

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
                    TILES_IN_BLOCK = fmax // pmax
                    for j in nl.affine_range(TILES_IN_BLOCK):
                        transpose_res_psum[m, i_lhs.p, j * pmax + i_lhs.x] = (
                            nisa.nc_matmul(
                                out_tile[
                                    i_lhs.p, (m * TILES_IN_BLOCK + j) * pmax + i_lhs.x
                                ],
                                identity_tensor[...],
                                is_transpose=True,
                            )
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
                        out_tile[i_lhs.p, m * pmax + i_lhs.x],
                        weights_buffer[m, i_rhs.p, i_rhs.x],
                    )

                output_sbuf[...] = nl.copy(res_psum[0], dtype=out_tensor.dtype)
                nl.store(
                    out_tensor[
                        b,
                        (hidden_buffer_degree * i + i_interleave_grp) * pmax + i_res.p,
                        i_res.x,
                    ],
                    value=output_sbuf,
                    mask=(
                        (hidden_buffer_degree * i + i_interleave_grp) * pmax + i_res.p
                        < seqlen
                    )
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


@nki.jit
def allocated_rms_norm(hidden, hidden_buffer_degree, norm_dtype=nl.float32, eps=1e-6):
    """
    Allocated kernel that computes RMSNorm(hidden). This kernel is designed to only handle fp16/bf16 tensor types.
    Internally, normalizations are cast to fp32 to avoid NaN errors.

    Args:
        hidden (_type_): Input tensor of the attention block in BSH layout
        norm_dtype (_type_, optional): Data type for RMS norm, should be f32 to avoid NaN. Defaults to nl.float32.
        eps (_type_, optional): RMS norm epsilon term. Defaults to 1e-6.
    """
    # Hidden should be in BSH layout.
    batch, batchless_shape = hidden.shape[0], hidden.shape[1:]
    seqlen, dim = batchless_shape

    assert dim <= 8192 and dim & 128 == 0, "Unsupported hidden dimension"

    out_tensor = nl.ndarray(
        (batch, seqlen, dim), dtype=hidden.dtype, buffer=nl.shared_hbm
    )

    pmax = nl.tile_size.pmax  # 128
    ix, iy = nl.mgrid[0:pmax, 0:dim]
    NUM_TILES = math.ceil(seqlen / pmax)
    TILES_INT = math.ceil(NUM_TILES / hidden_buffer_degree)
    scale = 1 / dim
    sbuf_base_addr = 0

    bias_placeholder = nl.ndarray(
        (par_dim(pmax), 1),
        dtype=np.float32,
        buffer=ncc.sbuf.mod_alloc(base_addr=sbuf_base_addr),
    )
    sbuf_base_addr = update_base_addr(sbuf_base_addr, bias_placeholder, True)
    bias_placeholder[...] = 0

    for b in nl.affine_range(batch):
        for i in nl.affine_range(TILES_INT):
            # Double buffer the input tensor
            in_bufs = nl.ndarray(
                (hidden_buffer_degree, par_dim(pmax), dim),
                dtype=hidden.dtype,
                buffer=ncc.sbuf.mod_alloc(
                    base_addr=sbuf_base_addr, num_free_tiles=(hidden_buffer_degree,)
                ),
            )
            sbuf_base_addr = update_base_addr(sbuf_base_addr, in_bufs, True)
            for i_interleave_grp in nl.affine_range(hidden_buffer_degree):
                in_bufs[i_interleave_grp] = nl.load(
                    hidden[
                        b, (hidden_buffer_degree * i + i_interleave_grp) * pmax + ix, iy
                    ],
                    mask=(hidden_buffer_degree * i + i_interleave_grp) * pmax + ix
                    < seqlen,
                )
                act = nl.ndarray(
                    (par_dim(pmax), dim),
                    dtype=norm_dtype,
                    buffer=ncc.sbuf.mod_alloc(base_addr=sbuf_base_addr),
                )
                sbuf_base_addr = update_base_addr(sbuf_base_addr, act, True)

                # Write the RMS and RMS Reciprocal tensors back out here, in-place
                square_sum = nl.ndarray(
                    (par_dim(pmax), 1),
                    dtype=norm_dtype,
                    buffer=ncc.sbuf.mod_alloc(base_addr=sbuf_base_addr),
                )
                sbuf_base_addr = update_base_addr(sbuf_base_addr, square_sum, True)

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

                # Apply normalization
                output_tile = nl.ndarray(
                    (par_dim(pmax), dim),
                    dtype=hidden.dtype,
                    buffer=ncc.sbuf.mod_alloc(base_addr=sbuf_base_addr),
                )
                sbuf_base_addr = update_base_addr(sbuf_base_addr, output_tile, True)

                output_tile[...] = nl.multiply(
                    in_bufs[i_interleave_grp],
                    square_sum[...],
                    dtype=hidden.dtype,
                )
                # Store result
                nl.store(
                    out_tensor[
                        b,
                        (hidden_buffer_degree * i + i_interleave_grp) * pmax + ix,
                        iy,
                    ],
                    value=output_tile,
                    mask=(hidden_buffer_degree * i + i_interleave_grp) * pmax + ix
                    < seqlen,
                )
                sbuf_base_addr = update_base_addr(sbuf_base_addr, output_tile, False)
                sbuf_base_addr = update_base_addr(sbuf_base_addr, square_sum, False)
                sbuf_base_addr = update_base_addr(sbuf_base_addr, act, False)
            sbuf_base_addr = update_base_addr(sbuf_base_addr, in_bufs, False)
    sbuf_base_addr = update_base_addr(sbuf_base_addr, bias_placeholder, False)
    return out_tensor
