# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from src.allocation.utils import update_base_addr

import math
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.compiler as ncc
from neuronxcc.nki.language import par_dim
import numpy as np


@nki.jit
def weighted_rmsnorm(hidden, gamma, eps):
    """
    Calculate out_tensor = hidden/RMS(hidden) * gamma
    Where RMS(hidden) = sqrt((1/N) * sum(hidden * hidden))
    N = hidden.shape[1]
    Reduction (mean) is performed in the free (2nd) dimension

    Args:
        hidden (_type_): input hidden tensor
        gamma (_type_): gamma weight

    Returns:
        _type_: _description_
    """
    out_tensor = nl.ndarray(hidden.shape, dtype=hidden.dtype, buffer=nl.shared_hbm)

    # Make sure shapes match
    batch, batchless_shape = hidden.shape[0], hidden.shape[1:]
    seqlen, dim = batchless_shape
    _dim = gamma.shape[0]
    assert (
        dim == _dim
    ), f"gamma shape {gamma.shape} does not match hidden shape {hidden.shape}"

    # Calculate number of tiles
    pmax = nl.tile_size.pmax  # 128
    NUM_PAR_TILES = math.ceil(seqlen / pmax)

    # Generate tensor indices to index input tensor
    i_par = nl.arange(pmax)[:, None]
    i_dim = nl.arange(dim)[None, :]
    i_weight = nl.arange(1)[:, None]

    # Load RMSNorm weight once, reused by rows/tiles of hidden
    gamma_tile = nl.load(gamma.reshape((1, dim))[i_weight, i_dim])
    gamma_bcast = gamma_tile.broadcast_to((pmax, dim))

    for batch_index in nl.affine_range(batch):
        for par_tile_idx in nl.affine_range(NUM_PAR_TILES):
            # Load input data from external memory to on-chip memory
            par_mask = par_tile_idx * pmax + i_par < seqlen
            hidden_tile = nl.load(
                hidden[batch_index, par_tile_idx * pmax + i_par, i_dim], mask=par_mask
            )

            # Compute element-wise square of hidden
            in_square = nl.square(hidden_tile)

            # Calculate sum of squared elements, along last dimension
            square_sum = nl.sum(in_square, axis=[-1])

            # Scale and get a reciprocal
            mean = square_sum / dim
            mean = mean + eps

            # Take square root of mean and then reciprocal with
            # rsqrt API (one ISA instruction)
            rms_reciprocal = nl.rsqrt(mean)

            # Scale the input tensor
            out_tile = nl.multiply(hidden_tile, rms_reciprocal)

            # Multiply with the RMSNorm weight
            out_tile[...] = nl.multiply(out_tile, gamma_bcast, mask=par_mask)

            nl.store(
                out_tensor[batch_index, par_tile_idx * pmax + i_par, i_dim],
                value=out_tile,
                mask=par_mask,
            )

    return out_tensor


@nki.compiler.skip_middle_end_transformations
@nki.jit
def allocated_weighted_rmsnorm(hidden, gamma, hidden_buffer_degree, eps=1e-6):
    """
    Allocated kernel that computes RMSNorm(hidden). This kernel is designed to only handle fp16/bf16 tensor types.
    Internally, normalizations are cast to fp32 to avoid NaN errors.

    Args:
        hidden (_type_): Input tensor of the attention block in BSH layout
        eps (_type_, optional): RMS norm epsilon term. Defaults to 1e-6.
    """
    # Hidden should be in BSH layout.
    batch, batchless_shape = hidden.shape[0], hidden.shape[1:]
    seqlen, dim = batchless_shape
    assert (
        dim == gamma.shape[0]
    ), f"gamma {gamma.shape} does not match with hidden {hidden.shape}"
    assert gamma.dtype == hidden.dtype, "Gamma must match hidden dtype"
    assert dim <= 8192 and dim % 128 == 0, "Unsupported hidden dimension"

    norm_dtype = nl.float32

    out_tensor = nl.ndarray(
        (batch, seqlen, dim), dtype=hidden.dtype, buffer=nl.shared_hbm
    )

    pmax = nl.tile_size.pmax  # 128
    i_p = nl.arange(pmax)[:, None]
    i_f = nl.arange(dim)[None, :]
    i_w = nl.arange(1)[:, None]
    NUM_TILES = math.ceil(seqlen / pmax)
    TILES_INT = math.ceil(NUM_TILES / hidden_buffer_degree)
    scale = 1 / dim
    sbuf_base_addr = 0

    # Allocate broadcasted gamma buffer
    gamma_bcast = nl.ndarray(
        (par_dim(pmax), dim),
        dtype=gamma.dtype,
        buffer=ncc.sbuf.mod_alloc(base_addr=sbuf_base_addr),
    )
    sbuf_base_addr = update_base_addr(sbuf_base_addr, gamma_bcast, True)
    # FIXME: Change to more efficient NKI broadcast APIs when they become available.
    for i in nl.affine_range(pmax):
        gamma_bcast[i, i_f] = nl.load(
            gamma.reshape((1, dim))[i_w, i_f], dtype=gamma.dtype
        )

    bias_placeholder = nl.ndarray(
        (par_dim(pmax), 1),
        dtype=np.float32,
        buffer=ncc.sbuf.mod_alloc(base_addr=sbuf_base_addr),
    )
    sbuf_base_addr = update_base_addr(sbuf_base_addr, bias_placeholder, True)
    bias_placeholder[...] = 0

    for b in nl.affine_range(batch):
        for i in nl.affine_range(TILES_INT):
            # Buffer the input tensor
            in_bufs = nl.ndarray(
                (hidden_buffer_degree, par_dim(pmax), dim),
                dtype=hidden.dtype,
                buffer=ncc.sbuf.mod_alloc(
                    base_addr=sbuf_base_addr, num_free_tiles=(hidden_buffer_degree,)
                ),
            )
            sbuf_base_addr = update_base_addr(sbuf_base_addr, in_bufs, True)
            for i_interleave_grp in nl.affine_range(hidden_buffer_degree):
                seq_pos = (hidden_buffer_degree * i + i_interleave_grp) * pmax
                mask = seq_pos + i_p < seqlen
                in_bufs[i_interleave_grp] = nl.load(
                    hidden[b, seq_pos + i_p, i_f],
                    mask=mask,
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

                # Multiply with the RMSNorm weight
                output_tile[...] = nl.multiply(
                    output_tile[...], gamma_bcast[...], mask=mask
                )

                # Store result
                nl.store(
                    out_tensor[b, seq_pos + i_p, i_f],
                    value=output_tile,
                    mask=mask,
                )
                sbuf_base_addr = update_base_addr(sbuf_base_addr, output_tile, False)
                sbuf_base_addr = update_base_addr(sbuf_base_addr, square_sum, False)
                sbuf_base_addr = update_base_addr(sbuf_base_addr, act, False)
            sbuf_base_addr = update_base_addr(sbuf_base_addr, in_bufs, False)
    sbuf_base_addr = update_base_addr(sbuf_base_addr, bias_placeholder, False)
    sbuf_base_addr = update_base_addr(sbuf_base_addr, gamma_bcast, False)
    return out_tensor
