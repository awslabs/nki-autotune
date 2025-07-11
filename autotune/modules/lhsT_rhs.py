# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.compiler.backends.neuron.tensors import KernelHBMTensor
from neuronxcc.nki.typing import tensor

from autotune.modules.dma import load_tensor_block, save_result_acc, save_result_block, save_result_dma
from autotune.modules.matmul import GEMMCompatibility, matmul_blocks_lhsT


@nki.jit
def lhsT_rhs_gemm_general(
    lhsT: tensor,
    rhs: tensor,
    NUM_BLOCK_M: int,
    NUM_BLOCK_N: int,
    NUM_BLOCK_K: int,
    loop_order: str,
    tensor_positions: Tuple[int, int, int],
):
    """
    NOTE
    1. result_block init is before K loop.
    2. matmul_blocks_lhsT is after lhsT_block, rhs_block loads.
    3. save_result_block is after K loop.

    position = 0
    loop_0
        position = 1
        loop_1
            position = 2
            loop_2
                position = 3
    """
    mm = GEMMCompatibility(transposed_lhs=True)
    mm(
        input_tensors=(lhsT, rhs),
        kernel_kwargs={"NUM_BLOCK_M": NUM_BLOCK_M, "NUM_BLOCK_N": NUM_BLOCK_N, "NUM_BLOCK_K": NUM_BLOCK_K},
    )
    result = nl.ndarray((mm.M, mm.N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    if len(loop_order) != 3 or sorted(loop_order) != sorted("MNK"):
        raise ValueError(f"Invalid loop_order: {loop_order}. Must contain exactly M, N, and K.")
    M_position = loop_order.index("M")
    N_position = loop_order.index("N")
    K_position = loop_order.index("K")
    lhsT_block_position, rhs_block_position, result_block_position = tensor_positions
    matmul_position = max(lhsT_block_position, rhs_block_position)

    # TODO: can we relax this constraint?
    assert (
        result_block_position <= K_position and result_block_position <= matmul_position
    ), f"result_block init must be before K loop and matmul. Received result_block_position {result_block_position}, K_position {K_position}, matmul_position {matmul_position}."

    curr_position = 0
    curr_block_ids = []
    if result_block_position == curr_position:
        result_block_shape = get_block_shape(mm, loop_order, ("M", "N"), curr_position)
        result_block = nl.zeros(result_block_shape, dtype=result.dtype, buffer=nl.sbuf)
    if lhsT_block_position == curr_position:
        lhsT_block_shape = get_block_shape(mm, loop_order, ("K", "M"), curr_position)
        lhsT_ofs = get_block_ofs(mm, loop_order, ("K", "M"), curr_position, curr_block_ids)
        lhsT_block = load_tensor_block(input_tensor=lhsT, ofs=lhsT_ofs, load_shape=lhsT_block_shape)
    if rhs_block_position == curr_position:
        rhs_block_shape = get_block_shape(mm, loop_order, ("K", "N"), curr_position)
        rhs_ofs = get_block_ofs(mm, loop_order, ("K", "N"), curr_position, curr_block_ids)
        rhs_block = load_tensor_block(input_tensor=rhs, ofs=rhs_ofs, load_shape=rhs_block_shape)
    if matmul_position == curr_position:
        result_ofs = get_block_ofs(mm, loop_order, ("M", "N"), curr_position, curr_block_ids)
        matmul_blocks_lhsT(lhsT_block, rhs_block, result_block, ofs=result_ofs)
    for block_id_0 in nl.affine_range(getattr(mm, f"NUM_BLOCK_{loop_order[0]}")):
        curr_position = 1
        curr_block_ids = [block_id_0]
        if result_block_position == curr_position:
            result_block_shape = get_block_shape(mm, loop_order, ("M", "N"), curr_position)
            result_block = nl.zeros(result_block_shape, dtype=result.dtype, buffer=nl.sbuf)
        if lhsT_block_position == curr_position:
            lhsT_block_shape = get_block_shape(mm, loop_order, ("K", "M"), curr_position)
            lhsT_ofs = get_block_ofs(mm, loop_order, ("K", "M"), curr_position, curr_block_ids)
            lhsT_block = load_tensor_block(input_tensor=lhsT, ofs=lhsT_ofs, load_shape=lhsT_block_shape)
        if rhs_block_position == curr_position:
            rhs_block_shape = get_block_shape(mm, loop_order, ("K", "N"), curr_position)
            rhs_ofs = get_block_ofs(mm, loop_order, ("K", "N"), curr_position, curr_block_ids)
            rhs_block = load_tensor_block(input_tensor=rhs, ofs=rhs_ofs, load_shape=rhs_block_shape)
        if matmul_position == curr_position:
            result_ofs = get_block_ofs(mm, loop_order, ("M", "N"), curr_position, curr_block_ids)
            matmul_blocks_lhsT(lhsT_block, rhs_block, result_block, ofs=result_ofs)

        for block_id_1 in nl.affine_range(getattr(mm, f"NUM_BLOCK_{loop_order[1]}")):
            curr_position = 2
            curr_block_ids = [block_id_0, block_id_1]
            if result_block_position == curr_position:
                result_block_shape = get_block_shape(mm, loop_order, ("M", "N"), curr_position)
                result_block = nl.zeros(result_block_shape, dtype=result.dtype, buffer=nl.sbuf)
            if lhsT_block_position == curr_position:
                lhsT_block_shape = get_block_shape(mm, loop_order, ("K", "M"), curr_position)
                lhsT_ofs = get_block_ofs(mm, loop_order, ("K", "M"), curr_position, curr_block_ids)
                lhsT_block = load_tensor_block(input_tensor=lhsT, ofs=lhsT_ofs, load_shape=lhsT_block_shape)
            if rhs_block_position == curr_position:
                rhs_block_shape = get_block_shape(mm, loop_order, ("K", "N"), curr_position)
                rhs_ofs = get_block_ofs(mm, loop_order, ("K", "N"), curr_position, curr_block_ids)
                rhs_block = load_tensor_block(input_tensor=rhs, ofs=rhs_ofs, load_shape=rhs_block_shape)
            if matmul_position == curr_position:
                result_ofs = get_block_ofs(mm, loop_order, ("M", "N"), curr_position, curr_block_ids)
                matmul_blocks_lhsT(lhsT_block, rhs_block, result_block, ofs=result_ofs)

            for block_id_2 in nl.affine_range(getattr(mm, f"NUM_BLOCK_{loop_order[2]}")):
                curr_position = 3
                curr_block_ids = [block_id_0, block_id_1, block_id_2]
                if lhsT_block_position == curr_position:
                    lhsT_block_shape = get_block_shape(mm, loop_order, ("K", "M"), curr_position)
                    lhsT_ofs = get_block_ofs(mm, loop_order, ("K", "M"), curr_position, curr_block_ids)
                    lhsT_block = load_tensor_block(input_tensor=lhsT, ofs=lhsT_ofs, load_shape=lhsT_block_shape)
                if rhs_block_position == curr_position:
                    rhs_block_shape = get_block_shape(mm, loop_order, ("K", "N"), curr_position)
                    rhs_ofs = get_block_ofs(mm, loop_order, ("K", "N"), curr_position, curr_block_ids)
                    rhs_block = load_tensor_block(input_tensor=rhs, ofs=rhs_ofs, load_shape=rhs_block_shape)
                if matmul_position == curr_position:
                    result_ofs = get_block_ofs(mm, loop_order, ("M", "N"), curr_position, curr_block_ids)
                    matmul_blocks_lhsT(lhsT_block, rhs_block, result_block, ofs=result_ofs)
            curr_position = 2

            if result_block_position == curr_position:
                if M_position == 0:
                    tile_index_ofs = (block_id_0 * mm.TILES_IN_BLOCK_M, block_id_1 * mm.TILES_IN_BLOCK_N)
                if N_position == 0:
                    tile_index_ofs = (block_id_1 * mm.TILES_IN_BLOCK_M, block_id_0 * mm.TILES_IN_BLOCK_N)
                save_result_block(result, result_block, tile_index_ofs=tile_index_ofs)

        curr_position = 1
        if result_block_position == curr_position:
            if M_position == 0:
                tile_index_ofs = (block_id_0 * mm.TILES_IN_BLOCK_M, 0)
            if N_position == 0:
                tile_index_ofs = (0, block_id_0 * mm.TILES_IN_BLOCK_N)
            save_result_block(result, result_block, tile_index_ofs=tile_index_ofs)

    curr_position = 0
    if result_block_position == curr_position:
        save_result_block(result, result_block, tile_index_ofs=(0, 0))
    return result


def get_block_shape(
    mm: GEMMCompatibility, loop_order: str, dims: Tuple[str, str], curr_position: int
) -> Tuple[int, int, int, int]:
    num_blocks = []
    for dim in dims:
        dim_position = loop_order.index(dim)
        if dim_position < curr_position:
            num_block = 1
        else:
            num_block = getattr(mm, f"NUM_BLOCK_{dim}")
        num_blocks.append(num_block)
    block_shape = (
        getattr(mm, f"TILE_{dims[0]}"),
        num_blocks[0] * getattr(mm, f"TILES_IN_BLOCK_{dims[0]}"),
        num_blocks[1] * getattr(mm, f"TILES_IN_BLOCK_{dims[1]}"),
        getattr(mm, f"TILE_{dims[1]}"),
    )
    return block_shape


def get_block_ofs(
    mm: GEMMCompatibility, loop_order: str, dims: Tuple[str, str], curr_position: int, curr_block_ids
) -> Tuple[int, int]:
    block_ofs = []
    for dim in dims:
        dim_position = loop_order.index(dim)
        if dim_position < curr_position:
            ofs = curr_block_ids[dim_position] * getattr(mm, f"BLOCK_{dim}")
        else:
            ofs = 0
        block_ofs.append(ofs)
    block_ofs = tuple(block_ofs)
    return block_ofs


@nki.jit
def lhsT_rhs_gemm(lhsT: tensor, rhs: tensor, NUM_BLOCK_M: int, NUM_BLOCK_N: int, NUM_BLOCK_K: int, template: str):
    mm = GEMMCompatibility(transposed_lhs=True)
    mm(
        input_tensors=(lhsT, rhs),
        kernel_kwargs={"NUM_BLOCK_M": NUM_BLOCK_M, "NUM_BLOCK_N": NUM_BLOCK_N, "NUM_BLOCK_K": NUM_BLOCK_K},
    )
    result = nl.ndarray((mm.M, mm.N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    if template == "MNK":
        matmul_MNK(lhsT, rhs, mm, result)
    elif template == "MKN":
        matmul_MKN(lhsT, rhs, mm, result)
    elif template == "NMK":
        matmul_NMK(lhsT, rhs, mm, result)
    elif template == "NKM":
        matmul_NKM(lhsT, rhs, mm, result)
    elif template == "KMN":
        matmul_KMN(lhsT, rhs, mm, result)
    elif template == "KNM":
        matmul_KNM(lhsT, rhs, mm, result)
    else:
        raise NotImplementedError(f"Loop order {template} GEMM does not exist.")
    return result


def matmul_NMK(lhsT: tensor, rhs: tensor, mm: GEMMCompatibility, result: KernelHBMTensor):
    for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
        for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
            result_block = nl.zeros(
                (nl.par_dim(mm.TILE_M), mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                dtype=result.dtype,
                buffer=nl.sbuf,
            )
            for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):
                lhsT_block = load_tensor_block(
                    input_tensor=lhsT,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_M * mm.BLOCK_M),
                    load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_M, mm.TILE_M),
                )
                rhs_block = load_tensor_block(
                    input_tensor=rhs,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                    load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                )
                matmul_blocks_lhsT(lhsT_block, rhs_block, result_block, ofs=(0, 0))
            save_result_block(result, result_block, m_ofs=block_id_M * mm.BLOCK_M, n_ofs=block_id_N * mm.BLOCK_N)
    return result


def matmul_MNK(lhsT: tensor, rhs: tensor, mm: GEMMCompatibility, result: KernelHBMTensor):
    for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
        for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
            result_block = nl.zeros(
                (nl.par_dim(mm.TILE_M), mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                dtype=result.dtype,
                buffer=nl.sbuf,
            )
            for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):
                lhsT_block = load_tensor_block(
                    input_tensor=lhsT,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_M * mm.BLOCK_M),
                    load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_M, mm.TILE_M),
                )
                rhs_block = load_tensor_block(
                    input_tensor=rhs,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                    load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                )
                matmul_blocks_lhsT(lhsT_block, rhs_block, result_block, ofs=(0, 0))
            save_result_block(result, result_block, m_ofs=block_id_M * mm.BLOCK_M, n_ofs=block_id_N * mm.BLOCK_N)
    return result


def matmul_KMN(lhsT: tensor, rhs: tensor, mm: GEMMCompatibility, result: KernelHBMTensor):
    result_block = nl.zeros(
        (nl.par_dim(mm.TILE_M), mm.TILES_IN_M, mm.TILES_IN_N, mm.TILE_N), dtype=result.dtype, buffer=nl.sbuf
    )
    for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):
        for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
            lhsT_block = load_tensor_block(
                input_tensor=lhsT,
                ofs=(block_id_K * mm.BLOCK_K, block_id_M * mm.BLOCK_M),
                load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_M, mm.TILE_M),
            )
            for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
                rhs_block = load_tensor_block(
                    input_tensor=rhs,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                    load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                )
                matmul_blocks_lhsT(
                    lhsT_block,
                    rhs_block,
                    result_block,
                    ofs=(block_id_M * mm.TILES_IN_BLOCK_M, block_id_N * mm.TILES_IN_BLOCK_N),
                )
    save_result_block(result, result_block, m_ofs=0, n_ofs=0)
    return result


def matmul_KNM(lhsT: tensor, rhs: tensor, mm: GEMMCompatibility, result: KernelHBMTensor):

    result_tiles = nl.zeros(
        (
            mm.NUM_BLOCK_K,
            mm.NUM_BLOCK_M,
            mm.NUM_BLOCK_N,
            mm.TILES_IN_BLOCK_M,
            mm.TILES_IN_BLOCK_N,
            nl.par_dim(mm.TILE_M),
            mm.TILE_N,
        ),
        dtype=result.dtype,
        buffer=nl.sbuf,
    )

    for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):
        for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
            # TILES_IN_BLOCK_K, TILE_K, BLOCK_N
            rhs_tiles = load_tensor_block(
                input_tensor=rhs,
                ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
            )
            for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_M
                lhsT_tiles = load_tensor_block(
                    input_tensor=lhsT,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_M * mm.BLOCK_M),
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_M),
                )

                matmul_block(lhsT_tiles, rhs_tiles, result_tiles[block_id_K, block_id_M, block_id_N])

    save_result_acc(result, result_tiles, mm.BLOCK_M, mm.BLOCK_N)
    return result


def matmul_NKM(lhsT: tensor, rhs: tensor, mm: GEMMCompatibility, result: KernelHBMTensor):

    # Blocking N dimension (the RHS free dimension)
    for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
        result_tiles = nl.zeros(
            (mm.NUM_BLOCK_M, mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, nl.par_dim(mm.TILE_M), mm.TILE_N),
            dtype=result.dtype,
            buffer=nl.sbuf,
        )

        # Blocking K dimension (the contraction dimension)
        # Use `sequential_range` because we do not want the compiler to change this loop by,
        # for example, vectorizing it
        for block_id_K in nl.sequential_range(mm.NUM_BLOCK_K):
            # Loading tiles from rhs
            # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
            rhs_tiles = load_tensor_block(
                input_tensor=rhs,
                ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
            )

            # Blocking M dimension (the LHS free dimension)
            for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
                # Loading tiles from lhsT
                lhsT_tiles = load_tensor_block(
                    input_tensor=lhsT,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_M * mm.BLOCK_M),
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_M),
                )

                matmul_block(lhsT_tiles, rhs_tiles, result_tiles[block_id_M])

        for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
            """
            save_result_block(
                result,
                result_tiles[block_id_M],
                m_ofs=block_id_M * mm.BLOCK_M,
                n_ofs=block_id_N * mm.BLOCK_N,
            )
            """
            save_result_dma(
                result,
                result_tiles,
                block_id_M,
                m_ofs=block_id_M * mm.TILES_IN_BLOCK_M,
                n_ofs=block_id_N * mm.BLOCK_N,
                TILE_K=mm.TILE_K,
            )

    return result


def matmul_MKN(lhsT: tensor, rhs: tensor, mm: GEMMCompatibility, result: KernelHBMTensor):
    # Blocking M dimension (the LHS free dimension)
    for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
        result_tiles = nl.zeros(
            (mm.NUM_BLOCK_N, mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, nl.par_dim(mm.TILE_M), mm.TILE_N),
            dtype=result.dtype,
            buffer=nl.sbuf,
        )

        # Blocking K dimension (the contraction dimension)
        # Use `sequential_range` because we do not want the compiler to change this loop by,
        # for example, vectorizing it
        for block_id_K in nl.sequential_range(mm.NUM_BLOCK_K):
            # Loading tiles from lhsT
            lhsT_tiles = load_tensor_block(
                input_tensor=lhsT,
                ofs=(block_id_K * mm.BLOCK_K, block_id_M * mm.BLOCK_M),
                load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_M),
            )

            # Blocking N dimension (the RHS free dimension)
            for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
                # Loading tiles from rhs
                # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
                rhs_tiles = load_tensor_block(
                    input_tensor=rhs,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
                )

                matmul_block(lhsT_tiles, rhs_tiles, result_tiles[block_id_N])

        for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
            save_result_dma(
                result,
                result_tiles,
                block_id_N,
                m_ofs=block_id_M * mm.TILES_IN_BLOCK_M,
                n_ofs=block_id_N * mm.BLOCK_N,
                TILE_K=mm.TILE_K,
            )

    return result
