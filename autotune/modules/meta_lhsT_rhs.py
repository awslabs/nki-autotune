# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Dict, Tuple

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor

from autotune.modules.dma import load_tensor_block, save_result_block
from autotune.modules.layout import get_block_ofs, get_block_shape
from autotune.modules.matmul import GEMMCompatibility, matmul_blocks_lhsT


def process_template(loop_order: str, tensor_positions: Dict[str, int]) -> Tuple[Dict, Dict]:
    """
    Constraints:
        1. Loop order must contain exactly the characters 'M', 'N', and 'K'
        2. lhsT_block and rhs_block loads must be on the same side of K loop
        3. result_block_position = K_position - 1
        4. matmul_position = max(lhsT_block_position, rhs_block_position)
        5. save_position = K_position - 1, mirroring result_block_position

    Position reference:
    position = -1
    loop_0:
        position = 0
        loop_1:
            position = 1
            loop_2:
                position = 2
            position = 1
        position = 0
    position = -1
    """
    assert sorted(loop_order) == sorted("MNK"), f"Invalid loop_order: {loop_order}. Must contain exactly M, N, and K."
    K_position = loop_order.index("K")
    lhsT_block_position = tensor_positions["lhsT_block"]
    rhs_block_position = tensor_positions["rhs_block"]
    assert (lhsT_block_position < K_position and rhs_block_position < K_position) or (
        lhsT_block_position >= K_position and rhs_block_position >= K_position
    ), f"lhsT_block and rhs_block must be on the same side of K loop. Received lhsT_block_position {lhsT_block_position}, rhs_block_position {rhs_block_position}, K_position {K_position}."
    tensor_positions["result_block"] = K_position - 1
    tensor_positions["matmul"] = max(lhsT_block_position, rhs_block_position)
    tensor_positions["save"] = K_position - 1
    loop_order_dict = {}
    for index, value in enumerate(loop_order):
        loop_order_dict[index] = value
        loop_order_dict[value] = index
    return loop_order_dict, tensor_positions


@nki.jit
def lhsT_rhs_gemm_general(
    lhsT: tensor,
    rhs: tensor,
    NUM_BLOCK_M: int,
    NUM_BLOCK_N: int,
    NUM_BLOCK_K: int,
    loop_order_str: str,
    tensor_positions: Dict[str, int],
):
    """
    Perform general matrix multiplication between a transposed left-hand side matrix and right-hand side matrix
    using block-based computation with customizable loop ordering and tensor load/store positions.

    This function implements a blocked GEMM operation that computes result = lhsT^T @ rhs, where tensors
    are processed in blocks to optimize memory access patterns. The loop ordering and tensor operation
    positions can be configured to explore different performance characteristics.

    Args:
        lhsT: tensor of shape (K, M) - Transposed left-hand side input matrix
        rhs: tensor of shape (K, N) - Right-hand side input matrix
        NUM_BLOCK_M: int - Number of blocks along M dimension
        NUM_BLOCK_N: int - Number of blocks along N dimension
        NUM_BLOCK_K: int - Number of blocks along K dimension
        loop_order: Dict - Dict specifying the loop ordering (e.g., "MNK")
        tensor_positions: Dict[str, int] - Positions where tensor operations occur:
            (lhsT_block, rhs_block)

    Returns:
        tensor of shape (M, N) - Result of the matrix multiplication
    FIXME: compute the global coordinate, use the global coordinate to update tensors
    """
    kernel_kwargs = {"NUM_BLOCK_M": NUM_BLOCK_M, "NUM_BLOCK_N": NUM_BLOCK_N, "NUM_BLOCK_K": NUM_BLOCK_K}
    mm = GEMMCompatibility(transposed_lhs=True)
    mm((lhsT, rhs), kernel_kwargs)
    result = nl.ndarray((mm.M, mm.N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    loop_order, op_positions = process_template(loop_order_str, tensor_positions)

    position = -1
    curr_block_ids = []
    if op_positions["result_block"] == position:
        result_block = init_result_block(mm, loop_order, position, dtype=result.dtype)
    for block_id_0 in nl.affine_range(getattr(mm, f"NUM_BLOCK_{loop_order[0]}")):
        position = 0
        curr_block_ids.append(block_id_0)
        if op_positions["lhsT_block"] == position:
            lhsT_block, lhsT_block_ofs = load_input_block(lhsT, ("K", "M"), mm, loop_order, position, curr_block_ids)
        if op_positions["rhs_block"] == position:
            rhs_block, rhs_block_ofs = load_input_block(rhs, ("K", "N"), mm, loop_order, position, curr_block_ids)
        if op_positions["result_block"] == position:
            result_block = init_result_block(mm, loop_order, position, dtype=result.dtype)
        for block_id_1 in nl.affine_range(getattr(mm, f"NUM_BLOCK_{loop_order[1]}")):
            position = 1
            curr_block_ids.append(block_id_1)
            if op_positions["lhsT_block"] == position:
                lhsT_block, lhsT_block_ofs = load_input_block(
                    lhsT, ("K", "M"), mm, loop_order, position, curr_block_ids
                )
            if op_positions["rhs_block"] == position:
                rhs_block, rhs_block_ofs = load_input_block(rhs, ("K", "N"), mm, loop_order, position, curr_block_ids)
            if op_positions["result_block"] == position:
                result_block = init_result_block(mm, loop_order, position, dtype=result.dtype)
            for block_id_2 in nl.affine_range(getattr(mm, f"NUM_BLOCK_{loop_order[2]}")):
                position = 2
                curr_block_ids.append(block_id_2)
                if op_positions["lhsT_block"] == position:
                    lhsT_block, lhsT_block_ofs = load_input_block(
                        lhsT, ("K", "M"), mm, loop_order, position, curr_block_ids
                    )
                if op_positions["rhs_block"] == position:
                    rhs_block, rhs_block_ofs = load_input_block(
                        rhs, ("K", "N"), mm, loop_order, position, curr_block_ids
                    )
                if op_positions["matmul"] == position:
                    """
                    FIXME:
                    For matmul ofs, it should be the relative offset from the input tensors.
                    Not the global coordinates.
                    get_block_ofs calculates the global ofs.
                    """
                    result_ofs = get_block_ofs(mm, loop_order, ("M", "N"), position, curr_block_ids)
                    matmul_blocks_lhsT(lhsT_block, rhs_block, result_block, ofs=result_ofs)
            position = 1
            curr_block_ids.pop(-1)
            if op_positions["save"] == position:
                if loop_order["M"] == 0:
                    tile_index_ofs = (block_id_0 * mm.TILES_IN_BLOCK_M, 0)
                if loop_order["N"] == 0:
                    tile_index_ofs = (0, block_id_0 * mm.TILES_IN_BLOCK_N)
                save_result_block(result, result_block, tile_index_ofs=tile_index_ofs)
            if op_positions["matmul"] == position:
                result_ofs = get_block_ofs(mm, loop_order, ("M", "N"), position, curr_block_ids)
                matmul_blocks_lhsT(lhsT_block, rhs_block, result_block, ofs=result_ofs)
        position = 0
        curr_block_ids.pop(-1)
        # Inlined maybe_save for position 0
        if op_positions["save"] == position:
            tile_index_ofs = (0, 0)
            save_result_block(result, result_block, tile_index_ofs=tile_index_ofs)
        if op_positions["matmul"] == position:
            result_ofs = get_block_ofs(mm, loop_order, ("M", "N"), position, curr_block_ids)
            matmul_blocks_lhsT(lhsT_block, rhs_block, result_block, ofs=result_ofs)
    position = -1
    curr_block_ids.pop(-1)
    # Inlined maybe_save for position -1
    if op_positions["save"] == position:
        tile_index_ofs = (0, 0)
        save_result_block(result, result_block, tile_index_ofs=tile_index_ofs)
    if op_positions["matmul"] == position:
        result_ofs = get_block_ofs(mm, loop_order, ("M", "N"), position, curr_block_ids)
        matmul_blocks_lhsT(lhsT_block, rhs_block, result_block, ofs=result_ofs)
    return result


def load_input_block(
    input_tensor, tensor_dims: Tuple[str, str], mm: GEMMCompatibility, loop_order: Dict, position: int, curr_block_ids
):
    block_shape = get_block_shape(mm, loop_order, tensor_dims, position)
    block_ofs = get_block_ofs(mm, loop_order, tensor_dims, position, curr_block_ids)
    block = load_tensor_block(input_tensor=input_tensor, ofs=block_ofs, load_shape=block_shape)
    return block, block_ofs


def init_result_block(mm: GEMMCompatibility, loop_order: Dict, position: int, dtype):
    result_block_shape = get_block_shape(mm, loop_order, ("M", "N"), position)
    result_block = nl.zeros(result_block_shape, dtype=dtype, buffer=nl.sbuf)
    return result_block
