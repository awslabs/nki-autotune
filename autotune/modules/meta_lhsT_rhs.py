# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Dict

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor

from autotune.modules.dma import load_tensor_block, save_result_block
from autotune.modules.layout import get_block_ofs, get_block_shape
from autotune.modules.matmul import GEMMCompatibility, matmul_blocks_lhsT


def check_template(loop_order: Dict[str, int], tensor_positions: Dict[str, int]):
    """
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
    if sorted(loop_order.keys()) != sorted("MNK"):
        raise ValueError(f"Invalid loop_order: {loop_order}. Must contain exactly M, N, and K.")
    K_position = loop_order["K"]
    lhsT_block_position = tensor_positions["lhsT_block"]
    rhs_block_position = tensor_positions["rhs_block"]
    result_block_position = tensor_positions["result_block"]
    matmul_position = tensor_positions["matmul"]
    assert (
        result_block_position < K_position and result_block_position < matmul_position
    ), f"result_block init must be before K loop and matmul. Received result_block_position {result_block_position}, K_position {K_position}, matmul_position {matmul_position}."
    assert matmul_position == max(
        lhsT_block_position, rhs_block_position
    ), f"matmul must be right after lhsT_block, rhs_block loads. Received matmul_position {matmul_position}, lhsT_block_position {lhsT_block_position}, rhs_block_position {rhs_block_position}."
    assert (lhsT_block_position <= K_position and rhs_block_position <= K_position) or (
        lhsT_block_position >= K_position and rhs_block_position >= K_position
    ), f"lhsT_block and rhs_block must be on the same side of K loop. Received lhsT_block_position {lhsT_block_position}, rhs_block_position {rhs_block_position}, K_position {K_position}."


@nki.jit
def lhsT_rhs_gemm_general(
    lhsT: tensor,
    rhs: tensor,
    NUM_BLOCK_M: int,
    NUM_BLOCK_N: int,
    NUM_BLOCK_K: int,
    loop_order: Dict[str, int],
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
        loop_order: Dict[str, int] - Dict specifying the loop ordering (e.g., "MNK")
        tensor_positions: Dict[str, int] - Positions where tensor operations occur:
            (lhsT_block, rhs_block, result_block)

    Returns:
        tensor of shape (M, N) - Result of the matrix multiplication

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

    Constraints:
        1. result_block initialization must occur before K loop and matmul operation
        2. matmul operation must occur after lhsT_block and rhs_block loads
        3. lhsT_block and rhs_block loads must be on the same side of K loop
        4. Loop order must contain exactly the characters 'M', 'N', and 'K'
        5. Save is right after the K loop
    FIXME: compute the global coordinate, use the global coordinate to update tensors

    standard Python:
    if xxx:
        tensor = init()
    f(tensor)

    NKI:
    if xxx:
        lhs = init()
    else:
        lhs = init()
    matmul(lhs, rhs) --> X
    """
    kernel_kwargs = {"NUM_BLOCK_M": NUM_BLOCK_M, "NUM_BLOCK_N": NUM_BLOCK_N, "NUM_BLOCK_K": NUM_BLOCK_K}
    mm = GEMMCompatibility(transposed_lhs=True)
    mm((lhsT, rhs), kernel_kwargs)
    result = nl.ndarray((mm.M, mm.N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    loop_order_lookup = {value: key for key, value in loop_order.items()}

    maybe_init(tensor_positions, -1, loop_order, [], mm, (lhsT, rhs), result)
    for block_id_0 in nl.affine_range(getattr(mm, f"NUM_BLOCK_{loop_order_lookup[0]}")):
        maybe_init(tensor_positions, 0, loop_order, [block_id_0], mm, (lhsT, rhs), result)
        for block_id_1 in nl.affine_range(getattr(mm, f"NUM_BLOCK_{loop_order_lookup[1]}")):
            maybe_init(tensor_positions, 1, loop_order, [block_id_0, block_id_1], mm, (lhsT, rhs), result)
            for block_id_2 in nl.affine_range(getattr(mm, f"NUM_BLOCK_{loop_order_lookup[2]}")):
                maybe_init(
                    tensor_positions, 2, loop_order, [block_id_0, block_id_1, block_id_2], mm, (lhsT, rhs), result
                )
                maybe_compute(tensor_positions, 2, loop_order, [block_id_0, block_id_1, block_id_2], mm)
            maybe_save(tensor_positions, 1, loop_order, [block_id_0, block_id_1], mm, result)
            maybe_compute(tensor_positions, 1, loop_order, [block_id_0, block_id_1], mm)
        maybe_save(tensor_positions, 0, loop_order, [block_id_0], mm, result)
        maybe_compute(tensor_positions, 0, loop_order, [block_id_0], mm)
    maybe_save(tensor_positions, -1, loop_order, [], mm, result)
    maybe_compute(tensor_positions, -1, loop_order, [], mm)
    return result


def maybe_init(
    tensor_positions_arg: Dict[str, int],
    curr_position: int,
    loop_order: Dict[str, int],
    curr_block_ids,
    mm: GEMMCompatibility,
    input_tensors,
    result,
):
    if tensor_positions_arg["lhsT_block"] == curr_position:
        lhsT_block_shape = get_block_shape(mm, loop_order, ("K", "M"), curr_position)
        lhsT_ofs = get_block_ofs(mm, loop_order, ("K", "M"), curr_position, curr_block_ids)
        print(f"generated lhsT_ofs = {lhsT_ofs}")
        lhsT_block = load_tensor_block(input_tensor=input_tensors[0], ofs=lhsT_ofs, load_shape=lhsT_block_shape)
        print(f"generated lhsT_block = {lhsT_block.shape}")
    if tensor_positions_arg["rhs_block"] == curr_position:
        rhs_block_shape = get_block_shape(mm, loop_order, ("K", "N"), curr_position)
        rhs_ofs = get_block_ofs(mm, loop_order, ("K", "N"), curr_position, curr_block_ids)
        print(f"generated rhs_ofs = {rhs_ofs}")
        rhs_block = load_tensor_block(input_tensor=input_tensors[1], ofs=rhs_ofs, load_shape=rhs_block_shape)
        print(f"generated rhs_block = {rhs_block.shape}")
    if tensor_positions_arg["result_block"] == curr_position:
        result_block_shape = get_block_shape(mm, loop_order, ("M", "N"), curr_position)
        result_block = nl.zeros(result_block_shape, dtype=result.dtype, buffer=nl.sbuf)


def maybe_compute(
    tensor_positions: Dict[str, int],
    curr_position: int,
    loop_order: Dict[str, int],
    curr_block_ids,
    mm: GEMMCompatibility,
):
    if tensor_positions["matmul"] == curr_position:
        result_ofs = get_block_ofs(mm, loop_order, ("M", "N"), curr_position, curr_block_ids)
        matmul_blocks_lhsT(lhsT_block, rhs_block, result_block, ofs=result_ofs)


def maybe_save(
    tensor_positions: Dict[str, int],
    curr_position: int,
    loop_order: Dict[str, int],
    curr_block_ids,
    mm: GEMMCompatibility,
    result,
):
    if tensor_positions["result_block"] == curr_position:
        if curr_position == 0:
            tile_index_ofs = (0, 0)
        elif curr_position == 1:
            if loop_order["M"] == 0:
                tile_index_ofs = (curr_block_ids[0] * mm.TILES_IN_BLOCK_M, 0)
            if loop_order["N"] == 0:
                tile_index_ofs = (0, curr_block_ids[0] * mm.TILES_IN_BLOCK_N)
        elif curr_position == 2:
            if loop_order["M"] == 0:
                tile_index_ofs = (curr_block_ids[0] * mm.TILES_IN_BLOCK_M, curr_block_ids[1] * mm.TILES_IN_BLOCK_N)
            if loop_order["N"] == 0:
                tile_index_ofs = (curr_block_ids[1] * mm.TILES_IN_BLOCK_M, curr_block_ids[0] * mm.TILES_IN_BLOCK_N)
        else:
            raise ValueError(
                f"Invalid curr_position {curr_position} for result_block save. curr_position is in (0,1,2)."
            )
        save_result_block(result, result_block, tile_index_ofs=tile_index_ofs)
