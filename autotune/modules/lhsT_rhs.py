# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Dict, Tuple

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.compiler.backends.neuron.tensors import KernelHBMTensor
from neuronxcc.nki.typing import tensor

from autotune.modules.dma import load_tensor_block, save_result_acc, save_result_block, save_result_dma
from autotune.modules.matmul import GEMMCompatibility, matmul_blocks_lhsT
from autotune.typing import INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE


def preprocessing(input_tensors: INPUT_TENSORS_DTYPE, kernel_kwargs: KERNEL_KWARGS_DTYPE):
    mm = GEMMCompatibility(transposed_lhs=True)
    mm(input_tensors=input_tensors, kernel_kwargs=kernel_kwargs)
    loop_order = kernel_kwargs["loop_order"]
    tensor_positions = kernel_kwargs["tensor_positions"]
    if len(loop_order) != 3 or sorted(loop_order) != sorted("MNK"):
        raise ValueError(f"Invalid loop_order: {loop_order}. Must contain exactly M, N, and K.")
    M_position = loop_order.index("M")
    N_position = loop_order.index("N")
    K_position = loop_order.index("K")
    lhsT_block_position = tensor_positions["lhsT_block_position"]
    rhs_block_position = tensor_positions["rhs_block_position"]
    result_block_position = tensor_positions["result_block_position"]
    matmul_position = max(lhsT_block_position, rhs_block_position)
    assert (
        result_block_position <= K_position and result_block_position <= matmul_position
    ), f"result_block init must be before K loop and matmul. Received result_block_position {result_block_position}, K_position {K_position}, matmul_position {matmul_position}."
    assert (
        matmul_position <= lhsT_block_position and matmul_position <= rhs_block_position
    ), f"matmul must be after lhsT_block, rhs_block loads. Received matmul_position {matmul_position}, lhsT_block_position {lhsT_block_position}, rhs_block_position {rhs_block_position}."
    assert (lhsT_block_position <= K_position and rhs_block_position <= K_position) or (
        lhsT_block_position > K_position and rhs_block_position > K_position
    ), f"lhsT_block and rhs_block must be on the same side of K loop. Received lhsT_block_position {lhsT_block_position}, rhs_block_position {rhs_block_position}, K_position {K_position}."


def lhsT_rhs_gemm_general(
    lhsT: tensor,
    rhs: tensor,
    NUM_BLOCK_M: int,
    NUM_BLOCK_N: int,
    NUM_BLOCK_K: int,
    loop_order: str,
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
        loop_order: str - Three-character string specifying the loop ordering (e.g., "MNK", "NMK")
        tensor_positions: Dict[str, int] - Positions where tensor operations occur:
            (lhsT_block_position, rhs_block_position, result_block_position)

    Returns:
        tensor of shape (M, N) - Result of the matrix multiplication

    Position reference:
        position = -1  # Before all loops
        loop_0        # Outermost loop
            position = 0
            loop_1    # Middle loop
                position = 1
                loop_2  # Innermost loop
                    position = 2

    Constraints:
        1. result_block initialization must occur before K loop and matmul operation
        2. matmul operation must occur after lhsT_block and rhs_block loads
        3. lhsT_block and rhs_block loads must be on the same side of K loop
        4. Loop order must contain exactly the characters 'M', 'N', and 'K'
    FIXME: compute the global coordinate, use the global coordinate to update tensors
    save and matmul locations should be mirror?
    """
    input_tensors = (lhsT, rhs)
    kernel_kwargs = {
        "NUM_BLOCK_M": NUM_BLOCK_M,
        "NUM_BLOCK_N": NUM_BLOCK_N,
        "NUM_BLOCK_K": NUM_BLOCK_K,
        "loop_order": loop_order,
        "tensor_positions": tensor_positions,
    }
    preprocessing(input_tensors, kernel_kwargs)
    mm = GEMMCompatibility(transposed_lhs=True)
    mm(input_tensors, kernel_kwargs)
    result = nl.ndarray((mm.M, mm.N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    for loop_id in [0, 1, 2]:
        loop_size = getattr(mm, f"NUM_BLOCK_{loop_order[loop_id]}")
        print(f"Loop {loop_id}: {loop_order[loop_id]} size {loop_size}")

    tensor_blocks = {"lhsT_block": None, "rhs_block": None, "result_block": None}
    maybe_init(tensor_positions, 0, loop_order, [], mm, (lhsT, rhs), result, tensor_blocks)
    for block_id_0 in nl.affine_range(getattr(mm, f"NUM_BLOCK_{loop_order[0]}")):
        maybe_init(tensor_positions, 1, loop_order, [block_id_0], mm, (lhsT, rhs), result, tensor_blocks)
        for block_id_1 in nl.affine_range(getattr(mm, f"NUM_BLOCK_{loop_order[1]}")):
            maybe_init(
                tensor_positions, 2, loop_order, [block_id_0, block_id_1], mm, (lhsT, rhs), result, tensor_blocks
            )
            for block_id_2 in nl.affine_range(getattr(mm, f"NUM_BLOCK_{loop_order[2]}")):
                maybe_init(
                    tensor_positions,
                    3,
                    loop_order,
                    [block_id_0, block_id_1, block_id_2],
                    mm,
                    (lhsT, rhs),
                    result,
                    tensor_blocks,
                )
            maybe_save(
                tensor_positions, 2, loop_order, [block_id_0, block_id_1], mm, tensor_blocks["result_block"], result
            )
        maybe_save(tensor_positions, 1, loop_order, [block_id_0], mm, tensor_blocks["result_block"], result)
    maybe_save(tensor_positions, 0, loop_order, [block_id_0], mm, tensor_blocks["result_block"], result)
    return result


def maybe_init(
    tensor_positions: Dict[str, int],
    curr_position: int,
    loop_order: str,
    curr_block_ids,
    mm: GEMMCompatibility,
    input_tensors,
    result,
    tensor_blocks,
):
    lhsT, rhs = input_tensors
    if tensor_positions["lhsT_block"] == curr_position:
        lhsT_block_shape = get_block_shape(mm, loop_order, ("K", "M"), curr_position)
        lhsT_ofs = get_block_ofs(mm, loop_order, ("K", "M"), curr_position, curr_block_ids)
        tensor_blocks["lhsT_block"] = load_tensor_block(input_tensor=lhsT, ofs=lhsT_ofs, load_shape=lhsT_block_shape)
    if tensor_positions["rhs_block"] == curr_position:
        rhs_block_shape = get_block_shape(mm, loop_order, ("K", "N"), curr_position)
        rhs_ofs = get_block_ofs(mm, loop_order, ("K", "N"), curr_position, curr_block_ids)
        tensor_blocks["rhs_block"] = load_tensor_block(input_tensor=rhs, ofs=rhs_ofs, load_shape=rhs_block_shape)
    if tensor_positions["result_block"] == curr_position:
        result_block_shape = get_block_shape(mm, loop_order, ("M", "N"), curr_position)
        tensor_blocks["result_block"] = nl.zeros(result_block_shape, dtype=result.dtype, buffer=nl.sbuf)


def maybe_compute(
    tensor_positions: Tuple[int, int, int, int],
    curr_position: int,
    loop_order: str,
    curr_block_ids,
    mm: GEMMCompatibility,
    input_tensors,
    result,
):
    lhsT_block_position, rhs_block_position, result_block_position, matmul_position = tensor_positions
    lhsT, rhs = input_tensors
    if matmul_position == curr_position:
        result_ofs = get_block_ofs(mm, loop_order, ("M", "N"), curr_position, curr_block_ids)
        matmul_blocks_lhsT(lhsT_block, rhs_block, result_block, ofs=result_ofs)


def maybe_save(
    tensor_positions: Dict[str, int],
    curr_position: int,
    loop_order: str,
    curr_block_ids,
    mm: GEMMCompatibility,
    result_block,
    result,
):
    M_position = loop_order.index("M")
    N_position = loop_order.index("N")
    K_position = loop_order.index("K")
    if tensor_positions["result_block"] == curr_position:
        print(f"Saving {result_block.shape} into {result.shape}.")
        if curr_position == 0:
            save_result_block(result, result_block, tile_index_ofs=(0, 0))
        elif curr_position == 1:
            if M_position == 0:
                tile_index_ofs = (curr_block_ids[0] * mm.TILES_IN_BLOCK_M, 0)
            if N_position == 0:
                tile_index_ofs = (0, curr_block_ids[0] * mm.TILES_IN_BLOCK_N)
            save_result_block(result, result_block, tile_index_ofs=tile_index_ofs)
        elif curr_position == 2:
            if M_position == 0:
                tile_index_ofs = (curr_block_ids[0] * mm.TILES_IN_BLOCK_M, curr_block_ids[1] * mm.TILES_IN_BLOCK_N)
            if N_position == 0:
                tile_index_ofs = (curr_block_ids[1] * mm.TILES_IN_BLOCK_M, curr_block_ids[0] * mm.TILES_IN_BLOCK_N)
            save_result_block(result, result_block, tile_index_ofs=tile_index_ofs)
        else:
            raise ValueError(
                f"Invalid curr_position {curr_position} for result_block save. curr_position is in (0,1,2)."
            )


def get_block_shape(
    mm: GEMMCompatibility, loop_order: str, dims: Tuple[str, str], curr_position: int
) -> Tuple[int, int, int, int]:
    """
    Calculate the shape of tensor blocks in a GEMM operation.

    This function computes the shape of blocks based on the current position in nested loops
    and the specified dimensions. It determines how many blocks should be processed together
    for each dimension based on loop nesting.

    Args:
        mm (GEMMCompatibility): Object containing GEMM configuration parameters and block dimensions
        loop_order (str): String representing the order of loops (e.g., "MNK")
        dims (Tuple[str, str]): Tuple of dimension names to calculate shape for (e.g., ("M", "N"))
        curr_position (int): Current position in the nested loops (0, 1, 2, or 3)

    Returns:
        Tuple[int, int, int, int]: A 4-tuple representing the block shape:
            - First element: Tile size for the first dimension
            - Second element: Number of blocks * tiles in block for the first dimension
            - Third element: Number of blocks * tiles in block for the second dimension
            - Fourth element: Tile size for the second dimension

    Note:
        For dimensions with loop position less than curr_position, num_block is set to 1.
        For other dimensions, num_block is set to the corresponding NUM_BLOCK_{dim} value.
    """
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
    print(
        f"get_block_shape: dependent dims {dims}. curr loop position {curr_position}. loop_order {loop_order}.\n--> block_shape{block_shape}."
    )
    return block_shape


def get_block_ofs(
    mm: GEMMCompatibility, loop_order: str, dims: Tuple[str, str], curr_position: int, curr_block_ids
) -> Tuple[int, int]:
    """
    Calculate the offset positions for blocks in a GEMM operation.

    This function computes the starting offsets for blocks along specified dimensions based on
    the current position in nested loops and the corresponding block indices.

    Args:
        mm (GEMMCompatibility): Object containing GEMM configuration parameters and block dimensions
        loop_order (str): String representing the order of loops (e.g., "MNK")
        dims (Tuple[str, str]): Tuple of dimension names to calculate offsets for (e.g., ("K", "M"))
        curr_position (int): Current position in the nested loops (0, 1, 2, or 3)
        curr_block_ids (List[int]): List of current block indices from the outer loops

    Returns:
        Tuple[int, int]: Tuple containing the offsets for the specified dimensions

    Note:
        For dimensions with loop position less than curr_position, offset is calculated as
        block_id * block_size. For other dimensions, offset is 0.
    """
    block_ofs = []
    block_ofs_str = []
    for dim in dims:
        dim_position = loop_order.index(dim)
        if dim_position < curr_position:
            block_size = getattr(mm, f"BLOCK_{dim}")
            ofs = curr_block_ids[dim_position] * getattr(mm, f"BLOCK_{dim}")
            block_ofs_str.append(f"block_id * {block_size}")
        else:
            ofs = 0
            block_ofs_str.append("0")
        block_ofs.append(ofs)
    block_ofs = tuple(block_ofs)
    print(
        f"get_block_ofs: dependent dims {dims}. curr loop position {curr_position}. loop_order {loop_order}.\n-->block_ofs{block_ofs_str}."
    )
    # FIXME: if curr_position > dim_position, there should be an offset. But need to check against the target block init location too.
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
