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
        5. Save is right after the K loop
    FIXME: compute the global coordinate, use the global coordinate to update tensors
    """
    kernel_kwargs = {
        "NUM_BLOCK_M": NUM_BLOCK_M,
        "NUM_BLOCK_N": NUM_BLOCK_N,
        "NUM_BLOCK_K": NUM_BLOCK_K,
        "loop_order": loop_order,
        "tensor_positions": tensor_positions,
    }
    preprocessing((lhsT, rhs), kernel_kwargs)
    mm = GEMMCompatibility(transposed_lhs=True)
    mm((lhsT, rhs), kernel_kwargs)
    result = nl.ndarray((mm.M, mm.N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    loop_order_lookup = {value: key for key, value in loop_order.items()}
    for block_id_0 in nl.affine_range(getattr(mm, f"NUM_BLOCK_{loop_order_lookup[0]}")):
        lhsT_block_shape = get_block_shape(mm, loop_order, ("K", "M"), 0)
        lhsT_ofs = get_block_ofs(mm, loop_order, ("K", "M"), 0, [block_id_0])
        lhsT_block = load_tensor_block(input_tensor=(lhsT, rhs)[0], ofs=lhsT_ofs, load_shape=lhsT_block_shape)
        for block_id_1 in nl.affine_range(getattr(mm, f"NUM_BLOCK_{loop_order_lookup[1]}")):
            result_block_shape = get_block_shape(mm, loop_order, ("M", "N"), 1)
            result_block = nl.zeros(result_block_shape, dtype=result.dtype, buffer=nl.sbuf)
            for block_id_2 in nl.affine_range(getattr(mm, f"NUM_BLOCK_{loop_order_lookup[2]}")):
                rhs_block_shape = get_block_shape(mm, loop_order, ("K", "N"), 2)
                rhs_ofs = get_block_ofs(mm, loop_order, ("K", "N"), 2, [block_id_0, block_id_1, block_id_2])
                rhs_block = load_tensor_block(input_tensor=(lhsT, rhs)[1], ofs=rhs_ofs, load_shape=rhs_block_shape)
                result_ofs = get_block_ofs(mm, loop_order, ("M", "N"), 2, [block_id_0, block_id_1, block_id_2])
                matmul_blocks_lhsT((lhsT, rhs)[0], (lhsT, rhs)[1], result_block, ofs=result_ofs)
            tile_index_ofs = [block_id_0, block_id_1][0] * mm.TILES_IN_BLOCK_M, 0
            save_result_block(result, result_block, tile_index_ofs=tile_index_ofs)
    return result
