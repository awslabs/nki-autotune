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
    input_tensors = (
        array(
            [
                [1.07198727, -0.31218023, -0.09930849, ..., -0.51547836, -0.47679939, 1.92945271],
                [0.31075926, 0.68949698, -0.88782456, ..., 0.98312237, -2.36411664, 0.59528954],
                [0.43911272, 0.02446459, 0.17861453, ..., 1.7861571, 0.04354124, 1.63839492],
                ...,
                [1.28768357, -0.39772362, -1.07578675, ..., -0.60086019, 1.16430279, 0.24716993],
                [-0.3015216, 0.85011728, -0.95484792, ..., -0.73601038, 1.00303695, -1.91069109],
                [0.0479285, 0.89038464, -0.38486427, ..., -1.1231122, 1.18761205, 0.83054011],
            ]
        ),
        array(
            [
                [1.56909402, -0.85676822, 1.24248637, ..., -1.06028833, 0.78592501, -0.22142861],
                [0.35509051, 0.6532109, 1.11587553, ..., 1.21222685, -0.50704064, -0.58092818],
                [-1.154777, -0.29297318, -0.8193523, ..., 1.45737435, -0.3687348, 0.41072923],
                ...,
                [-1.43092223, 0.51981606, 1.06490223, ..., -0.95172617, -0.74853812, -0.19990597],
                [-0.63193876, -0.44835362, -0.37104225, ..., -0.36362285, 0.06437683, -0.12645351],
                [0.57725309, 0.70752592, 0.7722807, ..., 0.20923559, 1.08430467, 0.50356845],
            ]
        ),
    )
    kernel_kwargs = {
        "NUM_BLOCK_M": 2,
        "NUM_BLOCK_N": 1,
        "NUM_BLOCK_K": 4,
        "loop_order": "MKN",
        "tensor_positions": {"result_block": -1, "rhs_block": 1, "lhsT_block": 2},
    }
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
    assert (
        lhsT_block_position <= K_position
        and rhs_block_position <= K_position
        or (lhsT_block_position > K_position and rhs_block_position > K_position)
    ), f"lhsT_block and rhs_block must be on the same side of K loop. Received lhsT_block_position {lhsT_block_position}, rhs_block_position {rhs_block_position}, K_position {K_position}."
    mm = GEMMCompatibility(transposed_lhs=True)
    mm(input_tensors, kernel_kwargs)
    result = nl.ndarray(
        (mm.M, mm.N),
        dtype=array(
            [
                [1.07198727, -0.31218023, -0.09930849, ..., -0.51547836, -0.47679939, 1.92945271],
                [0.31075926, 0.68949698, -0.88782456, ..., 0.98312237, -2.36411664, 0.59528954],
                [0.43911272, 0.02446459, 0.17861453, ..., 1.7861571, 0.04354124, 1.63839492],
                ...,
                [1.28768357, -0.39772362, -1.07578675, ..., -0.60086019, 1.16430279, 0.24716993],
                [-0.3015216, 0.85011728, -0.95484792, ..., -0.73601038, 1.00303695, -1.91069109],
                [0.0479285, 0.89038464, -0.38486427, ..., -1.1231122, 1.18761205, 0.83054011],
            ]
        ).dtype,
        buffer=nl.shared_hbm,
    )
    for loop_id in [0, 1, 2]:
        loop_size = getattr(mm, f"NUM_BLOCK_{'MKN'[loop_id]}")
        print(f"Loop {loop_id}: {'MKN'[loop_id]} size {loop_size}")
    tensor_blocks = {"lhsT_block": None, "rhs_block": None, "result_block": None}
    (lhsT, rhs) = input_tensors
    if 2 == 0:
        lhsT_block_shape = get_block_shape(mm, "MKN", ("K", "M"), 0)
        lhsT_ofs = get_block_ofs(mm, "MKN", ("K", "M"), 0, curr_block_ids)
        tensor_blocks["lhsT_block"] = load_tensor_block(input_tensor=lhsT, ofs=lhsT_ofs, load_shape=lhsT_block_shape)
    if 1 == 0:
        rhs_block_shape = get_block_shape(mm, "MKN", ("K", "N"), 0)
        rhs_ofs = get_block_ofs(mm, "MKN", ("K", "N"), 0, curr_block_ids)
        tensor_blocks["rhs_block"] = load_tensor_block(input_tensor=rhs, ofs=rhs_ofs, load_shape=rhs_block_shape)
    if -1 == 0:
        result_block_shape = get_block_shape(mm, "MKN", ("M", "N"), 0)
        tensor_blocks["result_block"] = nl.zeros(result_block_shape, dtype=result.dtype, buffer=nl.sbuf)
    for block_id_0 in nl.affine_range(getattr(mm, f"NUM_BLOCK_{'M'}")):
        (lhsT, rhs) = input_tensors
        if 2 == 1:
            lhsT_block_shape = get_block_shape(mm, "MKN", ("K", "M"), 1)
            lhsT_ofs = get_block_ofs(mm, "MKN", ("K", "M"), 1, curr_block_ids)
            tensor_blocks["lhsT_block"] = load_tensor_block(
                input_tensor=lhsT, ofs=lhsT_ofs, load_shape=lhsT_block_shape
            )
        if 1 == 1:
            rhs_block_shape = get_block_shape(mm, "MKN", ("K", "N"), 1)
            rhs_ofs = get_block_ofs(mm, "MKN", ("K", "N"), 1, curr_block_ids)
            tensor_blocks["rhs_block"] = load_tensor_block(input_tensor=rhs, ofs=rhs_ofs, load_shape=rhs_block_shape)
        if -1 == 1:
            result_block_shape = get_block_shape(mm, "MKN", ("M", "N"), 1)
            tensor_blocks["result_block"] = nl.zeros(result_block_shape, dtype=result.dtype, buffer=nl.sbuf)
        for block_id_1 in nl.affine_range(getattr(mm, f"NUM_BLOCK_{'K'}")):
            (lhsT, rhs) = input_tensors
            if 2 == 2:
                lhsT_block_shape = get_block_shape(mm, "MKN", ("K", "M"), 2)
                lhsT_ofs = get_block_ofs(mm, "MKN", ("K", "M"), 2, curr_block_ids)
                tensor_blocks["lhsT_block"] = load_tensor_block(
                    input_tensor=lhsT, ofs=lhsT_ofs, load_shape=lhsT_block_shape
                )
            if 1 == 2:
                rhs_block_shape = get_block_shape(mm, "MKN", ("K", "N"), 2)
                rhs_ofs = get_block_ofs(mm, "MKN", ("K", "N"), 2, curr_block_ids)
                tensor_blocks["rhs_block"] = load_tensor_block(
                    input_tensor=rhs, ofs=rhs_ofs, load_shape=rhs_block_shape
                )
            if -1 == 2:
                result_block_shape = get_block_shape(mm, "MKN", ("M", "N"), 2)
                tensor_blocks["result_block"] = nl.zeros(result_block_shape, dtype=result.dtype, buffer=nl.sbuf)
            for block_id_2 in nl.affine_range(getattr(mm, f"NUM_BLOCK_{'N'}")):
                (lhsT, rhs) = input_tensors
                if 2 == 3:
                    lhsT_block_shape = get_block_shape(mm, "MKN", ("K", "M"), 3)
                    lhsT_ofs = get_block_ofs(mm, "MKN", ("K", "M"), 3, curr_block_ids)
                    tensor_blocks["lhsT_block"] = load_tensor_block(
                        input_tensor=lhsT, ofs=lhsT_ofs, load_shape=lhsT_block_shape
                    )
                if 1 == 3:
                    rhs_block_shape = get_block_shape(mm, "MKN", ("K", "N"), 3)
                    rhs_ofs = get_block_ofs(mm, "MKN", ("K", "N"), 3, curr_block_ids)
                    tensor_blocks["rhs_block"] = load_tensor_block(
                        input_tensor=rhs, ofs=rhs_ofs, load_shape=rhs_block_shape
                    )
                if -1 == 3:
                    result_block_shape = get_block_shape(mm, "MKN", ("M", "N"), 3)
                    tensor_blocks["result_block"] = nl.zeros(result_block_shape, dtype=result.dtype, buffer=nl.sbuf)
            M_position = "MKN".index("M")
            N_position = "MKN".index("N")
            K_position = "MKN".index("K")
            if -1 == 2:
                print(f"Saving {result_block.shape} into {result.shape}.")
                if 2 == 0:
                    save_result_block(result, result_block, tile_index_ofs=(0, 0))
                elif 2 == 1:
                    if M_position == 0:
                        tile_index_ofs = (curr_block_ids[0] * mm.TILES_IN_BLOCK_M, 0)
                    if N_position == 0:
                        tile_index_ofs = (0, curr_block_ids[0] * mm.TILES_IN_BLOCK_N)
                    save_result_block(result, result_block, tile_index_ofs=tile_index_ofs)
                elif 2 == 2:
                    if M_position == 0:
                        tile_index_ofs = (
                            curr_block_ids[0] * mm.TILES_IN_BLOCK_M,
                            curr_block_ids[1] * mm.TILES_IN_BLOCK_N,
                        )
                    if N_position == 0:
                        tile_index_ofs = (
                            curr_block_ids[1] * mm.TILES_IN_BLOCK_M,
                            curr_block_ids[0] * mm.TILES_IN_BLOCK_N,
                        )
                    save_result_block(result, result_block, tile_index_ofs=tile_index_ofs)
                else:
                    raise ValueError(f"Invalid curr_position {2} for result_block save. curr_position is in (0,1,2).")
        M_position = "MKN".index("M")
        N_position = "MKN".index("N")
        K_position = "MKN".index("K")
        if -1 == 1:
            print(f"Saving {result_block.shape} into {result.shape}.")
            if 1 == 0:
                save_result_block(result, result_block, tile_index_ofs=(0, 0))
            elif 1 == 1:
                if M_position == 0:
                    tile_index_ofs = (curr_block_ids[0] * mm.TILES_IN_BLOCK_M, 0)
                if N_position == 0:
                    tile_index_ofs = (0, curr_block_ids[0] * mm.TILES_IN_BLOCK_N)
                save_result_block(result, result_block, tile_index_ofs=tile_index_ofs)
            elif 1 == 2:
                if M_position == 0:
                    tile_index_ofs = (curr_block_ids[0] * mm.TILES_IN_BLOCK_M, curr_block_ids[1] * mm.TILES_IN_BLOCK_N)
                if N_position == 0:
                    tile_index_ofs = (curr_block_ids[1] * mm.TILES_IN_BLOCK_M, curr_block_ids[0] * mm.TILES_IN_BLOCK_N)
                save_result_block(result, result_block, tile_index_ofs=tile_index_ofs)
            else:
                raise ValueError(f"Invalid curr_position {1} for result_block save. curr_position is in (0,1,2).")
    M_position = "MKN".index("M")
    N_position = "MKN".index("N")
    K_position = "MKN".index("K")
    if -1 == 0:
        print(f"Saving {result_block.shape} into {result.shape}.")
        if 0 == 0:
            save_result_block(result, result_block, tile_index_ofs=(0, 0))
        elif 0 == 1:
            if M_position == 0:
                tile_index_ofs = (curr_block_ids[0] * mm.TILES_IN_BLOCK_M, 0)
            if N_position == 0:
                tile_index_ofs = (0, curr_block_ids[0] * mm.TILES_IN_BLOCK_N)
            save_result_block(result, result_block, tile_index_ofs=tile_index_ofs)
        elif 0 == 2:
            if M_position == 0:
                tile_index_ofs = (curr_block_ids[0] * mm.TILES_IN_BLOCK_M, curr_block_ids[1] * mm.TILES_IN_BLOCK_N)
            if N_position == 0:
                tile_index_ofs = (curr_block_ids[1] * mm.TILES_IN_BLOCK_M, curr_block_ids[0] * mm.TILES_IN_BLOCK_N)
            save_result_block(result, result_block, tile_index_ofs=tile_index_ofs)
        else:
            raise ValueError(f"Invalid curr_position {0} for result_block save. curr_position is in (0,1,2).")
    return result
