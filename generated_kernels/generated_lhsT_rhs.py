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
                [-0.58552254, 0.82513408, 0.90590002, ..., 0.84077132, 0.95775049, -0.19211647],
                [1.50290511, -0.31209765, -1.87150032, ..., -0.38305428, -0.21117559, 0.79743439],
                [-0.74053773, -0.29703832, 0.00303174, ..., -0.21105282, 1.44679936, -0.07773064],
                ...,
                [-0.47037808, 0.1784227, -0.08350567, ..., -0.76830431, -0.48008096, -1.41857323],
                [0.15638925, -1.13477069, 0.56901469, ..., 1.50995044, -2.10118244, -0.65829124],
                [1.08972249, 1.6802071, 0.72236505, ..., -1.58119528, 0.15843823, 1.67798022],
            ]
        ),
        array(
            [
                [2.18550842, -1.91582956, 1.37567496, ..., 1.95206612, 0.83215697, -0.27622464],
                [0.56974602, 1.48373379, 0.79509591, ..., -2.4084338, 0.86330918, -0.22988297],
                [0.2090868, 0.92394872, 1.82043672, ..., -0.78625257, 0.03705751, -0.22594891],
                ...,
                [-0.04194087, 1.1056323, 1.67096284, ..., 0.59181688, -0.67401548, 2.27050145],
                [0.41807036, 0.09912868, 0.48347282, ..., 1.78798545, 0.41753529, 1.59763302],
                [0.35960773, -2.6124684, 1.28682621, ..., 0.91166439, -0.35160135, -2.31829221],
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
                [-0.58552254, 0.82513408, 0.90590002, ..., 0.84077132, 0.95775049, -0.19211647],
                [1.50290511, -0.31209765, -1.87150032, ..., -0.38305428, -0.21117559, 0.79743439],
                [-0.74053773, -0.29703832, 0.00303174, ..., -0.21105282, 1.44679936, -0.07773064],
                ...,
                [-0.47037808, 0.1784227, -0.08350567, ..., -0.76830431, -0.48008096, -1.41857323],
                [0.15638925, -1.13477069, 0.56901469, ..., 1.50995044, -2.10118244, -0.65829124],
                [1.08972249, 1.6802071, 0.72236505, ..., -1.58119528, 0.15843823, 1.67798022],
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
