import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl


def scale_tile(input_tile, scale_tile):
    """
    Scale rows of a 2D tile by corresponding values in a scale tile.

    This function performs in-place scaling of each row in the input tile by
    multiplying it with the corresponding scalar value from the scale tile.

    Parameters:
    -----------
    input_tile : ndarray
        A 2D array with shape (par_size, free_size) that will be modified in-place.

    scale_tile : ndarray
        A 2D array with shape (par_size, 1) containing scaling factors for each row
        of the input tile.

    Returns:
    --------
    None
        The function modifies input_tile in-place.
    """
    par_size, free_size = input_tile.shape
    assert scale_tile.shape == (
        par_size,
        1,
    ), f"Scale tile shape mismatch: expected ({par_size}, 1), got {scale_tile.shape}. The scale_tile must have free size = 1 and the same number of partition dimension as input_tile."
    i_grid = nl.mgrid[0:par_size, 0:free_size]
    scaled_tile = nl.ndarray((nl.par_dim(par_size), free_size), dtype=input_tile.dtype, buffer=nl.sbuf)
    scaled_tile[...] = nl.multiply(input_tile[i_grid.p, i_grid.x], scale_tile[...], dtype=input_tile.dtype)
    input_tile[i_grid.p, i_grid.x] = nl.copy(scaled_tile, dtype=input_tile.dtype)


def scale_block(input_block, scale_factor, op: str):
    """
    Scales down input_block by dividing/multiplying each element by the corresponding scale_factor.

    This function divides elements of the input_block by their corresponding values in the
    scale_factor tensor. The operation is performed in-place, modifying the input_block.
    The computation is done independently for each block in the multi-dimensional tensors.

    Args:
        input_block (nl.ndarray): Input tensor with shape (par_size, *block_sizes, bcast_size).
                                 This tensor is modified in-place.
        scale_factor (nl.ndarray): Scale factor tensor with shape (par_size, *block_sizes, free_size).

    Raises:
        AssertionError: If input_block and scale_factor have different shapes in all dimensions
                        except for the last one.

    Note:
        - This function modifies input_block in-place.
        - The last dimensions of input_block and scale_factor may differ, with the elements from
          scale_factor's last dimension being broadcast as needed.
    """
    assert (
        input_block.shape[:-1] == scale_factor.shape[:-1]
    ), f"input_block {input_block.shape} scale_factor {scale_factor.shape} shape mismatch."
    par_size = input_block.shape[0]
    bcast_size = input_block.shape[-1]
    free_size = scale_factor.shape[-1]
    i_input = nl.mgrid[0:par_size, 0:bcast_size]
    i_scale_factor = nl.mgrid[0:par_size, 0:free_size]

    block_sizes = input_block.shape[1:-1]
    num_blocks = 1
    for dim in block_sizes:
        num_blocks *= dim

    for block_id in nl.affine_range(num_blocks):
        block_indices = []
        remaining = block_id
        for dim in reversed(block_sizes):
            block_indices.insert(0, remaining % dim)
            remaining //= dim
        input_coordinates = [i_input.p] + block_indices + [i_input.x]
        scale_coordinates = [i_scale_factor.p] + block_indices + [i_scale_factor.x]

        if op == "divide":
            input_block[tuple(input_coordinates)] = nl.divide(
                input_block[tuple(input_coordinates)], scale_factor[tuple(scale_coordinates)]
            )
        elif op == "multiply":
            input_block[tuple(input_coordinates)] = nl.multiply(
                input_block[tuple(input_coordinates)], scale_factor[tuple(scale_coordinates)]
            )
        else:
            raise NotImplementedError(f"OP must be divide/multiply. Received {op}.")


def blocked_activation(out_block, op, data_block, scale_block, bias_block):
    """
    Applies an activation operation to blocks of data with bias and scaling.

    This function iterates through all blocks in the input tensors and applies
    the specified activation operation element-wise according to:
    out = op(data, bias, scale)

    output = f_activation(data * scale + bias)

    Args:
        op: The activation operation to apply.
        data_block (nl.ndarray): Input data tensor with shape (par_size, *block_sizes, free_size).
        bias_block (nl.ndarray): Bias values tensor with the same shape as data_block.
        scale_block (nl.ndarray): Scaling factors tensor with the same shape as data_block.
        out_block (nl.ndarray): Output tensor with the same shape as data_block where
                                results will be stored.

    Raises:
        AssertionError: If any of bias_block, scale_block, or out_block have
                        different shapes than data_block.

    Note:
        This function processes the input tensors block by block, where blocks are determined
        by the middle dimensions of the tensors. The activation is applied independently
        to each element according to the formula defined by the nisa.activation operation.
    """
    assert bias_block.shape == data_block.shape
    assert scale_block.shape == data_block.shape
    assert out_block.shape == data_block.shape
    par_size = data_block.shape[0]
    free_size = data_block.shape[-1]
    i_data = nl.mgrid[0:par_size, 0:free_size]

    block_sizes = data_block.shape[1:-1]
    num_blocks = 1
    for dim in block_sizes:
        num_blocks *= dim

    for block_id in nl.affine_range(num_blocks):
        block_indices = []
        remaining = block_id
        for dim in reversed(block_sizes):
            block_indices.insert(0, remaining % dim)
            remaining //= dim
        coordinates = [i_data.p] + block_indices + [i_data.x]
        out_block[tuple(coordinates)] = nisa.activation(
            op=op,
            data=data_block[tuple(coordinates)],
            bias=bias_block[tuple(coordinates)],
            scale=scale_block[tuple(coordinates)],
        )
