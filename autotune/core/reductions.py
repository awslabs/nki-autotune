import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np


def compute_max_vals(input_block):
    """
    Computes the maximum values along the last dimension ('free_size') for each position in the input block.

    This function iterates through all blocks in a multi-dimensional tensor and computes the maximum
    value along the last dimension for each position. The resulting tensor preserves all dimensions
    except the last one, which is reduced to size 1.

    Args:
        input_block (nl.ndarray): Input tensor with shape (par_size, *block_sizes, free_size) where:
            - par_size: Size of the partition dimension
            - block_sizes: Variable number of middle dimensions representing the block structure
            - free_size: Size of the last dimension over which the maximum is computed

    Returns:
        nl.ndarray: Tensor containing maximum values with shape (par_size, *block_sizes, 1).
                    The output has the same data type as the input_block.

    Example:
        If input_block has shape (32, 4, 5, 128), the output will have shape (32, 4, 5, 1),
        where each value is the maximum along the last dimension of size 128.
    """
    block_max_vals = nl.ndarray(tuple(input_block.shape[:-1]) + (1,), dtype=input_block.dtype, buffer=nl.sbuf)
    par_size = input_block.shape[0]
    free_size = input_block.shape[-1]
    i_input = nl.mgrid[0:par_size, 0:free_size]
    i_max_vals = nl.mgrid[0:par_size, 0:1]
    num_blocks = 1
    for dim in input_block.shape[1:-1]:
        num_blocks *= dim
    for block_id in nl.affine_range(num_blocks):
        block_indices = []
        remaining = block_id
        for dim in reversed(input_block.shape[1:-1]):
            block_indices.insert(0, remaining % dim)
            remaining //= dim
        max_vals_coordinates = [i_max_vals.p] + block_indices + [i_max_vals.x]
        input_coordinates = [i_input.p] + block_indices + [i_input.x]
        block_max_vals[tuple(max_vals_coordinates)] = nisa.tensor_reduce(
            op=np.max, data=input_block[tuple(input_coordinates)], axis=(1,), dtype=input_block.dtype, negate=False
        )
    return block_max_vals


def compute_sums(input_block):
    """
    Computes the sums along the last dimension ('free_size') for each position in the input block.

    This function iterates through all blocks in a multi-dimensional tensor and computes the maximum
    value along the last dimension for each position. The resulting tensor preserves all dimensions
    except the last one, which is reduced to size 1.

    Args:
        input_block (nl.ndarray): Input tensor with shape (par_size, *block_sizes, free_size) where:
            - par_size: Size of the partition dimension
            - block_sizes: Variable number of middle dimensions representing the block structure
            - free_size: Size of the last dimension over which the maximum is computed

    Returns:
        nl.ndarray: Tensor containing maximum values with shape (par_size, *block_sizes, 1).
                    The output has the same data type as the input_block.

    Example:
        If input_block has shape (32, 4, 5, 128), the output will have shape (32, 4, 5, 1),
        where each value is the maximum along the last dimension of size 128.
    """
    block_sums = nl.ndarray(tuple(input_block.shape[:-1]) + (1,), dtype=input_block.dtype, buffer=nl.sbuf)
    par_size = input_block.shape[0]
    free_size = input_block.shape[-1]
    i_input = nl.mgrid[0:par_size, 0:free_size]
    i_max_vals = nl.mgrid[0:par_size, 0:1]
    num_blocks = 1
    for dim in input_block.shape[1:-1]:
        num_blocks *= dim
    for block_id in nl.affine_range(num_blocks):
        block_indices = []
        remaining = block_id
        for dim in reversed(input_block.shape[1:-1]):
            block_indices.insert(0, remaining % dim)
            remaining //= dim
        max_vals_coordinates = [i_max_vals.p] + block_indices + [i_max_vals.x]
        input_coordinates = [i_input.p] + block_indices + [i_input.x]
        sum_exp_block[i_sum_exp_block.p, tile_id_M, i_sum_exp_block.x] = nisa.tensor_reduce(
            op=np.add,
            data=exp_block[i_exp_block.p, tile_id_M, 0, i_exp_block.x],
            axis=(1,),
            dtype=exp_block.dtype,
            negate=False,
        )
    return block_max_vals
