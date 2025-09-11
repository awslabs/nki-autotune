import neuronxcc.nki.language as nl
import numpy as np

from autotune.core.metrics import check_correctness
from autotune.core.tensor import SBUFTensor
from autotune.typing import INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE, OUTPUT_TENSORS_DTYPE


class GEMMCorrectness:
    def __init__(self, transposed_lhs: bool) -> None:
        self.transposed_lhs = transposed_lhs

    def __call__(
        self,
        input_tensors: INPUT_TENSORS_DTYPE,
        kernel_kwargs: KERNEL_KWARGS_DTYPE,
        nki_out_tensors: OUTPUT_TENSORS_DTYPE,
    ):
        data_type = np.float32
        atol, rtol = 1e-5, 1e-2
        lhs, rhs = input_tensors
        if self.transposed_lhs:
            golden = nl.static_cast(lhsT_rhs_gemm_np(lhs, rhs), data_type)
        else:
            golden = nl.static_cast(lhs_rhs_gemm_np(lhs, rhs), data_type)
        nki_out_tensor = nl.static_cast(nki_out_tensors[0], data_type)

        # Use the centralized check_correctness function from metrics module
        check_correctness(golden, nki_out_tensor, atol, rtol)


def max_nki_index(index1, index2):
    if index1 == 0:
        max_index = index2
    elif index2 == 0:
        max_index = index1
    else:
        max_index = max(index1, index2)
    return max_index


def calculate_tile_overlap(coords1: dict, coords2: dict) -> tuple[int, int]:
    """
    Calculate the overlapping tile region between two tile coordinate ranges.

    Args:
        coords1: First tile coordinates dictionary with 'start_tile_index' and 'num_tiles'
        coords2: Second tile coordinates dictionary with 'start_tile_index' and 'num_tiles'

    Returns:
        Tuple of (overlap_start, num_overlap_tiles)
    """
    start_1 = coords1["start_tile_index"]
    start_2 = coords2["start_tile_index"]
    overlap_start = max_nki_index(start_1, start_2)
    # print(f"start_1 {start_1} start_2 {start_2} --> overlap_start = {overlap_start}.")

    num_tiles_1 = coords1["num_tiles"]
    num_tiles_2 = coords2["num_tiles"]
    num_overlap_tiles = min(num_tiles_1, num_tiles_2)
    # print(f"num_tiles_1 {num_tiles_1} num_tiles_2 {num_tiles_2} --> num_overlap_tiles = {num_overlap_tiles}.")

    return overlap_start, num_overlap_tiles


def calculate_tile_overlap_ranges(lhs_tiles: SBUFTensor, rhs_tiles: SBUFTensor, result_tiles: SBUFTensor) -> dict:
    """
    Calculate the overlapping tile ranges for matrix multiplication.

    Args:
        lhs_tiles: Left-hand side SBUF tensor
        rhs_tiles: Right-hand side SBUF tensor
        result_tiles: Result SBUF tensor

    Returns:
        Dictionary containing:
        - num_tiles: (num_M_tiles, num_N_tiles, num_K_tiles)
        - global_starts: Dictionary with global tile ranges for each dimension
            - M: M_start
            - N: N_start
            - K: K_start
        - result_offsets: (M_offset, N_offset) for result tensor local indexing
    """
    # Calculate overlapping regions for each dimension (in global coordinates)
    K_start, num_K_tiles = calculate_tile_overlap(lhs_tiles.tile_coordinates["K"], rhs_tiles.tile_coordinates["K"])
    M_start, num_M_tiles = calculate_tile_overlap(lhs_tiles.tile_coordinates["M"], result_tiles.tile_coordinates["M"])
    N_start, num_N_tiles = calculate_tile_overlap(rhs_tiles.tile_coordinates["N"], result_tiles.tile_coordinates["N"])

    # Calculate local offsets for result tensor (still needed for direct tensor access)
    result_M_offset = M_start - result_tiles.tile_coordinates["M"]["start_tile_index"]
    result_N_offset = N_start - result_tiles.tile_coordinates["N"]["start_tile_index"]

    return {
        "num_tiles": (num_M_tiles, num_N_tiles, num_K_tiles),
        "global_starts": {"M": M_start, "N": N_start, "K": K_start},
        "result_offsets": (result_M_offset, result_N_offset),
    }


def lhs_rhs_gemm_np(lhs, rhs):
    """
    Calculate the general matrix multiplication (GEMM) between lhs and rhs.

    Parameters:
    -----------
    lhs : numpy.ndarray
        Left-hand side matrix or tensor. Can have an extra batch dimension.
    rhs : numpy.ndarray
        Right-hand side matrix.

    Returns:
    --------
    numpy.ndarray
        Result of the matrix multiplication.
    """
    return np.matmul(lhs, rhs)


def lhsT_rhs_gemm_np(lhsT, rhs):
    """
    Calculate the general matrix multiplication (GEMM) between lhsT and rhs.

    Parameters:
    -----------
    lhs : numpy.ndarray
        Left-hand side matrix or tensor. Can have an extra batch dimension.
    rhs : numpy.ndarray
        Right-hand side matrix.

    Returns:
    --------
    numpy.ndarray
        Result of the matrix multiplication.
    """
    if len(lhsT.shape) == 2:
        lhs = np.transpose(lhsT, (1, 0))
    elif len(lhsT.shape) == 3:  # Batch dimension exists
        lhs = np.transpose(lhsT, (0, 2, 1))
    else:
        raise NotImplementedError(f"lhsT shape {lhsT.shape} is not supported in GEMM.")
    return np.matmul(lhs, rhs)
