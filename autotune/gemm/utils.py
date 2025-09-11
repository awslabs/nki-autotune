# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from autotune.core.tensor import SBUFTensor


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
