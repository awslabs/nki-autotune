"""Tile overlap calculation utilities for GEMM operations."""

from autotune.tensor import SBUFTensor


def max_nki_index(index1: int, index2: int) -> int:
    """Return the effective maximum of two NKI indices.

    Treats 0 as a wildcard (unspecified), so max(0, x) = x.

    Args:
        index1: First tile index.
        index2: Second tile index.

    Returns:
        The effective maximum index.
    """
    if index1 == 0:
        max_index = index2
    elif index2 == 0:
        max_index = index1
    else:
        max_index = max(index1, index2)
    return max_index


def calculate_tile_overlap(coords1: dict[str, int], coords2: dict[str, int]) -> dict[str, int]:
    """Calculate the overlapping tile region between two tile coordinate ranges.

    Args:
        coords1: First tile coordinates with 'start_tile_index' and 'num_tiles'.
        coords2: Second tile coordinates with 'start_tile_index' and 'num_tiles'.

    Returns:
        Dict with 'start' and 'num_tiles' for the overlap region.
    """
    start_1 = coords1["start_tile_index"]
    start_2 = coords2["start_tile_index"]
    overlap_start = max_nki_index(start_1, start_2)
    num_tiles_1 = coords1["num_tiles"]
    num_tiles_2 = coords2["num_tiles"]
    num_overlap_tiles = min(num_tiles_1, num_tiles_2)
    return {"start": overlap_start, "num_tiles": num_overlap_tiles}


def _get_axis_data(tiles: SBUFTensor, axis: str) -> dict[str, int]:
    """Get tile coordinate data for an axis as a plain dict.

    Args:
        tiles: SBUF tensor with tile coordinates.
        axis: Axis name.

    Returns:
        Dict with start_tile_index and num_tiles.
    """
    return {
        "start_tile_index": tiles.tile_coordinates.get_start(axis),
        "num_tiles": tiles.tile_coordinates.get_num_tiles(axis),
    }


def calculate_tile_overlap_ranges(lhs_tiles: SBUFTensor, rhs_tiles: SBUFTensor, result_tiles: SBUFTensor) -> dict:
    """Calculate overlapping tile ranges for matrix multiplication.

    Args:
        lhs_tiles: Left-hand side SBUF tensor.
        rhs_tiles: Right-hand side SBUF tensor.
        result_tiles: Result SBUF tensor.

    Returns:
        Flat dictionary with num_M, num_N, num_K, M_start, N_start,
        K_start, res_M_off, res_N_off keys.
    """
    k_overlap = calculate_tile_overlap(_get_axis_data(lhs_tiles, "K"), _get_axis_data(rhs_tiles, "K"))
    m_overlap = calculate_tile_overlap(_get_axis_data(lhs_tiles, "M"), _get_axis_data(result_tiles, "M"))
    n_overlap = calculate_tile_overlap(_get_axis_data(rhs_tiles, "N"), _get_axis_data(result_tiles, "N"))

    return {
        "num_M": m_overlap["num_tiles"],
        "num_N": n_overlap["num_tiles"],
        "num_K": k_overlap["num_tiles"],
        "M_start": m_overlap["start"],
        "N_start": n_overlap["start"],
        "K_start": k_overlap["start"],
        "res_M_off": m_overlap["start"] - result_tiles.tile_coordinates.get_start("M"),
        "res_N_off": n_overlap["start"] - result_tiles.tile_coordinates.get_start("N"),
    }
