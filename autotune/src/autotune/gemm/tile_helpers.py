"""Tile parameter helpers for NKI GEMM kernel template."""

from autotune.gemm.config import GEMMConfig
from autotune.tensor import TileCoordinates

_POS_TO_INT: dict[str, int] = {"0": 0, "1": 1, "2": 2}


def _get_tile_size(config: GEMMConfig, axis: str) -> int:
    """Get tile size for an axis.

    Args:
        config: GEMM configuration.
        axis: Axis letter ("M", "N", or "K").

    Returns:
        Tile size for the axis.
    """
    result = config.TILE_K
    if axis == "M":
        result = config.TILE_M
    if axis == "N":
        result = config.TILE_N
    return result


def _get_tiles_per_block(config: GEMMConfig, axis: str) -> int:
    """Get tiles per block for an axis.

    Args:
        config: GEMM configuration.
        axis: Axis letter ("M", "N", or "K").

    Returns:
        Tiles per block for the axis.
    """
    result = config.TILES_PER_BLOCK_K
    if axis == "M":
        result = config.TILES_PER_BLOCK_M
    if axis == "N":
        result = config.TILES_PER_BLOCK_N
    return result


def _get_tiles_in(config: GEMMConfig, axis: str) -> int:
    """Get total tiles for an axis.

    Args:
        config: GEMM configuration.
        axis: Axis letter ("M", "N", or "K").

    Returns:
        Total tiles for the axis.
    """
    result = config.TILES_IN_K
    if axis == "M":
        result = config.TILES_IN_M
    if axis == "N":
        result = config.TILES_IN_N
    return result


def _add_axis_coords(config: GEMMConfig, axis: str, block_id: int, coords: TileCoordinates) -> None:
    """Add tile coordinates for one axis.

    Args:
        config: GEMM configuration.
        axis: Axis letter.
        block_id: Block ID for this axis, or -1 if axis not in current loop.
        coords: TileCoordinates to update.
    """
    if block_id >= 0:
        start_idx = block_id * _get_tiles_per_block(config, axis)
        num_tiles = _get_tiles_per_block(config, axis)
    else:
        start_idx = 0
        num_tiles = _get_tiles_in(config, axis)
    coords.add_axis(axis, start_idx, num_tiles)


def _get_block_id(axis: str, block_m: int, block_n: int, block_k: int) -> int:
    """Get the block ID for a specific axis from explicit block IDs.

    Args:
        axis: Axis letter ("M", "N", or "K").
        block_m: Block ID for M axis (-1 if not set).
        block_n: Block ID for N axis (-1 if not set).
        block_k: Block ID for K axis (-1 if not set).

    Returns:
        Block ID for the given axis.
    """
    result = block_k
    if axis == "M":
        result = block_m
    if axis == "N":
        result = block_n
    return result


def build_tile_sizes(par_axis: str, free_axis: str, config: GEMMConfig) -> dict[str, int]:
    """Build tile sizes dict for a tensor's axes.

    Args:
        par_axis: Partition axis name.
        free_axis: Free axis name.
        config: GEMM configuration.

    Returns:
        Dict mapping axis names to tile sizes.
    """
    tile_sizes: dict[str, int] = {}
    tile_sizes[par_axis] = _get_tile_size(config, par_axis)
    tile_sizes[free_axis] = _get_tile_size(config, free_axis)
    return tile_sizes


def build_tile_coords(
    par_axis: str, free_axis: str, config: GEMMConfig, block_m: int, block_n: int, block_k: int
) -> TileCoordinates:
    """Build tile coordinates for a tensor's axes.

    Args:
        par_axis: Partition axis name.
        free_axis: Free axis name.
        config: GEMM configuration.
        block_m: Block ID for M axis (-1 if not set).
        block_n: Block ID for N axis (-1 if not set).
        block_k: Block ID for K axis (-1 if not set).

    Returns:
        TileCoordinates with both axes registered.
    """
    coords = TileCoordinates()
    bid_par = _get_block_id(par_axis, block_m, block_n, block_k)
    bid_free = _get_block_id(free_axis, block_m, block_n, block_k)
    _add_axis_coords(config, par_axis, bid_par, coords)
    _add_axis_coords(config, free_axis, bid_free, coords)
    return coords


def _axis_in_lhs(axis: str) -> bool:
    """Check if axis belongs to LHS operand (M and K).

    Args:
        axis: Axis letter.

    Returns:
        True if axis is M or K.
    """
    result = False
    if axis == "M":
        result = True
    if axis == "K":
        result = True
    return result


def _axis_in_rhs(axis: str) -> bool:
    """Check if axis belongs to RHS operand (K and N).

    Args:
        axis: Axis letter.

    Returns:
        True if axis is K or N.
    """
    result = False
    if axis == "K":
        result = True
    if axis == "N":
        result = True
    return result


def _axis_in_result(axis: str) -> bool:
    """Check if axis belongs to result operand (M and N).

    Args:
        axis: Axis letter.

    Returns:
        True if axis is M or N.
    """
    result = False
    if axis == "M":
        result = True
    if axis == "N":
        result = True
    return result


def check_loop_needed(op_positions: dict, pos_int: int, axis: str) -> bool:
    """Check if a loop at this position is needed for any operand.

    Args:
        op_positions: Dict mapping op names to loop positions.
        pos_int: Current position as integer.
        axis: Axis letter at this position.

    Returns:
        True if the loop is needed.
    """
    needed = False
    if op_positions["lhs"] > pos_int and _axis_in_lhs(axis):
        needed = True
    if op_positions["rhs"] > pos_int and _axis_in_rhs(axis):
        needed = True
    if op_positions["result"] > pos_int and _axis_in_result(axis):
        needed = True
    return needed


def get_num_blocks(config: GEMMConfig, axis: str) -> int:
    """Get the number of blocks for a given axis.

    Args:
        config: GEMM configuration.
        axis: Axis letter ("M", "N", or "K").

    Returns:
        Number of blocks for the axis.
    """
    result = config.NUM_BLOCK_K
    if axis == "M":
        result = config.NUM_BLOCK_M
    if axis == "N":
        result = config.NUM_BLOCK_N
    return result


def set_block(axis: str, val: int, bm: int, bn: int, bk: int) -> dict[str, int]:
    """Set one block ID by axis name, return updated block IDs.

    Args:
        axis: Axis letter ("M", "N", or "K").
        val: Block ID value to set.
        bm: Current M block ID.
        bn: Current N block ID.
        bk: Current K block ID.

    Returns:
        Dict with updated M, N, K block IDs.
    """
    if axis == "M":
        bm = val
    if axis == "N":
        bn = val
    if axis == "K":
        bk = val
    return {"M": bm, "N": bn, "K": bk}


def unset_block(axis: str, bm: int, bn: int, bk: int) -> dict[str, int]:
    """Reset one block ID to -1, return updated block IDs.

    Args:
        axis: Axis letter ("M", "N", or "K").
        bm: Current M block ID.
        bn: Current N block ID.
        bk: Current K block ID.

    Returns:
        Dict with updated M, N, K block IDs.
    """
    if axis == "M":
        bm = -1
    if axis == "N":
        bn = -1
    if axis == "K":
        bk = -1
    return {"M": bm, "N": bn, "K": bk}
