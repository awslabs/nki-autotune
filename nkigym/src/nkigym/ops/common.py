"""Shared utilities for NKIOp render methods.

Buffer layout is 8-dimensional::

    (min_tile_P, num_blocks_P, num_blocks_F,
     tiles_per_block_P, tiles_per_block_F,
     interleave_P, interleave_F, min_tile_F)

All dimensions are always explicit in shapes, even when 1.
Interleave groups per dimension = ``dim_tiles[d] // dim_min_tiles[d]``.
Every op loops over ``unified // op_tile`` chunks per block.
Each chunk consumes ``op_tile // min_tile`` groups (gpi).
"""


def hbm_range(block_var: str, tile: int) -> str:
    """Format an HBM slice expression: ``var*tile:var*tile+tile``."""
    return f"{block_var}*{tile}:{block_var}*{tile}+{tile}"


def hbm_chunk_range(block_var: str, unified: int, chunk_var: str, op_tile: int) -> str:
    """HBM slice with block and chunk offsets.

    For ops whose op_tile > min_tile (e.g. matmul N=512),
    computes ``block*unified + chunk*op_tile`` as start.
    """
    start = f"{block_var}*{unified}+{chunk_var}*{op_tile}"
    return f"{start}:{start}+{op_tile}"


def d1_slice(tile_p: int, tile_f: int) -> str:
    """Degree-1 8-dim slice: partition range, all middle and interleave dims 0, free range."""
    return f"0:{tile_p}, 0, 0, 0, 0, 0, 0, 0:{tile_f}"


def sub_range(chunk_var: str, gpi: int) -> str:
    """Interleave group sub-range for a chunk: ``chunk*gpi:chunk*gpi+gpi``.

    Returns the range of interleave groups consumed by one chunk iteration.
    """
    return f"{chunk_var}*{gpi}:{chunk_var}*{gpi}+{gpi}"


def operand_slice(
    from_hbm: bool, min_p: int, block_p: str, block_f: str, group_p: str, group_f: str, min_f: int
) -> str:
    """Slice for reading an operand at a specific (block, group) position.

    For HBM operands (degree-1 staging buffer), all middle dims are 0.
    For inter-op SBUF operands, block and group variables index the buffer.
    """
    mid = "0, 0, 0, 0" if from_hbm else f"{block_p}, {block_f}, 0, 0"
    grp = "0, 0" if from_hbm else f"{group_p}, {group_f}"
    return f"0:{min_p}, {mid}, {grp}, 0:{min_f}"
