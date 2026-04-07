"""Shared utilities for NKIOp render methods.

Buffer layout:
- Intra-op (staging/temp): 2D ``(op_tile_P, op_tile_F)``.
- Inter-op: 4D ``(min_tile_P, total_tiles_P, total_tiles_F, min_tile_F)``
  where ``total_tiles = num_blocks * interleave_groups_per_block``.

Flat tile index = ``i_block * intlv + i_interleave_group * gpi``.
"""


def hbm_chunk_range(block_var: str, unified: int, chunk_var: str, op_tile: int) -> str:
    """HBM slice with block and chunk offsets.

    Computes ``block*unified + chunk*op_tile`` as start.

    Args:
        block_var: Block loop variable name.
        unified: Unified tile size for this dimension.
        chunk_var: Chunk (interleave group) loop variable name.
        op_tile: Op tile size consumed per iteration.

    Returns:
        Range expression string.
    """
    start = f"{block_var}*{unified}+{chunk_var}*{op_tile}"
    return f"{start}:{start}+{op_tile}"


def intra_slice(tile_p: int, tile_f: int) -> str:
    """2D slice for intra-op buffers: ``0:tile_p, 0:tile_f``.

    Args:
        tile_p: Partition axis tile size.
        tile_f: Free axis tile size.

    Returns:
        Slice expression string.
    """
    return f"0:{tile_p}, 0:{tile_f}"


def flat_tile_index(block_var: str, intlv: int, chunk_var: str, gpi: int) -> str:
    """Flat tile index: ``block * intlv + chunk * gpi``.

    Maps (block, chunk) coordinates to a single index in the
    flattened total_tiles dimension of an inter-op buffer.

    Args:
        block_var: Block loop variable name.
        intlv: Interleave groups per block (tiles per block).
        chunk_var: Chunk (interleave group) loop variable name.
        gpi: Groups per iteration for the consuming op.

    Returns:
        Index expression string.
    """
    block_term = f"{block_var}*{intlv}" if intlv != 1 else block_var
    chunk_term = f"{chunk_var}*{gpi}" if gpi != 1 else chunk_var
    return f"{block_term}+{chunk_term}"


def emit_outer_loops(dim_order: tuple[str, ...], num_blocks_by_dim: dict[str, int]) -> list[str]:
    """Emit outer block/tile loop nest for dimensions in the given order.

    Each dimension gets a block loop and a tile loop (always 1),
    producing ``2 * len(dim_order)`` nesting levels.

    Args:
        dim_order: Ordered dimension IDs for the outer loops.
        num_blocks_by_dim: Maps dim ID to its block count.

    Returns:
        List of loop source lines with appropriate indentation.
    """
    lines: list[str] = []
    for i, dim_id in enumerate(dim_order):
        indent_b = " " * (4 * i * 2)
        indent_t = " " * (4 * (i * 2 + 1))
        lines.append(f"{indent_b}for i_block_{dim_id} in nl.affine_range({num_blocks_by_dim[dim_id]}):")
        lines.append(f"{indent_t}for i_tile_{dim_id} in nl.affine_range(1):")
    return lines


def flat_tile_range(block_var: str, intlv: int, chunk_var: str, gpi: int) -> str:
    """Flat tile range spanning ``gpi`` consecutive tiles.

    Used when an op consumes multiple interleave groups per
    iteration (gpi > 1), producing a range slice.

    Args:
        block_var: Block loop variable name.
        intlv: Interleave groups per block.
        chunk_var: Chunk loop variable name.
        gpi: Groups per iteration (range width).

    Returns:
        Range expression ``start:start+gpi``.
    """
    start = flat_tile_index(block_var, intlv, chunk_var, gpi)
    return f"{start}:{start}+{gpi}"
