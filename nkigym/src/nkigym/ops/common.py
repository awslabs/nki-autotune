"""Shared utilities for NKIOp render methods.

Buffer layout:
- Intra-op (staging/temp): 2D ``(op_tile_P, op_tile_F)``.
- Inter-op: 4D ``(min_tile_P, total_tiles_P, total_tiles_F, min_tile_F)``
  where ``total_tiles = num_blocks * tpb_hbm * interleave``.

Loop nest per dimension (outer to inner):
  ``i_block``, ``i_psum_batch``, ``i_tile``, ``i_ig``.

Flat tile index = ``i_block * (tpb_hbm * interleave)
    + i_psum_batch * (tpb_psum * interleave)
    + i_tile * interleave + i_ig * gpi``.
"""


def ind(depth: int) -> str:
    """Return indentation string for the given nesting depth.

    Args:
        depth: Number of indent levels (each level = 4 spaces).

    Returns:
        Indentation string.
    """
    return " " * (4 * depth)


def linear_expr(terms: list[tuple[str, int, int]]) -> str:
    """Build a linear index expression, omitting zero-contribution terms.

    Each term is ``(variable_name, stride, trip_count)``.  Terms whose
    ``trip_count <= 1`` are omitted because the variable is always 0
    (from a ``range(1)`` loop).

    Args:
        terms: List of (variable_name, stride, trip_count) tuples.

    Returns:
        Expression string, or ``"0"`` if all terms are omitted.
    """
    parts: list[str] = []
    for var, stride, trip_count in terms:
        if trip_count <= 1:
            continue
        if stride == 1:
            parts.append(var)
        else:
            parts.append(f"{var}*{stride}")
    return "+".join(parts) if parts else "0"


def linear_range(terms: list[tuple[str, int, int]], width: int) -> str:
    """Build a linear range expression ``start:start+width``.

    Args:
        terms: Same as ``linear_expr``.
        width: Range width appended as ``+width``.

    Returns:
        Range expression string ``start:start+width``.
    """
    start = linear_expr(terms)
    return f"{start}:{start}+{width}"


def intra_slice(tile_p: int, tile_f: int) -> str:
    """2D slice for intra-op buffers: ``0:tile_p, 0:tile_f``.

    Args:
        tile_p: Partition axis tile size.
        tile_f: Free axis tile size.

    Returns:
        Slice expression string.
    """
    return f"0:{tile_p}, 0:{tile_f}"


def emit_outer_loops(
    dim_order: tuple[str, ...],
    num_blocks_by_dim: dict[str, int],
    tpb_hbm_by_dim: dict[str, int],
    tpb_psum_by_dim: dict[str, int],
) -> list[str]:
    """Emit outer block/psum_batch/tile loop nest for dimensions in order.

    Each dimension gets three loops — ``i_block``, ``i_psum_batch``,
    ``i_tile`` — producing ``3 * len(dim_order)`` nesting levels.

    Args:
        dim_order: Ordered dimension IDs for the outer loops.
        num_blocks_by_dim: Maps dim ID to block count.
        tpb_hbm_by_dim: Maps dim ID to tiles per HBM block.
        tpb_psum_by_dim: Maps dim ID to tiles per PSUM batch.

    Returns:
        List of loop source lines with appropriate indentation.
    """
    lines: list[str] = []
    for i, dim_id in enumerate(dim_order):
        base = i * 3
        psum_batches = tpb_hbm_by_dim[dim_id] // tpb_psum_by_dim[dim_id]
        lines.append(f"{ind(base)}for i_block_{dim_id} in range({num_blocks_by_dim[dim_id]}):")
        lines.append(f"{ind(base + 1)}for i_psum_batch_{dim_id} in range({psum_batches}):")
        lines.append(f"{ind(base + 2)}for i_tile_{dim_id} in range({tpb_psum_by_dim[dim_id]}):")
    return lines


def flat_terms(cfg: dict, suffix: str, gpi_override: int | None = None) -> list[tuple[str, int, int]]:
    """Build linear_expr terms for flat tile index of a dimension.

    Args:
        cfg: Configuration dict.
        suffix: Dimension suffix (e.g. ``"K"``, ``"M"``).
        gpi_override: Override the gpi value (default uses cfg).

    Returns:
        List of (var, stride, trip_count) tuples for ``linear_expr``.
    """
    dim = cfg[f"dim_{suffix}"]
    gpi = gpi_override if gpi_override is not None else cfg[f"gpi_{suffix}"]
    interleave = cfg[f"interleave_{suffix}"]
    ig_trips = interleave // gpi if gpi > 0 else 0
    return [
        (f"i_block_{dim}", cfg[f"tpb_hbm_{suffix}"] * interleave, cfg[f"num_blocks_{suffix}"]),
        (f"i_psum_batch_{dim}", cfg[f"tpb_psum_{suffix}"] * interleave, cfg[f"psum_batches_{suffix}"]),
        (f"i_tile_{dim}", interleave, cfg[f"tpb_psum_{suffix}"]),
        (f"i_ig_{dim}", gpi, ig_trips),
    ]


def flat_terms_merged(cfg: dict, suffix: str) -> list[tuple[str, int, int]]:
    """Build flat terms for reshaped buffer read (gpi > 1 merge).

    When an op consumes multiple interleave groups per iteration,
    the index into the merged dimension uses ``ig_trips``
    as the stride unit.

    Args:
        cfg: Configuration dict.
        suffix: Dimension suffix.

    Returns:
        List of (var, stride, trip_count) tuples for ``linear_expr``.
    """
    dim = cfg[f"dim_{suffix}"]
    ig_trips = cfg[f"ig_trips_{suffix}"]
    return [
        (f"i_block_{dim}", cfg[f"tpb_hbm_{suffix}"] * ig_trips, cfg[f"num_blocks_{suffix}"]),
        (f"i_psum_batch_{dim}", cfg[f"tpb_psum_{suffix}"] * ig_trips, cfg[f"psum_batches_{suffix}"]),
        (f"i_tile_{dim}", ig_trips, cfg[f"tpb_psum_{suffix}"]),
        (f"i_ig_{dim}", 1, ig_trips),
    ]


def hbm_range(cfg: dict, suffix: str) -> str:
    """HBM slice range for DMA load/store of a dimension.

    Args:
        cfg: Configuration dict.
        suffix: Dimension suffix.

    Returns:
        Range expression string ``start:start+op_tile``.
    """
    dim = cfg[f"dim_{suffix}"]
    max_tile = cfg[f"max_tile_{suffix}"]
    op_tile = cfg[f"op_{suffix}"]
    return linear_range(
        [
            (f"i_block_{dim}", cfg[f"tpb_hbm_{suffix}"] * max_tile, cfg[f"num_blocks_{suffix}"]),
            (f"i_psum_batch_{dim}", cfg[f"tpb_psum_{suffix}"] * max_tile, cfg[f"psum_batches_{suffix}"]),
            (f"i_tile_{dim}", max_tile, cfg[f"tpb_psum_{suffix}"]),
            (f"i_ig_{dim}", op_tile, cfg[f"ig_trips_{suffix}"]),
        ],
        op_tile,
    )
