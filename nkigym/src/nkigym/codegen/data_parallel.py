"""Data-parallel loop generation: outermost loops over output tile coordinates."""

from nkigym.codegen.kernel_ir import KernelIR


def render_data_parallel_loops(ir: KernelIR) -> tuple[str, int]:
    """Emit the data-parallel loop nest from a KernelIR.

    Each data-parallel dimension (``is_data_parallel=True``)
    contributes 3 loops: block, tile, interleave. Loops are
    grouped by phase — all block loops outermost, then all tile
    loops, then all interleave loops. Within each phase,
    dimensions are ordered by dimension ID.

    Args:
        ir: Complete kernel IR.

    Returns:
        Tuple of (source lines, body indent level). The source
        lines contain the loop headers without a body placeholder.
    """
    da = ir.dim_analysis

    dp_dims = [dim_id for dim_id in sorted(da.dims) if da.dims[dim_id].is_data_parallel]

    tpb_map: dict[str, int] = {}
    for dim_id in dp_dims:
        tpb_map[dim_id] = _get_tpb(ir, dim_id)

    lines: list[str] = []
    indent = 1

    for dim_id in dp_dims:
        di = da.dims[dim_id]
        num_blocks = di.dim_size // (tpb_map[dim_id] * di.tile_size)
        pad = "    " * indent
        lines.append(f"{pad}for i_block_{dim_id} in range({num_blocks}):")
        indent += 1

    for dim_id in dp_dims:
        pad = "    " * indent
        lines.append(f"{pad}for i_tile_{dim_id} in range({tpb_map[dim_id]}):")
        indent += 1

    for dim_id in dp_dims:
        di = da.dims[dim_id]
        num_ig = di.tile_size // di.min_tile_size
        pad = "    " * indent
        lines.append(f"{pad}for i_ig_{dim_id} in range({num_ig}):")
        indent += 1

    return "\n".join(lines), indent


def _get_tpb(ir: KernelIR, dim_id: str) -> int:
    """Get tiles_per_block for a data-parallel dimension.

    All ops agree on tiles_per_block for a given dimension,
    so we take the value from the first op that has the key.
    """
    tpb = 1
    for op_idx in range(len(ir.op_graph.nodes)):
        key = (op_idx, dim_id)
        if key in ir.tiles_per_block:
            tpb = ir.tiles_per_block[key]
            break
    return tpb
