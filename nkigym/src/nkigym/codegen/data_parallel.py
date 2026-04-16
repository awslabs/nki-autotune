"""Data-parallel loop generation: outermost loops over output tile coordinates."""

from nkigym.codegen.kernel_ir import KernelIR, get_tpb


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

    all_ops = list(range(len(ir.op_graph.nodes)))
    tpb_map: dict[str, int] = {}
    for dim_id in dp_dims:
        tpb_map[dim_id] = get_tpb(ir, dim_id, all_ops)

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
