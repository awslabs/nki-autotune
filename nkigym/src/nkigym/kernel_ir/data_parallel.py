"""Data-parallel loop generation: outermost loops over output tile coordinates."""

from nkigym.kernel_ir.ir import KernelIR, get_tpb


def render_data_parallel_loops(ir: KernelIR, body_indent: int) -> tuple[str, int]:
    """Emit the data-parallel loop nest from a KernelIR.

    Each data-parallel dimension contributes 2 loops at the
    kernel level: block and logical tile. Physical-tile packing
    is per-op and hidden inside op gadgets.

    Loops are grouped by phase — all block loops outermost, then
    all logical tile loops. Dimension order within each phase is
    taken from ``ir.loop_order`` (the top-level string entries).

    Args:
        ir: Complete kernel IR.
        body_indent: Indentation level of the line *containing*
            the outermost ``for i_block_*`` (i.e. the level the
            innermost loop body starts at is ``body_indent +
            2 * num_dp_dims``).

    Returns:
        Tuple of ``(source_lines, inner_indent)`` where
        ``inner_indent`` is the indentation level of the
        innermost ``i_ltile_*`` loop body.
    """
    da = ir.dim_analysis

    dp_dims = [entry for entry in ir.loop_order if isinstance(entry, str)]
    dp_set = {d for d, di in da.dims.items() if di.is_data_parallel}
    if set(dp_dims) != dp_set:
        raise ValueError(f"loop_order DP entries {dp_dims} do not match DP dims {sorted(dp_set)}")

    tpb_map: dict[str, int] = {dim_id: get_tpb(ir, dim_id) for dim_id in dp_dims}

    lines: list[str] = []
    indent = body_indent

    for dim_id in dp_dims:
        di = da.dims[dim_id]
        num_blocks = di.dim_size // (tpb_map[dim_id] * di.logical_tile_size)
        pad = "    " * indent
        lines.append(f"{pad}for i_block_{dim_id} in range({num_blocks}):")
        indent += 1

    for dim_id in dp_dims:
        pad = "    " * indent
        lines.append(f"{pad}for i_ltile_{dim_id} in range({tpb_map[dim_id]}):")
        indent += 1

    return "\n".join(lines), indent
