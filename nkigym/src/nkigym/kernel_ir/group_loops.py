"""Per-group loop generation: one sibling block per fusion group.

Current scope: skeleton only — block and logical-tile loops over
each fusion group's own ``dim_order``, with a ``pass`` body. DMA
loads, ISA calls, memset, and PSUM staging are not yet emitted.

A dim's blocking status (``DimInfo.is_blocking``) only affects
fusion legality, not render — every dim a group's ops touch is a
loop.
"""

from nkigym.kernel_ir.ir import KernelIR, get_tpb


def render_group_loops(ir: KernelIR, body_indent: int) -> str:
    """Emit every fusion group's loop nest as a sibling block.

    Each fusion group emits its own nest over its
    ``group_dim_orders`` entry, ordered by topological sort of the
    group-level DAG derived from ``op_graph``. Each nest has two
    phases: all block loops outermost, then all logical-tile loops.
    Physical-tile iteration is per op (hidden in gadgets) and never
    appears at the kernel level.

    Groups with an empty ``dim_order`` emit just a comment + ``pass``.

    Args:
        ir: Complete kernel IR.
        body_indent: Indentation level of the kernel body (where
            each group's comment header and first ``for`` line are
            written).

    Returns:
        Indented NKI source lines for every group's nest, with a
        ``pass`` placeholder body per group.
    """
    lines: list[str] = []
    for group_idx in ir.op_graph.toposort_groups(ir.fusion_groups):
        lines.extend(_render_group(ir, group_idx, body_indent))

    return "\n".join(lines)


def _render_group(ir: KernelIR, group_idx: int, base_indent: int) -> list[str]:
    """Render one fusion group's full loop nest (skeleton only).

    Emits a comment header, then block and logical-tile loops
    over the group's dims in ``dim_order``, then a ``pass`` body.
    Groups with an empty ``dim_order`` emit just the comment +
    ``pass``.

    Args:
        ir: Complete kernel IR.
        group_idx: Index into ``ir.fusion_groups``.
        base_indent: Indentation level for the group's top line.

    Returns:
        Source lines for the group.
    """
    da = ir.dim_analysis
    group = ir.fusion_groups[group_idx]
    dim_order = ir.group_dim_orders[group_idx]
    tpb_by_dim = {dim_id: get_tpb(ir, dim_id) for dim_id in dim_order}

    lines: list[str] = []
    pad = "    " * base_indent

    op_names = ", ".join(ir.op_graph.op_classes[i].NAME for i in group)
    dim_str = ", ".join(dim_order) if dim_order else "(none)"
    lines.append(f"{pad}# Group {group_idx}: {op_names} [dims: {dim_str}]")

    indent = base_indent

    for dim_id in dim_order:
        di = da.dims[dim_id]
        num_blocks = di.dim_size // (tpb_by_dim[dim_id] * di.logical_tile_size)
        p = "    " * indent
        lines.append(f"{p}for i_block_{dim_id} in range({num_blocks}):")
        indent += 1

    for dim_id in dim_order:
        p = "    " * indent
        lines.append(f"{p}for i_ltile_{dim_id} in range({tpb_by_dim[dim_id]}):")
        indent += 1

    body_pad = "    " * indent
    lines.append(f"{body_pad}pass")
    lines.append("")

    return lines
