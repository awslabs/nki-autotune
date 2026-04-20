"""Per-group loop generation: one sibling block per fusion group.

A dim's blocking status (``DimInfo.is_blocking``) only affects
fusion legality, not render — every dim a group's ops touch is a
loop.
"""

from nkigym.kernel_ir.ir import KernelIR, get_tpb

DepthPlan = dict[int, dict[int, list[str]]]


def render_group_loops(
    ir: KernelIR, body_indent: int, before_plan: DepthPlan | None = None, after_plan: DepthPlan | None = None
) -> str:
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
        before_plan: ``{group_idx: {depth: [lines]}}`` — lines
            injected at indent ``body_indent + depth`` BEFORE the
            loop header at that depth is written (for depth 0:
            before the group's first block loop; for depth
            ``2 * N``: at the innermost body). Used for HBM loads.
        after_plan: ``{group_idx: {depth: [lines]}}`` — lines
            injected at indent ``body_indent + depth`` AFTER the
            loop whose body lives at indent ``body_indent + depth
            + 1`` has closed. Equivalently, ``depth`` is the indent
            of the ``for`` header whose body just finished. Used
            for PSUM→SBUF staging after a blocking dim's loop
            closes.

    Returns:
        Indented NKI source lines for every group's nest.
    """
    before = before_plan or {}
    after = after_plan or {}
    lines: list[str] = []
    for group_idx in ir.op_graph.toposort_groups([g.op_indices for g in ir.fusion_groups]):
        lines.extend(_render_group(ir, group_idx, body_indent, before.get(group_idx, {}), after.get(group_idx, {})))

    return "\n".join(lines)


def _render_group(
    ir: KernelIR,
    group_idx: int,
    base_indent: int,
    before_lines: dict[int, list[str]],
    after_lines: dict[int, list[str]],
) -> list[str]:
    """Render one fusion group's full loop nest.

    Emits a comment header, then block and logical-tile loops
    over the group's dims in ``dim_order``, then a ``pass`` body.
    ``before_lines[d]`` is injected at depth ``d`` just before the
    loop at that depth opens; ``after_lines[d]`` is injected after
    the loop at depth ``d`` closes. Groups with an empty
    ``dim_order`` emit just the comment + ``pass``.

    Args:
        ir: Complete kernel IR.
        group_idx: Index into ``ir.fusion_groups``.
        base_indent: Indentation level for the group's top line.
        before_lines: Per-depth lines emitted BEFORE the loop at
            that depth opens.
        after_lines: Per-depth lines emitted AFTER the loop at
            that depth closes.

    Returns:
        Source lines for the group.
    """
    da = ir.dim_analysis
    group = ir.fusion_groups[group_idx]
    dim_order = group.dim_order
    n = len(dim_order)
    tpb_by_dim = {dim_id: get_tpb(ir, dim_id) for dim_id in dim_order}

    lines: list[str] = []
    op_names = ", ".join(ir.op_graph.op_classes[i].NAME for i in group.op_indices)
    dim_str = ", ".join(dim_order) if dim_order else "(none)"
    lines.append("    " * base_indent + f"# Group {group_idx}: {op_names} [dims: {dim_str}]")

    def inject(plan: dict[int, list[str]], depth: int) -> None:
        """Append plan[depth] lines at indent ``base_indent + depth``."""
        for line in plan.get(depth, []):
            lines.append("    " * (base_indent + depth) + line)

    inject(before_lines, 0)
    for i, dim_id in enumerate(dim_order):
        di = da.dims[dim_id]
        num_blocks = di.dim_size // (tpb_by_dim[dim_id] * di.logical_tile_size)
        lines.append("    " * (base_indent + i) + f"for i_block_{dim_id} in range({num_blocks}):")
        inject(before_lines, i + 1)

    for i, dim_id in enumerate(dim_order):
        lines.append("    " * (base_indent + n + i) + f"for i_ltile_{dim_id} in range({tpb_by_dim[dim_id]}):")
        inject(before_lines, n + i + 1)

    if n > 0 and not before_lines.get(2 * n):
        lines.append("    " * (base_indent + 2 * n) + "pass")
    if n == 0 and not before_lines.get(0) and not after_lines.get(0):
        lines.append("    " * base_indent + "pass")

    for i in reversed(range(n)):
        inject(after_lines, n + i)
    for i in reversed(range(n)):
        inject(after_lines, i)
    lines.append("")

    return lines
