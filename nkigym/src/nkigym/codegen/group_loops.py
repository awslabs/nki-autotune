"""Per-group loop generation: one sibling block per fusion group."""

from nkigym.kernel_ir import KernelIR

DepthPlan = dict[int, dict[int, list[str]]]


def render_group_loops(
    ir: KernelIR, body_indent: int, before_plan: DepthPlan | None = None, after_plan: DepthPlan | None = None
) -> str:
    """Emit every fusion group's loop nest as a sibling block."""
    before = before_plan or {}
    after = after_plan or {}
    lines: list[str] = []
    for group_idx in ir.graph.toposort_groups():
        lines.extend(_render_group(ir, group_idx, body_indent, before.get(group_idx, {}), after.get(group_idx, {})))
    return "\n".join(lines)


def _render_group(
    ir: KernelIR,
    group_idx: int,
    base_indent: int,
    before_lines: dict[int, list[str]],
    after_lines: dict[int, list[str]],
) -> list[str]:
    """Render one fusion group's full loop nest."""
    context = ir.context
    group = ir.graph.groups[group_idx]
    dim_order = group.dim_order
    n = len(dim_order)
    tpb_by_dim = {dim_id: context.ltiles_per_block.get(dim_id, 1) for dim_id in dim_order}

    lines: list[str] = []
    op_names = ", ".join(type(op).NAME for op in group.ops)
    dim_str = ", ".join(dim_order) if dim_order else "(none)"
    lines.append("    " * base_indent + f"# Group {group_idx}: {op_names} [dims: {dim_str}]")

    def inject(plan: dict[int, list[str]], depth: int) -> None:
        """Append plan[depth] lines at indent ``base_indent + depth``."""
        for line in plan.get(depth, []):
            lines.append("    " * (base_indent + depth) + line)

    inject(before_lines, 0)
    for i, dim_id in enumerate(dim_order):
        di = context.dimensions[dim_id]
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
