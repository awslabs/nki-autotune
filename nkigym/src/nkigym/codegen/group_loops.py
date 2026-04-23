"""Per-group loop generation: one sibling block per fusion group."""

from nkigym.codegen.matmul_block_detect import gadget_absorbed_dims
from nkigym.kernel_ir import KernelIR
from nkigym.kernel_ir.context.context import DimInfo
from nkigym.kernel_ir.validate.emission import block_depth, body_depth, ltile_depth

DepthPlan = dict[int, dict[int, list[str]]]


def _dim_range_fn(di: DimInfo) -> str:
    """Pick the loop helper for ``di`` based on role.

    Currently emits plain ``range`` for both PARALLEL and SERIAL
    dims — experimental A/B with the NKI-specific hints showed
    no MFU delta in the matmul_block dispatch path.
    """
    _ = di
    return "range"


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
    """Render one fusion group's full loop nest as pair-interleaved ``(block, ltile)`` per dim.

    When a group contains a matmul-block-dispatched ``NKIMatmul``,
    the ltile loops for the matmul's axis dims are suppressed —
    the gadget iterates those internally. ``depth_indent`` tracks
    the Python indent level independently of logical slot depth so
    suppressed loops don't leave stair-step indentation gaps.
    """
    context = ir.context
    group = ir.graph.groups[group_idx]
    dim_order = group.dim_order
    n = len(dim_order)
    tpb_by_dim = {dim_id: context.ltiles_per_block.get(dim_id, 1) for dim_id in dim_order}
    absorbed = gadget_absorbed_dims(ir, group_idx)
    depth_indent = _compute_depth_indent(dim_order, absorbed, base_indent)

    lines: list[str] = []
    op_names = ", ".join(type(op).NAME for op in group.ops)
    dim_str = ", ".join(dim_order) if dim_order else "(none)"
    lines.append("    " * base_indent + f"# Group {group_idx}: {op_names} [dims: {dim_str}]")

    def inject(plan: dict[int, list[str]], depth: int) -> None:
        """Append plan[depth] lines at the Python indent for ``depth``."""
        for line in plan.get(depth, []):
            lines.append("    " * depth_indent[depth] + line)

    inject(before_lines, 0)
    for pos, dim_id in enumerate(dim_order):
        di = context.dimensions[dim_id]
        num_blocks = di.dim_size // (tpb_by_dim[dim_id] * di.logical_tile_size)
        b_depth = block_depth(pos)
        range_fn = _dim_range_fn(di)
        lines.append("    " * depth_indent[b_depth] + f"for i_block_{dim_id} in {range_fn}({num_blocks}):")
        inject(before_lines, b_depth + 1)
        l_depth = ltile_depth(pos)
        if dim_id not in absorbed:
            lines.append("    " * depth_indent[l_depth] + f"for i_ltile_{dim_id} in {range_fn}({tpb_by_dim[dim_id]}):")
        inject(before_lines, l_depth + 1)

    body = body_depth(n)
    if n > 0 and not before_lines.get(body):
        lines.append("    " * depth_indent[body] + "pass")
    if n == 0 and not before_lines.get(0) and not after_lines.get(0):
        lines.append("    " * base_indent + "pass")

    for pos in reversed(range(n)):
        inject(after_lines, ltile_depth(pos))
        inject(after_lines, block_depth(pos))
    lines.append("")
    return lines


def _compute_depth_indent(dim_order: list[str], absorbed: set[str], base_indent: int) -> dict[int, int]:
    """Map each logical slot depth to a Python indent level (# leading 4-space groups).

    Each non-absorbed ``for`` loop adds one indent level; absorbed
    ltile loops are skipped but still consume a logical depth slot
    so emission placements can reason about "between block and
    ltile" without issue.
    """
    n = len(dim_order)
    indent: dict[int, int] = {}
    current = base_indent
    indent[0] = current
    for pos, dim_id in enumerate(dim_order):
        current += 1
        b_depth = block_depth(pos)
        indent[b_depth + 1] = current
        l_depth = ltile_depth(pos)
        if dim_id not in absorbed:
            current += 1
        indent[l_depth + 1] = current
    indent[body_depth(n)] = current
    return indent
