"""render_ir: mechanical lowering of KernelIR to NKI source code."""

from nkigym.codegen.buffers import render_buffers
from nkigym.codegen.dma import build_op_to_group, render_hbm_loads, render_hbm_store, render_psum_staging
from nkigym.codegen.group_loops import render_group_loops
from nkigym.codegen.header import render_header, render_return
from nkigym.kernel_ir import KernelIR


def render_ir(ir: KernelIR) -> str:
    """Lower a KernelIR to NKI source code.

    Emits: header, tensor buffer allocations, one sibling loop
    nest per fusion group, HBM→SBUF loads at per-dim derived
    depths, PSUM→SBUF staging (innermost body for non-blocking
    producers, after the outermost blocking dim's block loop
    closes for blocking ones), and an SBUF→HBM store at the
    producing group's innermost body. Loop bodies hold ``pass``
    until ISA emission lands.
    """
    op_to_group = build_op_to_group(ir)
    header = render_header(ir.dim_analysis)
    buffers = render_buffers(ir, indent=1)

    before_plan = render_hbm_loads(ir, op_to_group)
    staging_before, staging_after = render_psum_staging(ir, op_to_group)
    for group_idx, depth_lines in staging_before.items():
        for depth, lines in depth_lines.items():
            before_plan.setdefault(group_idx, {}).setdefault(depth, []).extend(lines)

    store = render_hbm_store(ir, op_to_group)
    if store is not None:
        store_group, store_depth, store_line = store
        before_plan.setdefault(store_group, {}).setdefault(store_depth, []).append(store_line)

    group_src = render_group_loops(ir, body_indent=1, before_plan=before_plan, after_plan=staging_after)
    ret = render_return(ir.dim_analysis)

    parts = [header, buffers, group_src, ret]
    return "\n".join(parts) + "\n"
