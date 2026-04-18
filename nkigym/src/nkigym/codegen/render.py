"""render_ir: mechanical lowering of KernelIR to NKI source code."""

from nkigym.codegen.buffers import find_psum_tensors_needing_sbuf, render_buffers
from nkigym.codegen.dma import build_op_to_group, render_hbm_loads, render_hbm_store, render_psum_staging
from nkigym.codegen.group_loops import DepthPlan, render_group_loops
from nkigym.codegen.header import render_header, render_return
from nkigym.codegen.nki_ops import render_nki_ops
from nkigym.kernel_ir import KernelIR


def render_ir(ir: KernelIR) -> str:
    """Lower a KernelIR to NKI source code.

    Emits: header, tensor buffer allocations, one sibling loop
    nest per fusion group, HBM→SBUF loads at per-dim derived
    depths, per-op PSUM memsets and ISA calls, PSUM→SBUF staging
    (innermost body for non-blocking producers, after the
    outermost blocking dim's block loop closes for blocking
    ones), and an SBUF→HBM store at the producing group's
    innermost body.
    """
    op_to_group = build_op_to_group(ir)
    staged = find_psum_tensors_needing_sbuf(ir)
    header = render_header(ir.dim_analysis)
    buffers = render_buffers(ir, indent=1, staged=staged)

    before_plan: DepthPlan = render_hbm_loads(ir, op_to_group)
    _merge(before_plan, render_nki_ops(ir, op_to_group, staged))
    staging_before, staging_after = render_psum_staging(ir, op_to_group, staged)
    _merge(before_plan, staging_before)

    store_before, store_after = render_hbm_store(ir, op_to_group)
    _merge(before_plan, store_before)
    _merge(staging_after, store_after)

    group_src = render_group_loops(ir, body_indent=1, before_plan=before_plan, after_plan=staging_after)
    ret = render_return(ir.dim_analysis)

    parts = [header, buffers, group_src, ret]
    return "\n".join(parts) + "\n"


def _merge(dst: DepthPlan, src: DepthPlan) -> None:
    """Merge ``src`` into ``dst`` in place, preserving line order."""
    for group_idx, depth_lines in src.items():
        for depth, lines in depth_lines.items():
            dst.setdefault(group_idx, {}).setdefault(depth, []).extend(lines)
