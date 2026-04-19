"""render_ir: mechanical lowering of KernelIR to NKI source code."""

from nkigym.codegen.buffers import (
    build_tensor_to_groups,
    find_psum_tensors_needing_sbuf,
    prime_sbuf_cache,
    render_psum_allocations,
    render_sbuf_buffers,
)
from nkigym.codegen.dma import build_op_to_group, render_hbm_loads, render_hbm_store, render_psum_staging
from nkigym.codegen.group_loops import DepthPlan, render_group_loops
from nkigym.codegen.header import render_header, render_return
from nkigym.codegen.nki_ops import render_nki_ops
from nkigym.kernel_ir import KernelIR


def render_ir(ir: KernelIR) -> str:
    """Lower a KernelIR to NKI source code.

    Emits: header, then one sibling loop nest per fusion group.
    Each group's nest opens with buffer declarations at depth 0 —
    first any persistent SBUF tensors (touched by 2+ groups, whose
    first user is this group), then per-FG SBUF tensors (local to
    this group, stacked after the persistent range), then all
    PSUM buffers produced in this group. PSUM memsets for blocking
    producers fire at the outermost blocking loop depth so the
    accumulator is re-zeroed each output tile. HBM→SBUF loads fire
    at per-dim derived depths; ISA calls sit at the innermost body;
    PSUM→SBUF staging lands after the outermost blocking loop
    closes for blocking producers (innermost body otherwise); and
    an SBUF→HBM store fires at the producing group's deepest depth.
    """
    prime_sbuf_cache(ir)
    op_to_group = build_op_to_group(ir)
    tensor_to_groups = build_tensor_to_groups(ir)
    staged = find_psum_tensors_needing_sbuf(ir)
    header = render_header(ir.dim_analysis)

    before_plan: DepthPlan = render_hbm_loads(ir, op_to_group)
    _merge(before_plan, render_nki_ops(ir, op_to_group, staged))
    staging_before, staging_after = render_psum_staging(ir, op_to_group, staged)
    _merge(before_plan, staging_before)

    store_before, store_after = render_hbm_store(ir, op_to_group)
    _merge(before_plan, store_before)
    _merge(staging_after, store_after)

    sbuf_by_group = render_sbuf_buffers(ir, staged=staged, tensor_to_groups=tensor_to_groups)
    psum_allocs = render_psum_allocations(ir, op_to_group)
    for group_idx in range(len(ir.fusion_groups)):
        group_top = sbuf_by_group.get(group_idx, []) + psum_allocs.get(group_idx, [])
        if group_top:
            before_plan.setdefault(group_idx, {}).setdefault(0, [])[:0] = group_top

    group_src = render_group_loops(ir, body_indent=1, before_plan=before_plan, after_plan=staging_after)
    ret = render_return(ir.dim_analysis)

    return "\n".join([header, group_src, ret]) + "\n"


def _merge(dst: DepthPlan, src: DepthPlan) -> None:
    """Merge ``src`` into ``dst`` in place, preserving line order."""
    for group_idx, depth_lines in src.items():
        for depth, lines in depth_lines.items():
            dst.setdefault(group_idx, {}).setdefault(depth, []).extend(lines)
