"""render_ir: mechanical lowering of KernelIR to NKI source code."""

from nkigym.codegen.buffers import find_psum_tensors_needing_sbuf, render_buffers
from nkigym.codegen.dma import render_store
from nkigym.codegen.header import render_header, render_return
from nkigym.kernel_ir import KernelIR
from nkigym.kernel_ir.data_parallel import render_data_parallel_loops
from nkigym.kernel_ir.reduction import render_reduction_loops


def render_ir(ir: KernelIR) -> str:
    """Lower a KernelIR to NKI source code.

    Emits the kernel header, data-parallel loop nest, buffer
    allocations, reduction loop bodies with DMA loads, store,
    and return statement.

    Args:
        ir: Complete kernel IR.

    Returns:
        Complete NKI kernel source code.
    """
    needs_staging = find_psum_tensors_needing_sbuf(ir)
    header = render_header(ir.dim_analysis)
    dp_loops, dp_indent = render_data_parallel_loops(ir)
    buffers = render_buffers(ir, dp_indent, needs_staging)
    reduction = render_reduction_loops(ir, dp_indent, needs_staging)
    store = render_store(ir, dp_indent)
    ret = render_return(ir.dim_analysis)

    parts = [header]
    if dp_loops:
        parts.append(dp_loops)
    if buffers:
        parts.append(buffers)
    if reduction:
        parts.append(reduction)
    if store:
        parts.append(store)
    parts.append(ret)

    return "\n".join(parts) + "\n"
