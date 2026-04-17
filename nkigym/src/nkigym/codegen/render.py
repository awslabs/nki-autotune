"""render_ir: mechanical lowering of KernelIR to NKI source code.

Current scope: header, data-parallel block/logical-tile loops,
tensor buffer allocations at the top of the innermost DP body,
and per-fusion-group reduction-loop skeletons (``pass`` bodies).
DMA and ISA calls remain commented out until those stages are
verified.
"""

from nkigym.codegen.buffers import render_buffers
from nkigym.codegen.header import render_header, render_return
from nkigym.kernel_ir import KernelIR
from nkigym.kernel_ir.data_parallel import render_data_parallel_loops
from nkigym.kernel_ir.reduction import render_reduction_loops


def render_ir(ir: KernelIR) -> str:
    """Lower a KernelIR to NKI source code.

    Current scope: kernel header, the data-parallel block and
    logical-tile loops, tensor buffer allocations at the top of
    the innermost DP body, and per-fusion-group reduction-loop
    skeletons (``pass`` bodies), HBM output allocation, and
    return statement. DMA and ISA calls are not yet emitted.

    Args:
        ir: Complete kernel IR.

    Returns:
        NKI kernel source with DP loops, buffer allocations, and
        reduction loop skeletons.
    """
    da = ir.dim_analysis
    header = render_header(da)
    dp_src, inner_indent = render_data_parallel_loops(ir, body_indent=1)
    buffers = render_buffers(ir, inner_indent)
    reduction = render_reduction_loops(ir, body_indent=inner_indent)
    ret = render_return(da)

    parts = [header, dp_src, buffers, reduction, ret]
    return "\n".join(parts) + "\n"
