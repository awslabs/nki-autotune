"""render_ir: mechanical lowering of KernelIR to NKI source code.

Current scope: header, tensor buffer allocations, and per-fusion-group
loop-nest skeletons (``pass`` bodies). Each fusion group emits its own
complete nest over its ``group_dim_orders`` entry. DMA and ISA calls
remain commented out until those stages are verified.
"""

from nkigym.codegen.buffers import render_buffers
from nkigym.codegen.header import render_header, render_return
from nkigym.kernel_ir import KernelIR
from nkigym.kernel_ir.group_loops import render_group_loops


def render_ir(ir: KernelIR) -> str:
    """Lower a KernelIR to NKI source code.

    Current scope: kernel header, tensor buffer allocations, and
    one sibling loop-nest skeleton per fusion group (``pass`` body),
    HBM output allocation, and return statement. DMA and ISA calls
    are not yet emitted.

    Args:
        ir: Complete kernel IR.

    Returns:
        NKI kernel source with buffer allocations and per-group
        loop skeletons.
    """
    header = render_header(ir.dim_analysis)
    buffers = render_buffers(ir, indent=1)
    group_src = render_group_loops(ir, body_indent=1)
    ret = render_return(ir.dim_analysis)

    parts = [header, buffers, group_src, ret]
    return "\n".join(parts) + "\n"
