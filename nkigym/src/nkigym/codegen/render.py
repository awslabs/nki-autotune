"""render_ir: mechanical lowering of KernelIR to NKI source code."""

from nkigym.codegen.data_parallel import render_data_parallel_loops
from nkigym.codegen.header import render_header, render_return
from nkigym.codegen.kernel_ir import KernelIR


def render_ir(ir: KernelIR) -> str:
    """Lower a KernelIR to NKI source code.

    Emits the kernel header, data-parallel loop nest, and
    return statement. The body inside the loops is a ``...``
    placeholder for now.

    Args:
        ir: Complete kernel IR.

    Returns:
        Complete NKI kernel source code.
    """
    header = render_header(ir.dim_analysis)
    dp_loops = render_data_parallel_loops(ir)
    ret = render_return(ir.dim_analysis)

    parts = [header]
    if dp_loops:
        parts.append(dp_loops)
    parts.append(ret)

    return "\n".join(parts) + "\n"
