"""render_ir: mechanical lowering of KernelIR to NKI source code.

Current scope: header, data-parallel block/logical-tile loops,
and per-fusion-group reduction-loop skeletons (``pass`` bodies).
Buffer allocation, DMA, and ISA calls remain commented out until
those stages are verified.
"""

from nkigym.codegen.header import render_header, render_return
from nkigym.kernel_ir import KernelIR
from nkigym.kernel_ir.data_parallel import render_data_parallel_loops
from nkigym.kernel_ir.reduction import render_reduction_loops

"""
from collections import defaultdict

from nkigym.codegen.buffers import find_psum_tensors_needing_sbuf, render_buffers_for_names
from nkigym.codegen.dma import render_store
"""


def render_ir(ir: KernelIR) -> str:
    """Lower a KernelIR to NKI source code.

    Current scope: kernel header, the data-parallel block and
    logical-tile loops, and per-fusion-group reduction-loop
    skeletons (``pass`` bodies), HBM output allocation, and
    return statement. Tensor buffers, DMA, and ISA calls are not
    yet emitted.

    Args:
        ir: Complete kernel IR.

    Returns:
        NKI kernel source with DP and reduction loop skeletons.
    """
    da = ir.dim_analysis
    header = render_header(da)
    dp_src, inner_indent = render_data_parallel_loops(ir, body_indent=1)
    reduction = render_reduction_loops(ir, body_indent=inner_indent)
    ret = render_return(da)

    parts = [header, dp_src, reduction, ret]
    return "\n".join(parts) + "\n"
