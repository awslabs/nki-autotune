"""Code generation: lowering KernelIR to NKI source code."""

from nkigym.codegen.buffers import render_buffers
from nkigym.codegen.data_parallel import render_data_parallel_loops
from nkigym.codegen.dma import render_loads_for_group, render_store
from nkigym.codegen.header import render_header, render_return
from nkigym.codegen.kernel_ir import KernelIR, build_ir
from nkigym.codegen.reduction import render_reduction_loops
from nkigym.codegen.render import render_ir

__all__ = [
    "KernelIR",
    "build_ir",
    "render_buffers",
    "render_data_parallel_loops",
    "render_header",
    "render_ir",
    "render_loads_for_group",
    "render_reduction_loops",
    "render_return",
    "render_store",
]
