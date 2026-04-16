"""Code generation: lowering KernelIR to NKI source code."""

from nkigym.codegen.render import render_ir
from nkigym.header.header import render_header, render_return
from nkigym.kernel_ir import KernelIR, build_ir

__all__ = ["KernelIR", "build_ir", "render_header", "render_ir", "render_return"]
