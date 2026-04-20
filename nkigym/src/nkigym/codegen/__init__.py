"""Code generation: lowering KernelIR to NKI source code."""

from nkigym.codegen.header import render_header, render_return
from nkigym.codegen.render import render_ir

__all__ = ["render_header", "render_ir", "render_return"]
