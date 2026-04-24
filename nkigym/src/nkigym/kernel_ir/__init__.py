"""KernelIR: structured kernel representation for lowering to NKI source.

This module exports only the core IR types. The sampler, rewrites,
builder, and validator subpackages were written against an earlier
IR schema (groups, buffer_degrees, BufferPlacement) and need to be
rewritten to match the current ``ir.py`` schema before re-exporting.
"""

from nkigym.kernel_ir.ir import BufferScope, KernelIR, NumBuffers, Op, PhysicalBuffer
from nkigym.kernel_ir.types import DimInfo, DimRole, TensorInfo

__all__ = ["BufferScope", "DimInfo", "DimRole", "KernelIR", "NumBuffers", "Op", "PhysicalBuffer", "TensorInfo"]
