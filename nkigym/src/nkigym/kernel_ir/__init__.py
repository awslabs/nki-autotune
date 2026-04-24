"""KernelIR: structured kernel representation for lowering to NKI source."""

from nkigym.kernel_ir.build import build_ir
from nkigym.kernel_ir.ir import BufferScope, KernelIR, NumBuffers, Op, PhysicalBuffer
from nkigym.kernel_ir.types import DimInfo, DimRole, TensorInfo

__all__ = [
    "BufferScope",
    "DimInfo",
    "DimRole",
    "KernelIR",
    "NumBuffers",
    "Op",
    "PhysicalBuffer",
    "TensorInfo",
    "build_ir",
]
