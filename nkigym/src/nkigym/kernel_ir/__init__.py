"""KernelIR: structured kernel representation for lowering to NKI source."""

from nkigym.kernel_ir.build import build_ir
from nkigym.kernel_ir.ir import DimScope, KernelIR, Op, PhysicalBuffer
from nkigym.kernel_ir.types import DimInfo, DimRole, TensorInfo

__all__ = ["DimInfo", "DimRole", "DimScope", "KernelIR", "Op", "PhysicalBuffer", "TensorInfo", "build_ir"]
