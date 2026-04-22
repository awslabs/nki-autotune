"""Kernel-wide globals + per-op resolved data."""

from nkigym.kernel_ir.context.build import build_initial
from nkigym.kernel_ir.context.context import DimInfo, DimRole, KernelContext, TensorInfo
from nkigym.kernel_ir.context.parse import find_ops
from nkigym.kernel_ir.context.trace import trace_scalar_kwargs

__all__ = ["DimInfo", "DimRole", "KernelContext", "TensorInfo", "build_initial", "find_ops", "trace_scalar_kwargs"]
