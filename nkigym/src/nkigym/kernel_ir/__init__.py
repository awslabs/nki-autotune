"""KernelIR: structured kernel representation for lowering to NKI source."""

from nkigym.kernel_ir.context.build import build_initial, build_ir, build_naive_ir
from nkigym.kernel_ir.context.context import DimInfo, DimRole, KernelContext, TensorInfo
from nkigym.kernel_ir.graph.fusion_group import FusionGroup
from nkigym.kernel_ir.graph.graph import KernelGraph, rebuild_edges
from nkigym.kernel_ir.ir import KernelIR
from nkigym.kernel_ir.sampler.sampler import sample_valid_ir

__all__ = [
    "DimInfo",
    "DimRole",
    "FusionGroup",
    "KernelContext",
    "KernelGraph",
    "KernelIR",
    "TensorInfo",
    "build_initial",
    "build_ir",
    "build_naive_ir",
    "rebuild_edges",
    "sample_valid_ir",
]
