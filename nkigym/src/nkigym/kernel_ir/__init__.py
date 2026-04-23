"""KernelIR: structured kernel representation for lowering to NKI source."""

from nkigym.kernel_ir.build import build_initial, build_ir, build_naive_ir
from nkigym.kernel_ir.fusion_group import FusionGroup
from nkigym.kernel_ir.ir import KernelIR, insert_dma_nodes, rebuild_edges
from nkigym.kernel_ir.sampler.sampler import sample_valid_ir
from nkigym.kernel_ir.types import DimInfo, DimRole, TensorInfo

__all__ = [
    "DimInfo",
    "DimRole",
    "FusionGroup",
    "KernelIR",
    "TensorInfo",
    "build_initial",
    "build_ir",
    "build_naive_ir",
    "insert_dma_nodes",
    "rebuild_edges",
    "sample_valid_ir",
]
