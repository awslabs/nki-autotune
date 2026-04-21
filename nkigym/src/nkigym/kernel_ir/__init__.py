"""KernelIR: structured kernel representation for lowering to NKI source."""

from nkigym.kernel_ir.dim_analysis import DimAnalysis, DimInfo, TensorInfo, analyze_dims
from nkigym.kernel_ir.ir import KernelIR, build_ir, get_tpb
from nkigym.kernel_ir.op_graph import OpGraph, build_op_graph, insert_dma_nodes

__all__ = [
    "DimAnalysis",
    "DimInfo",
    "KernelIR",
    "OpGraph",
    "TensorInfo",
    "analyze_dims",
    "build_ir",
    "build_op_graph",
    "get_tpb",
    "insert_dma_nodes",
]
