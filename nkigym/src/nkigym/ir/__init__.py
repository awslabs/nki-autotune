"""Dim unification analysis + canonical schedule tree for an ``f_nkigym`` callable."""

from nkigym.ir.dimension_analysis import DimensionAnalysis, OpAxes, TensorDims, analyze_dimensions
from nkigym.ir.ir import KernelIR, build_initial_ir
from nkigym.ir.tree import ForNode, ISANode, KernelTree, NodeData, RootNode, TensorizeLoop, build_initial_tree

__all__ = [
    "DimensionAnalysis",
    "ForNode",
    "ISANode",
    "KernelIR",
    "KernelTree",
    "NodeData",
    "OpAxes",
    "RootNode",
    "TensorDims",
    "TensorizeLoop",
    "analyze_dimensions",
    "build_initial_ir",
    "build_initial_tree",
]
