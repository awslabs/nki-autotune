"""Op-graph structure: FusionGroup + KernelGraph."""

from nkigym.kernel_ir.graph.fusion_group import FusionGroup
from nkigym.kernel_ir.graph.graph import KernelGraph, rebuild_edges

__all__ = ["FusionGroup", "KernelGraph", "rebuild_edges"]
