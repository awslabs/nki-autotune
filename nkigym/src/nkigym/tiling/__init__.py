"""Tiling pass for NKI Gym.

Analyzes NumPy functions to extract dimension information, classifies
dimensions as parallel vs reduction, and generates tiled source code
with explicit slice offsets for each tile position.
"""

from nkigym.tiling.analysis import (
    TILE_SIZE,
    DimensionAnalysis,
    DimInfo,
    TensorSliceInfo,
    analyze_dimension,
    format_analysis,
)
from nkigym.tiling.tile_codegen import TensorNameGenerator, generate_tiled_ir

__all__ = [
    "TILE_SIZE",
    "DimensionAnalysis",
    "DimInfo",
    "TensorSliceInfo",
    "TensorNameGenerator",
    "analyze_dimension",
    "format_analysis",
    "generate_tiled_ir",
]
