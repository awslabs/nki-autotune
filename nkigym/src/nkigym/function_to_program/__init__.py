"""Convert user functions to GymProgram IR.

Submodules:
    parse: Source-to-program and program-to-source translation.
    analysis: Tiling dimension analysis.
    tile_codegen: Tiled GymProgram generation.
"""

from nkigym.function_to_program.analysis import TilingAnalysis, analyze_tiling
from nkigym.function_to_program.parse import program_to_source, source_to_program
from nkigym.function_to_program.tile_codegen import TensorNameGenerator, tile_program

__all__ = [
    "TensorNameGenerator",
    "TilingAnalysis",
    "analyze_tiling",
    "program_to_source",
    "source_to_program",
    "tile_program",
]
