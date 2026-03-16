"""AST-based codegen: callable -> NKIKernel -> NKI source.

Modules:
    types: NKIBlock, NKIKernel container types, normalize, render.
    analysis: Dimension analysis for tiling.
    codegen: Main codegen entry point.
"""

from nkigym.codegen.codegen import codegen
from nkigym.codegen.dce import dce
from nkigym.codegen.types import NKIBlock, NKIKernel, normalize
from nkigym.ops import NKIActivation, NKIAlloc, NKIDmaCopy, NKIMatmul, NKIOp, NKITensorCopy

__all__ = [
    "NKIActivation",
    "NKIAlloc",
    "NKIBlock",
    "NKIDmaCopy",
    "NKIKernel",
    "NKIMatmul",
    "NKIOp",
    "NKITensorCopy",
    "codegen",
    "dce",
    "normalize",
]
