"""Render orchestrator — runs lowering passes in order.

Stages (Tasks 22-25 extract each as a separate pass; until then the
work is fully consolidated inside :mod:`nkigym.codegen.lowering.emit_source`):

1. LowerDecomposedReduction — canonicalize post-fission reducer trees.
2. InjectMultiBuffer — Tensor.buffer_degree -> allocation shapes + slot exprs.
3. InjectSoftwarePipeline — LoopNode.pipeline_depth -> prologue/body/epilogue.
4. LowerPhases — (op_cls, phase) -> ISA call-site snippet.
5. PlaceBuffers — LCA walk -> buffer shape + position + allocator addresses.
6. EmitSource — textual NKI Python emission.
"""

from nkigym.codegen.ir import KernelModule
from nkigym.codegen.lowering.emit_source import emit_source, render_annotated

__all__ = ["render", "render_annotated"]


def render(module: KernelModule) -> str:
    """Render the :class:`KernelModule` to NKI source.

    Runs all six passes; current implementation collapses them into
    :func:`emit_source` until Tasks 22-25 extract each.
    """
    return emit_source(module)
