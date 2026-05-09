"""Render orchestrator — runs lowering passes in order.

The lowering pipeline is split across
:mod:`nkigym.codegen.lowering`:

1. :mod:`place_buffers` — buffer shapes + slot counts from the LCA walk.
2. :mod:`inject_multi_buffer` — slot-index expressions for the buffer
   degrees set by the ``MultiBuffer`` rewrite.
3. :mod:`inject_software_pipeline` — prologue/body/epilogue emission
   for LoopNodes with ``pipeline_depth > 1``.
4. :mod:`emit_ops` — per-op_cls ISA call-site emission.
5. :mod:`emit_source` — top-level forest walker; produces the NKI
   source string.

The :func:`render` entry point composes these passes through
:func:`nkigym.codegen.lowering.emit_source.emit_source`; the body
emitters and pipelined walker are wired up via cross-module imports
inside ``emit_source``.
"""

from nkigym.codegen.ir import KernelModule
from nkigym.codegen.lowering.emit_source import emit_source, render_annotated

__all__ = ["render", "render_annotated"]


def render(module: KernelModule) -> str:
    """Render the :class:`KernelModule` to NKI source."""
    return emit_source(module)
