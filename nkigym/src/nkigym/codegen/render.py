"""Top-level codegen entry: :class:`nkigym.ir.KernelIR` → NKI kernel source.

:func:`render` is the single public entry. It composes the rendered
source from three emitters: :func:`nkigym.codegen.header.emit_header`
for the prologue, :func:`nkigym.codegen.body.emit_body` for the
schedule-tree body, and :func:`nkigym.codegen.header.emit_return` for
the return-tensor allocation and trailing ``return`` statement.
"""

from __future__ import annotations

from nkigym.codegen.body import emit_body
from nkigym.codegen.header import emit_header, emit_return
from nkigym.ir import KernelIR


def render(ir: KernelIR) -> str:
    """Render ``ir`` to NKI kernel source.

    Pipeline:

    1. :func:`emit_header` — imports + ``@nki.jit`` signature + per-param
       shape assertions.
    2. :func:`emit_body` — body emission for the schedule tree.
    3. :func:`emit_return` — HBM allocation for the return tensor and
       trailing ``return`` statement.

    Args:
        ir: Fully-built :class:`KernelIR` envelope.

    Returns:
        Multi-line NKI source string ending with a trailing newline.
    """
    header = emit_header(ir)
    body = emit_body(ir)
    ret = emit_return(ir)
    return header + body + ret


__all__ = ["render"]
