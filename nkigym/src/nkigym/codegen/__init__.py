"""Codegen: :class:`nkigym.ir.KernelIR` → NKI source.

:func:`render` is the top-level entry; it composes
:func:`emit_header` (prologue), :func:`emit_body` (schedule-tree
body), and :func:`emit_return` (return-tensor allocation + trailing
``return``).
"""

from nkigym.codegen.body import emit_body
from nkigym.codegen.header import emit_header, emit_return
from nkigym.codegen.render import render

__all__ = ["emit_body", "emit_header", "emit_return", "render"]
