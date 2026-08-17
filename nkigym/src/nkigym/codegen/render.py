"""Top-level codegen entry: :class:`nkigym.ir.KernelIR` → NKI kernel source.

:func:`render` is the single public entry. It composes the prologue,
schedule-tree body, and trailing return statement.
"""

from __future__ import annotations

from nkigym.codegen.body import emit_body
from nkigym.codegen.header import emit_header
from nkigym.ir import KernelIR


def render(ir: KernelIR) -> str:
    """Render ``ir`` to NKI kernel source.

    Args:
        ir: Fully-built :class:`KernelIR` envelope.

    Returns:
        Multi-line NKI source string ending with a trailing newline.
    """
    return emit_header(ir) + emit_body(ir) + f"    return {', '.join(ir.return_names)}\n"


__all__ = ["render"]
