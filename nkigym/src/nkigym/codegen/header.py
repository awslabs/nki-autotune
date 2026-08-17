"""Kernel-prologue codegen.

:func:`emit_header` produces the fixed scaffolding above the kernel
body — imports, the ``@nki.jit`` decorator, the ``def`` line, and one
``assert <param>.shape == (...)`` line per kernel parameter.
"""

from __future__ import annotations

from nkigym.ir import KernelIR


def emit_header(ir: KernelIR) -> str:
    """Render imports + ``@nki.jit`` signature + per-param shape assertions.

    Args:
        ir: Fully-built :class:`KernelIR` envelope. The renderer reads
            ``func_name``, ``param_names``, and ``tensors`` (for
            parameter shapes).

    Returns:
        Multi-line source string ending with a trailing newline. The
        last line is the deepest shape assertion, so the body emitter
        can append directly.
    """
    lines: list[str] = []
    _emit_imports(lines)
    lines.append("")
    lines.append("")
    _emit_signature(lines, ir)
    _emit_shape_assertions(lines, ir)
    return "\n".join(lines) + "\n"


def _emit_imports(lines: list[str]) -> None:
    """Append the standard NKI import block."""
    lines.append("import nki")
    lines.append("import nki.isa as nisa")
    lines.append("import nki.language as nl")
    lines.append("from nki.isa.constants import oob_mode")


def _emit_signature(lines: list[str], ir: KernelIR) -> None:
    """Append ``@nki.jit`` and ``def <func_name>(<params>):`` in signature order."""
    lines.append("@nki.jit")
    params = ", ".join(ir.param_names)
    lines.append(f"def nki_{ir.func_name}({params}):")


def _emit_shape_assertions(lines: list[str], ir: KernelIR) -> None:
    """Append ``assert <param>.shape == (...)`` for every kernel parameter."""
    for name in ir.param_names:
        buf = ir.buffer(name)
        shape_tuple = str(tuple(buf.shape))
        lines.append(f"    assert {name}.shape == {shape_tuple}")


__all__ = ["emit_header"]
