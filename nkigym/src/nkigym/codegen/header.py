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
    return (
        "import nki\nimport nki.isa as nisa\nimport nki.language as nl\n"
        "from nki.isa.constants import oob_mode\n\n\n@nki.jit\n"
        f"def nki_{ir.func_name}({', '.join(ir.param_names)}):\n"
        + "".join(f"    assert {name}.shape == {tuple(ir.buffer(name).shape)}\n" for name in ir.param_names)
    )


__all__ = ["emit_header"]
