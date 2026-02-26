"""Lower GymProgram IR to NKI kernel source code.

Thin orchestrator that walks each GymStatement and dispatches to
the per-op ``to_nki`` method defined on GymOp subclasses.
"""

import numpy as np

from nkigym.codegen.context import _LoweringContext
from nkigym.ir.types import GymProgram
from nkigym.ops.base import GymOp


def lower_to_nki(program: GymProgram) -> str:
    """Lower a GymProgram to NKI kernel source code.

    Translates each IR statement into NKI instructions by dispatching
    to the ``to_nki`` method on the corresponding GymOp. Tracks buffer
    locations for PSUM staging, and wraps the result with imports,
    decorator, and function signature.

    Args:
        program: The GymProgram to lower.

    Returns:
        Complete NKI kernel source code string.

    Raises:
        KeyError: If an unknown op is encountered.
    """
    dtype_str = f"nl.{np.dtype(program.output_dtype).name}"
    ctx = _LoweringContext(params=program.params, dtype=dtype_str)
    for param_name in program.params:
        ctx.buffers[param_name] = "HBM"

    body_lines: list[str] = []
    for stmt in program.stmts:
        op = GymOp.get(stmt.op)()
        body_lines.extend(op.to_nki(stmt, ctx))

    body_lines.append(f"return {program.return_var}")

    params_str = ", ".join(program.params)
    header_lines = [
        "import nki",
        "import nki.language as nl",
        "import nki.isa as nisa",
        "",
        "",
        "@nki.jit",
        f"def {program.name}({params_str}):",
    ]

    indented_body = ["    " + line for line in body_lines]
    all_lines = header_lines + indented_body
    return "\n".join(all_lines) + "\n"
