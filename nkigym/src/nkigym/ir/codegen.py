"""GymProgram codegen: source rendering and compilation to callable."""

from collections.abc import Callable

import numpy as np

from nkigym.ir.tensor import TensorRef
from nkigym.ir.types import GymProgram, GymStatement
from nkigym.utils.source import source_to_callable


def _subscript(ref: TensorRef) -> str:
    """Render a TensorRef as a subscripted variable.

    Scalar tensors (empty slices) are rendered as plain variable names.

    Args:
        ref: Tensor reference with name and slices.

    Returns:
        String like ``tensor_0[0:128, 0:128]`` or ``scalar`` for scalars.
    """
    if not ref.slices:
        return ref.name
    slices = ", ".join(f"{s}:{e}" for s, e in ref.slices)
    return f"{ref.name}[{slices}]"


def _render_np_empty(stmt: GymStatement) -> str:
    """Render an np_empty statement.

    Args:
        stmt: The np_empty GymStatement.

    Returns:
        Source line like ``output = np.empty((128, 128), dtype=np.float32)``.
    """
    dtype = ""
    for key, value in stmt.kwargs:
        if key == "dtype":
            dtype = value
    shape_str = ", ".join(str(s) for s in stmt.output.shape)
    return f"    {stmt.output.name} = np.empty(({shape_str}), dtype={dtype})"


def _render_np_slice(stmt: GymStatement) -> str:
    """Render an np_slice statement.

    Args:
        stmt: The np_slice GymStatement.

    Returns:
        Source line like ``tensor_0 = a[0:128, 0:128]``.
    """
    src = None
    for key, value in stmt.kwargs:
        if key == "src":
            src = value
    slices = ", ".join(f"{s}:{e}" for s, e in src.slices)
    return f"    {stmt.output.name} = {src.name}[{slices}]"


def _render_np_store(stmt: GymStatement) -> str:
    """Render an np_store statement.

    Args:
        stmt: The np_store GymStatement.

    Returns:
        Source line like ``output[0:128, 0:128] = tensor_2[0:128, 0:128]``.
    """
    src = None
    dst = None
    for key, value in stmt.kwargs:
        if key == "src":
            src = value
        elif key == "dst":
            dst = value
    return f"    {_subscript(dst)} = {_subscript(src)}"


def _render_compute(stmt: GymStatement) -> str:
    """Render a compute GymStatement.

    Handles both first-write (``var = nkigym.op(...)``) and accumulation
    (``var[subscripts] += nkigym.op(...)``).

    Args:
        stmt: The compute GymStatement.

    Returns:
        Source line like ``tensor_2 = nkigym.nc_matmul(...)``.
    """
    acc_ref = None
    args: list[str] = []
    for key, value in stmt.kwargs:
        if key == "acc":
            acc_ref = value
        elif isinstance(value, TensorRef):
            args.append(_subscript(value))
        else:
            args.append(f"{key}={value}")

    args_str = ", ".join(args)
    call = f"nkigym.{stmt.op}({args_str})"

    if acc_ref is not None:
        return f"    {_subscript(acc_ref)} += {call}"
    return f"    {stmt.output.name} = {call}"


def program_to_source(program: GymProgram) -> str:
    """Render a tiled GymProgram as Python source code.

    Each statement is rendered directly from its TensorRef with no
    cross-statement shape tracking.

    Args:
        program: A tiled GymProgram.

    Returns:
        Complete Python source code string with imports.
    """
    lines: list[str] = ["import numpy as np", "import nkigym"]

    params_str = ", ".join(program.params)
    lines.append(f"def {program.name}({params_str}):")

    for stmt in program.stmts:
        if stmt.op == "np_empty":
            lines.append(_render_np_empty(stmt))
        elif stmt.op == "np_slice":
            lines.append(_render_np_slice(stmt))
        elif stmt.op == "np_store":
            lines.append(_render_np_store(stmt))
        else:
            lines.append(_render_compute(stmt))

    lines.append(f"    return {program.return_var}")
    lines.append("")

    return "\n".join(lines) + "\n"


def program_to_func(program: GymProgram) -> Callable[..., np.ndarray]:
    """Compile a GymProgram into an executable Python function.

    Generates source via ``program_to_source`` and compiles it with
    ``source_to_callable``. The returned function accepts the program's
    parameters as positional or keyword arguments.

    Args:
        program: A GymProgram to compile.

    Returns:
        A callable that executes the program and returns a numpy array.
    """
    source = program_to_source(program)
    return source_to_callable(source, program.name)
