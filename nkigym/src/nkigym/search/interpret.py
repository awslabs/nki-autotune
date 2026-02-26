"""Direct IR interpreter for GymProgram verification.

Executes a GymProgram by walking its statements and dispatching
to GymOp.simulate(), avoiding source generation, exec, and re-import.
"""

from typing import Any

import numpy as np

from nkigym.ir import GymProgram
from nkigym.ir.tensor import TensorRef
from nkigym.ops.base import GymOp

_OP_CACHE: dict[str, GymOp] = {}


def _get_op(op_name: str) -> GymOp:
    """Get a cached GymOp instance by name.

    Args:
        op_name: The op_name to look up.

    Returns:
        A cached GymOp instance.
    """
    cached = _OP_CACHE.get(op_name)
    if cached is None:
        cached = GymOp.get(op_name)()
        _OP_CACHE[op_name] = cached
    return cached


def _to_slice_tuple(slices: tuple[tuple[int, int], ...]) -> tuple[slice, ...]:
    """Convert (start, stop) pairs to a tuple of slice objects.

    Args:
        slices: Per-axis (start, stop) bounds.

    Returns:
        Tuple of slice objects for numpy indexing.
    """
    return tuple(slice(s, e) for s, e in slices)


def _interpret_compute(stmt: "GymStatement", env: dict[str, np.ndarray]) -> None:
    """Interpret a single compute statement via GymOp.simulate().

    Args:
        stmt: A compute GymStatement (nc_matmul, tensor_tensor, etc.).
        env: Mutable variable environment mapping names to arrays.
    """
    op_instance = _get_op(stmt.op)
    args: list[np.ndarray] = []
    kwargs: dict[str, object] = {}
    for key, value in stmt.kwargs:
        if not isinstance(value, TensorRef):
            continue
        arr = env[value.name]
        if value.slices:
            arr = arr[_to_slice_tuple(value.slices)]
        if key == "acc":
            kwargs["acc"] = arr
        else:
            args.append(arr)
    env[stmt.output.name] = op_instance.simulate(*args, **kwargs)


def _interpret_stmt(stmt: "GymStatement", env: dict[str, np.ndarray]) -> None:
    """Interpret one IR statement, updating the variable environment.

    Args:
        stmt: A single GymStatement.
        env: Mutable variable environment mapping names to arrays.
    """
    if stmt.op == "np_empty":
        env[stmt.output.name] = np.empty(stmt.output.shape, dtype=np.float32)
    elif stmt.op == "np_slice":
        src_ref = stmt.kwargs[0][1]
        env[stmt.output.name] = env[src_ref.name][_to_slice_tuple(src_ref.slices)]
    elif stmt.op == "np_store":
        src_ref, dst_ref = stmt.kwargs[0][1], stmt.kwargs[1][1]
        src_arr = env[src_ref.name][_to_slice_tuple(src_ref.slices)] if src_ref.slices else env[src_ref.name]
        env[dst_ref.name][_to_slice_tuple(dst_ref.slices)] = src_arr
    else:
        _interpret_compute(stmt, env)


def interpret_program(program: GymProgram, kernel_kwargs: dict[str, Any]) -> np.ndarray:
    """Execute a GymProgram by interpreting IR statements directly.

    Avoids the overhead of source generation, exec, and re-import.

    Args:
        program: The GymProgram to interpret.
        kernel_kwargs: Input arrays keyed by parameter name.

    Returns:
        The output array (value of the return variable).
    """
    env: dict[str, np.ndarray] = dict(kernel_kwargs)
    for stmt in program.stmts:
        _interpret_stmt(stmt, env)
    return env[program.return_var]
