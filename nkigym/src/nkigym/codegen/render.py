"""Kernel renderer: inspects a math function and emits NKI kernel source."""

import ast
import inspect
import textwrap
from collections.abc import Callable

import numpy as np

from nkigym.ops.base import NKIOp, RenderContext, Tensor

_INDENT = "    "


def _find_ops(func: Callable[..., np.ndarray]) -> list[tuple[NKIOp, dict[str, str]]]:
    """Find NKIOp subclasses used in a function via AST inspection.

    Looks for the ``OpClass()(key=var, ...)`` call pattern, resolves
    each class name against the function's globals, and extracts the
    operand map from keyword arguments.

    Args:
        func: Math function using nkigym ops.

    Returns:
        List of (NKIOp instance, operand_map) in source order.
    """
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    func_body = tree.body[0].body
    ops: list[tuple[NKIOp, dict[str, str]]] = []
    for stmt in func_body:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Call):
                continue
            if not isinstance(node.func.func, ast.Name):
                continue
            cls = func.__globals__.get(node.func.func.id)
            if cls is not None and issubclass(cls, NKIOp):
                operand_map = {kw.arg: kw.value.id for kw in node.keywords if isinstance(kw.value, ast.Name)}
                ops.append((cls(), operand_map))
    return ops


def _emit_header(
    func_name: str, param_names: list[str], input_specs: dict[str, tuple[tuple[int, ...], str]]
) -> list[str]:
    """Emit imports, decorator, signature, and input checks.

    Args:
        func_name: Kernel function name.
        param_names: Parameter names in order.
        input_specs: Maps parameter name to ``(shape, dtype_str)``.

    Returns:
        List of source lines.
    """
    lines = [
        "import nki",
        "import nki.language as nl",
        "import nki.isa as nisa",
        "from nki.backends.mlir_tracer.tensor import Tensor",
        "",
        "",
        "@nki.jit",
        f"def {func_name}_kernel({', '.join(f'{n}: Tensor' for n in param_names)}):",
    ]
    for name in param_names:
        shape, dtype_str = input_specs[name]
        shape_str = ", ".join(str(s) for s in shape)
        lines.append(f"{_INDENT}assert {name}.shape == ({shape_str})")
        lines.append(f"{_INDENT}assert {name}.dtype == nl.{dtype_str}")
    return lines


def render(func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]) -> str:
    """Inspect a math function and generate NKI kernel source.

    Parses the function AST to find NKIOp subclass calls,
    then calls each op's render() to emit the kernel body.

    Args:
        func: Math function using nkigym ops.
        input_specs: Maps each parameter name to ``(shape, dtype_str)``
            (e.g. ``{"lhs_T": ((2048, 2048), "bfloat16")}``).

    Returns:
        Complete NKI kernel source string.
    """
    param_names = list(inspect.signature(func).parameters.keys())
    for name in param_names:
        if name not in input_specs:
            raise ValueError(f"Missing input_spec for parameter: {name!r}")

    ctx = RenderContext()
    for name in param_names:
        shape, dtype_str = input_specs[name]
        ctx.tensors[name] = Tensor(name=name, shape=shape, dtype=dtype_str, location="hbm")

    ops = _find_ops(func)
    lines = _emit_header(func.__name__, param_names, input_specs)
    for op, operand_map in ops:
        for line in op.render(ctx, operand_map):
            lines.append(f"{_INDENT}{line}")

    return "\n".join(lines)
