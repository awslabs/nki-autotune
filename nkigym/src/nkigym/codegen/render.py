"""Kernel renderer: inspects a math function and emits NKI kernel source."""

import ast
import inspect
import textwrap
from collections.abc import Callable

import numpy as np

from nkigym.ops.base import NKIOp, RenderContext, Tensor

_INDENT = "    "


_UNASSIGNED = (ast.Expr, ast.Return)


def _extract_op_call(stmt: ast.stmt) -> tuple[ast.Call | None, str | None]:
    """Extract the OpClass()(kwargs) call and output name from a statement.

    Returns ``(call_node, output_name)``.  *output_name* is ``None`` for
    unassigned calls (bare expression / return).
    """
    call_node: ast.Call | None = None
    output_name: str | None = None

    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        if isinstance(target, ast.Name) and isinstance(stmt.value, ast.Call):
            call_node = stmt.value
            output_name = target.id
    elif isinstance(stmt, _UNASSIGNED) and isinstance(stmt.value, ast.Call):
        call_node = stmt.value

    return call_node, output_name


def _resolve_op_class(call_node: ast.Call, func_globals: dict[str, object]) -> type[NKIOp] | None:
    """Resolve the NKIOp subclass from an ``OpClass()(...)`` call node."""
    cls: type[NKIOp] | None = None
    if isinstance(call_node.func, ast.Call) and isinstance(call_node.func.func, ast.Name):
        candidate = func_globals.get(call_node.func.func.id)
        if isinstance(candidate, type) and issubclass(candidate, NKIOp):
            cls = candidate
    return cls


def _find_ops(func: Callable[..., np.ndarray]) -> tuple[list[tuple[NKIOp, dict[str, str], str]], str]:
    """Find NKIOp subclasses and the return variable via AST inspection.

    Looks for the ``result = OpClass()(key=var, ...)`` assignment pattern,
    resolves each class name against the function's globals, and extracts
    the operand map from keyword arguments.  Unassigned op calls raise
    an error — the variable name becomes the output tensor name.

    Also extracts the ``return <name>`` variable in the same pass.

    Args:
        func: Math function using nkigym ops.

    Returns:
        ``(ops, return_name)`` where *ops* is a list of
        ``(NKIOp instance, operand_map, output_name)`` in source order.
    """
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    func_def = tree.body[0]
    assert isinstance(func_def, ast.FunctionDef)
    func_body = func_def.body
    ops: list[tuple[NKIOp, dict[str, str], str]] = []
    return_name: str | None = None
    for stmt in func_body:
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name):
            return_name = stmt.value.id

        call_node, output_name = _extract_op_call(stmt)
        if call_node is None:
            continue
        cls = _resolve_op_class(call_node, func.__globals__)
        if cls is None:
            continue
        if output_name is None:
            raise ValueError(f"Op {cls.NAME!r} result must be assigned to a variable")
        operand_map = {
            kw.arg: kw.value.id for kw in call_node.keywords if kw.arg is not None and isinstance(kw.value, ast.Name)
        }
        ops.append((cls(), operand_map, output_name))

    if return_name is None:
        raise ValueError("Math function must have a 'return <variable>' statement")
    return ops, return_name


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


def _resolve_output_shape(op: NKIOp, operand_map: dict[str, str], ctx: RenderContext) -> tuple[tuple[int, ...], str]:
    """Trace the output shape and dtype from an op's operand axes.

    Builds an axis-label-to-size mapping from the input operands,
    then resolves the output shape via ``OUTPUT_AXES``.  The dtype
    follows the first input operand.

    Args:
        op: The NKIOp instance.
        operand_map: Maps op slot name to tensor name in ctx.
        ctx: Running render context with tensors.

    Returns:
        ``(output_shape, dtype_str)`` for the op's output.
    """
    axis_sizes: dict[str, int] = {}
    first_dtype: str | None = None
    for slot, axes in op.OPERAND_AXES.items():
        tensor = ctx.tensors[operand_map[slot]]
        if first_dtype is None:
            first_dtype = tensor.dtype
        for axis_label, size in zip(axes, tensor.shape, strict=True):
            axis_sizes[axis_label] = size

    if first_dtype is None:
        raise ValueError(f"Op {op.NAME!r} has no input operands")

    if len(op.OUTPUT_AXES) != 1:
        raise ValueError(
            f"Op {op.NAME!r} has {len(op.OUTPUT_AXES)} outputs — " f"multi-output ops are not yet supported"
        )

    output_axes = next(iter(op.OUTPUT_AXES.values()))
    output_shape = tuple(axis_sizes[a] for a in output_axes)
    return output_shape, first_dtype


def render(func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]) -> str:
    """Inspect a math function and generate NKI kernel source.

    Parses the function AST to find NKIOp subclass calls,
    then calls each op's render() to emit the kernel body.
    Output HBM tensors are allocated via ``nl.ndarray`` with
    shape traced from operand axes and dtype from the first input.

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

    ops, return_name = _find_ops(func)

    for op, operand_map, output_name in ops:
        output_shape, output_dtype = _resolve_output_shape(op, operand_map, ctx)
        ctx.tensors[output_name] = Tensor(name=output_name, shape=output_shape, dtype=output_dtype, location="hbm")

    if return_name not in ctx.tensors:
        raise ValueError(f"Return variable {return_name!r} is not a known tensor")
    return_tensor = ctx.tensors[return_name]
    shape_str = ", ".join(str(s) for s in return_tensor.shape)

    lines = _emit_header(func.__name__, param_names, input_specs)
    lines.append(
        f"{_INDENT}hbm_{return_name} = nl.ndarray(({shape_str}), dtype=nl.{return_tensor.dtype},"
        f" buffer=nl.shared_hbm)"
    )

    for op, operand_map, _ in ops:
        for line in op.render(ctx, operand_map):
            lines.append(f"{_INDENT}{line}")

    lines.append(f"{_INDENT}return hbm_{return_name}")

    return "\n".join(lines)
