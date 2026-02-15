"""Parse nkigym callables into GymProgram IR."""

import ast
from collections.abc import Callable
from typing import Any

from nkigym.ir.tensor import TensorRef
from nkigym.ir.types import GymProgram, GymStatement
from nkigym.ops.base import GymOp
from nkigym.utils.source import callable_to_source


def _full_slices(shape: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Build full-range slices from a shape.

    Args:
        shape: Tensor shape tuple.

    Returns:
        Per-axis (0, size) bounds.
    """
    return tuple((0, s) for s in shape)


def func_to_program(func: Callable, input_shapes: dict[str, tuple[int, ...]], output_dtype: type) -> GymProgram:
    """Parse a nkigym callable into a specialized GymProgram.

    Parses the function AST, specializes with the given input shapes,
    infers output shapes through the op chain, and produces a fully-typed
    program where every tensor reference carries shape and slice info.

    Args:
        func: A callable using nkigym ops.
        input_shapes: Mapping from parameter names to shape tuples.
        output_dtype: Numpy dtype type for output allocation.

    Returns:
        Specialized GymProgram with TensorRef on all tensor references.

    Raises:
        ValueError: If the function structure is not recognized.
    """
    source = callable_to_source(func)
    tree = ast.parse(source)
    func_def = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break
    if func_def is None:
        raise ValueError("Expected a function definition")

    name = func_def.name
    params = tuple(arg.arg for arg in func_def.args.args)

    stmts: list[GymStatement] = []
    return_var = ""

    for node in func_def.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue

        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                call = node.value
                if _is_nkigym_call(call):
                    stmts.extend(_parse_call(call, target.id))
                    continue

        if isinstance(node, ast.Return):
            if isinstance(node.value, ast.Name):
                return_var = node.value.id
                continue
            if isinstance(node.value, ast.Call) and _is_nkigym_call(node.value):
                stmts.extend(_parse_call(node.value, "_return"))
                return_var = "_return"
                continue

        raise ValueError(f"Unsupported statement: {ast.dump(node)}")

    if not return_var:
        raise ValueError("Function must have a return statement")

    specialized = _specialize(stmts, params, input_shapes)

    return GymProgram(
        name=name,
        params=params,
        input_shapes=tuple((p, input_shapes[p]) for p in params),
        stmts=tuple(specialized),
        return_var=return_var,
        output_dtype=output_dtype,
    )


def _specialize(
    stmts: list[GymStatement], params: tuple[str, ...], input_shapes: dict[str, tuple[int, ...]]
) -> list[GymStatement]:
    """Specialize statements with concrete shapes.

    Walks the statements, infers shapes through ``op.output_shape()``,
    and replaces tensor kwargs values and outputs with TensorRef carrying
    full-range slices.

    Args:
        stmts: Parsed statements with string kwargs and placeholder outputs.
        params: Function parameter names.
        input_shapes: Mapping from parameter names to shape tuples.

    Returns:
        Specialized statements with TensorRef on all tensor references.
    """
    var_shapes: dict[str, tuple[int, ...]] = {}
    for param_name in params:
        var_shapes[param_name] = input_shapes[param_name]

    result: list[GymStatement] = []
    for stmt in stmts:
        op_cls = GymOp.get(stmt.op)
        n_inputs = len(op_cls.inputs)

        tensor_kwargs = stmt.kwargs[:n_inputs]
        config_kwargs = stmt.kwargs[n_inputs:]

        new_kwargs: list[tuple[str, Any]] = []
        input_shapes: list[tuple[int, ...]] = []
        for operand_name, var_name in tensor_kwargs:
            shape = var_shapes[var_name]
            input_shapes.append(shape)
            new_kwargs.append((operand_name, TensorRef(var_name, shape, _full_slices(shape))))

        for key, value in config_kwargs:
            new_kwargs.append((key, value))

        out_shape = op_cls().output_shape(tuple(input_shapes))
        out_name = stmt.output.name
        var_shapes[out_name] = out_shape

        result.append(
            GymStatement(
                op=stmt.op, kwargs=tuple(new_kwargs), output=TensorRef(out_name, out_shape, _full_slices(out_shape))
            )
        )

    return result


def _flatten_call(call: ast.Call, output: str, stmts: list[GymStatement], counter: list[int]) -> None:
    """Flatten a potentially nested ``nkigym.<op>(...)`` call into GymStatements.

    Recursively processes nested nkigym calls, introducing intermediate
    variables for inner calls before emitting the outer call.

    Args:
        call: AST Call node for a ``nkigym.<op>(...)`` call.
        output: Variable name to assign the result to.
        stmts: Accumulator list for emitted statements.
        counter: Single-element list used as a mutable counter for
            generating unique intermediate variable names.

    Raises:
        ValueError: If the op is not registered or arg count mismatches.
    """
    op_name = call.func.attr
    op_cls = GymOp.get(op_name)
    operand_names = tuple(t.name for t in op_cls.inputs)

    resolved_args: list[str] = []
    for arg in call.args:
        if isinstance(arg, ast.Call) and _is_nkigym_call(arg):
            tmp_name = f"_nested_{counter[0]}"
            counter[0] += 1
            _flatten_call(arg, tmp_name, stmts, counter)
            resolved_args.append(tmp_name)
        else:
            resolved_args.append(_arg_name(arg))

    if len(resolved_args) != len(operand_names):
        raise ValueError(
            f"Op {op_name!r} expects {len(operand_names)} positional args "
            f"({', '.join(operand_names)}), got {len(resolved_args)}"
        )

    kwargs: list[tuple[str, str]] = []
    for operand, var in zip(operand_names, resolved_args):
        kwargs.append((operand, var))

    for kw in call.keywords:
        kwargs.append((kw.arg, _expr_to_str(kw.value)))

    stmts.append(GymStatement(op=op_name, kwargs=tuple(kwargs), output=TensorRef(output, (), ())))


def _parse_call(call: ast.Call, output: str) -> list[GymStatement]:
    """Parse a ``nkigym.<op>(...)`` call into GymStatements.

    Handles nested nkigym calls by flattening them into a sequence of
    statements with intermediate variables.

    Args:
        call: AST Call node for a ``nkigym.<op>(...)`` call.
        output: Variable name to assign the result to.

    Returns:
        List of GymStatements (one for simple calls, multiple for nested).

    Raises:
        ValueError: If the op is not registered or arg count mismatches.
    """
    stmts: list[GymStatement] = []
    counter: list[int] = [0]
    _flatten_call(call, output, stmts, counter)
    return stmts


def _is_nkigym_call(call: ast.Call) -> bool:
    """Check if a call is to ``nkigym.<op>``.

    Args:
        call: AST Call node.

    Returns:
        True if the call is to ``nkigym.<attr>``.
    """
    return (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "nkigym"
    )


def _arg_name(node: ast.expr) -> str:
    """Extract variable name from an AST expression.

    Args:
        node: AST expression node (expected ``ast.Name``).

    Returns:
        Variable name string.

    Raises:
        ValueError: If the expression is not a simple Name.
    """
    if isinstance(node, ast.Name):
        return node.id
    raise ValueError(f"Expected a variable name, got {ast.dump(node)}")


def _expr_to_str(node: ast.expr) -> str:
    """Convert an AST expression to its source string.

    Handles dotted names (e.g., ``np.tanh``), simple names, and
    numeric/string constants.

    Args:
        node: AST expression node.

    Returns:
        Source-level string representation.

    Raises:
        ValueError: If the expression type is not supported.
    """
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return repr(node.value)
    raise ValueError(f"Unsupported kwarg expression: {ast.dump(node)}")
