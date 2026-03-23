"""AST parsing for user workload functions.

Extracts ``_OpCall`` entries from a user function that uses
``nkigym.<op>(...)`` calls for the schedule-based pipeline.
"""

import ast

import numpy as np

from nkigym.codegen.analysis import _OpCall
from nkigym.ops.base import NKIOp


def find_func_def(source: str) -> ast.FunctionDef:
    """Find the first FunctionDef in parsed source.

    Args:
        source: Python source code string.

    Returns:
        The AST FunctionDef node.
    """
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    raise ValueError("Expected a function definition")


def _is_nkigym_call(call: ast.Call) -> bool:
    """Check if a call is to ``nkigym.<op>``.

    Args:
        call: AST Call node.

    Returns:
        True if the call targets ``nkigym.<attr>``.
    """
    func = call.func
    return isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "nkigym"


def _eval_expr(node: ast.expr) -> object:
    """Evaluate an AST expression to a Python object.

    Resolves ``np.X`` attribute accesses and literal constants.

    Args:
        node: AST expression node.

    Returns:
        The resolved Python object.
    """
    result = None
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "np":
            result = getattr(np, node.attr)
    elif isinstance(node, ast.Constant):
        result = node.value
    if result is None:
        raise ValueError(f"Unsupported kwarg expression: {ast.dump(node)}")
    return result


def _arg_name(node: ast.expr) -> str:
    """Extract variable name from an AST Name node.

    Args:
        node: AST expression node.

    Returns:
        Variable name string.
    """
    if not isinstance(node, ast.Name):
        raise ValueError(f"Expected a variable name, got {ast.dump(node)}")
    return node.id


def parse_body(func_def: ast.FunctionDef) -> list[_OpCall]:
    """Parse function body into a list of _OpCall.

    Args:
        func_def: The parsed AST FunctionDef node.

    Returns:
        List of parsed operation calls.
    """
    op_calls: list[_OpCall] = []
    counter: list[int] = [0]
    for node in func_def.body:
        if not _try_parse_node(node, op_calls, counter):
            raise ValueError(f"Unsupported statement: {ast.dump(node)}")
    return op_calls


def _try_parse_node(node: ast.stmt, op_calls: list[_OpCall], counter: list[int]) -> bool:
    """Try to parse a single AST statement node.

    Args:
        node: AST statement node.
        op_calls: Accumulator for parsed op calls.
        counter: Mutable counter for intermediate names.

    Returns:
        True if the node was successfully parsed.
    """
    result = False
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        result = True
    elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        result = True
    elif isinstance(node, ast.Return):
        result = _try_parse_return(node, op_calls, counter)
    elif isinstance(node, ast.Assign) and len(node.targets) == 1:
        result = _try_parse_assign(node, op_calls, counter)
    return result


def _try_parse_assign(node: ast.Assign, op_calls: list[_OpCall], counter: list[int]) -> bool:
    """Try to parse an assignment statement as an nkigym op call.

    Args:
        node: AST Assign node.
        op_calls: Accumulator for parsed op calls.
        counter: Mutable counter for intermediate names.

    Returns:
        True if the assignment was a recognized nkigym call.
    """
    target = node.targets[0]
    result = False
    if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
        if _is_nkigym_call(node.value):
            _flatten_call(node.value, target.id, op_calls, counter)
            result = True
    return result


def _try_parse_return(node: ast.Return, op_calls: list[_OpCall], counter: list[int]) -> bool:
    """Try to parse a return statement.

    Args:
        node: AST Return node.
        op_calls: Accumulator for parsed op calls.
        counter: Mutable counter for intermediate names.

    Returns:
        True if the return was successfully parsed.
    """
    result = False
    if isinstance(node.value, ast.Name):
        result = True
    elif isinstance(node.value, ast.Call) and _is_nkigym_call(node.value):
        _flatten_call(node.value, "_return", op_calls, counter)
        result = True
    return result


def _flatten_call(call: ast.Call, output: str, op_calls: list[_OpCall], counter: list[int]) -> None:
    """Flatten a nkigym call (possibly nested) into _OpCall entries.

    Args:
        call: AST Call node for a ``nkigym.<op>(...)`` call.
        output: Variable name to assign the result to.
        op_calls: Accumulator list for emitted op calls.
        counter: Mutable counter for intermediate variable names.
    """
    assert isinstance(call.func, ast.Attribute)
    op_name = call.func.attr
    registry = NKIOp.all_ops()
    if op_name not in registry:
        raise ValueError(f"Unknown op: {op_name!r}")
    stmt_type = registry[op_name]
    resolved_args = _resolve_call_args(call, op_calls, counter)
    config_kwargs: list[tuple[str, object]] = []
    for kw in call.keywords:
        assert kw.arg is not None
        config_kwargs.append((kw.arg, _eval_expr(kw.value)))
    op_calls.append(
        _OpCall(
            stmt_type=stmt_type, input_vars=tuple(resolved_args), config_kwargs=tuple(config_kwargs), output_var=output
        )
    )


def _resolve_call_args(call: ast.Call, op_calls: list[_OpCall], counter: list[int]) -> list[str]:
    """Resolve positional arguments of a nkigym call.

    Args:
        call: AST Call node.
        op_calls: Accumulator for nested op calls.
        counter: Mutable counter for intermediate names.

    Returns:
        List of resolved variable names.
    """
    resolved: list[str] = []
    for arg in call.args:
        if isinstance(arg, ast.Call) and _is_nkigym_call(arg):
            tmp_name = f"_nested_{counter[0]}"
            counter[0] += 1
            _flatten_call(arg, tmp_name, op_calls, counter)
            resolved.append(tmp_name)
        else:
            resolved.append(_arg_name(arg))
    return resolved
