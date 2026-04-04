"""AST parsing for user workload functions.

Extracts ``_OpCall`` entries from a user function that uses
``nkigym.<op>(...)`` calls for the schedule-based pipeline.
"""

import ast
import operator

import numpy as np

from nkigym.codegen.analysis import _OpCall
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_1d import NKIActivation1D
from nkigym.ops.base import NKIOp, _get_output_axes_tuple

_BINOP_FNS: dict[type, object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


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


def _eval_binop(node: ast.BinOp) -> object:
    """Evaluate a binary operation on constant operands.

    Args:
        node: AST BinOp node.

    Returns:
        Result of the binary operation.
    """
    left = _eval_expr(node.left)
    right = _eval_expr(node.right)
    op_fn = _BINOP_FNS.get(type(node.op))
    if op_fn is None:
        raise ValueError(f"Unsupported binary op: {ast.dump(node)}")
    return op_fn(left, right)


def _eval_np_attr(node: ast.Attribute) -> object:
    """Resolve ``np.X`` attribute access to the numpy object."""
    if isinstance(node.value, ast.Name) and node.value.id == "np":
        return getattr(np, node.attr)
    raise ValueError(f"Unsupported attribute: {ast.dump(node)}")


def _eval_unary(node: ast.UnaryOp) -> object:
    """Evaluate unary negation."""
    if isinstance(node.op, ast.USub):
        return -_eval_expr(node.operand)
    raise ValueError(f"Unsupported unary op: {ast.dump(node)}")


def _eval_list(node: ast.List) -> list[object]:
    """Evaluate an AST List node to a Python list.

    Args:
        node: AST List node.

    Returns:
        List of evaluated elements.
    """
    return [_eval_expr(elt) for elt in node.elts]


_EVAL_DISPATCH: dict[type, object] = {
    ast.Attribute: _eval_np_attr,
    ast.Constant: lambda n: n.value,
    ast.BinOp: _eval_binop,
    ast.UnaryOp: _eval_unary,
    ast.Name: lambda n: n.id,
    ast.List: _eval_list,
}


def _eval_expr(node: ast.expr) -> object:
    """Evaluate an AST expression to a Python object.

    Resolves ``np.X`` attribute accesses, literal constants,
    binary operations, unary negation, and variable names.

    Args:
        node: AST expression node.

    Returns:
        The resolved Python object.
    """
    handler = _EVAL_DISPATCH.get(type(node))
    if handler is None:
        raise ValueError(f"Unsupported kwarg expression: {ast.dump(node)}")
    return handler(node)


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


def _maybe_reclassify_activation(op: _OpCall, output_axes_map: dict[str, tuple[str, ...]]) -> _OpCall:
    """Reclassify NKIActivation to NKIActivation1D if input is 1D.

    Args:
        op: Parsed op call to check.
        output_axes_map: Maps variable name to output axes of its producer op.

    Returns:
        Original or reclassified op call.
    """
    is_1d = (
        op.stmt_type is NKIActivation
        and op.input_vars[0] in output_axes_map
        and len(output_axes_map[op.input_vars[0]]) == 1
    )
    return op._replace(stmt_type=NKIActivation1D) if is_1d else op


def _resolve_op_variants(op_calls: list[_OpCall]) -> list[_OpCall]:
    """Post-parse pass to reclassify ops based on producer output shapes.

    Traces the SSA chain to determine operand dimensionality and
    reclassifies NKIActivation to NKIActivation1D when input is 1D.

    Args:
        op_calls: Parsed op calls from the function body.

    Returns:
        Op calls with reclassified types where appropriate.
    """
    output_axes_map: dict[str, tuple[str, ...]] = {}
    result: list[_OpCall] = []
    for op in op_calls:
        resolved = _maybe_reclassify_activation(op, output_axes_map)
        output_axes_map[resolved.output_var] = _get_output_axes_tuple(resolved.stmt_type)
        result.append(resolved)
    return result


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
    return _resolve_op_variants(op_calls)


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


def _disambiguate_op(op_name: str, call: ast.Call) -> str:
    """Disambiguate user function name to internal op registry key.

    - ``activation`` with ``reduce_op`` kwarg → ``activation_reduce``
    - ``tensor_scalar`` with < 2 positional args → ``tensor_scalar_const``

    Args:
        op_name: User-facing function name from AST.
        call: AST Call node with keyword arguments.

    Returns:
        Internal op registry key.
    """
    kwarg_names = {kw.arg for kw in call.keywords}
    result = op_name
    if op_name == "transpose":
        result = "nc_transpose"
    elif op_name == "activation" and "reduce_op" in kwarg_names:
        result = "activation_reduce"
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
    op_name = _disambiguate_op(call.func.attr, call)
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
