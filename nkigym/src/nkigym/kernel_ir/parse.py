"""AST-based discovery of NKIOp calls in math functions."""

import ast
import inspect
import textwrap
from collections.abc import Callable
from typing import Any

import numpy as np

from nkigym.ops.base import NKIOp

_ParsedOp = tuple[type[NKIOp], dict[str, str], dict[str, Any], list[str]]


def _resolve_op_class(node: ast.expr, func_globals: dict[str, object]) -> type[NKIOp] | None:
    """Resolve NKIOp subclass from an ``OpClass()(...)`` call node."""
    is_double_call = (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Call) and isinstance(node.func.func, ast.Name)
    )
    result: type[NKIOp] | None = None
    if is_double_call:
        inner = getattr(node, "func")
        name_node = getattr(inner, "func")
        candidate = func_globals.get(name_node.id)
        if isinstance(candidate, type) and issubclass(candidate, NKIOp):
            result = candidate
    return result


def _extract_name_kwargs(call: ast.Call) -> dict[str, str]:
    """Return ``{arg_name: variable_name}`` for Name-valued kwargs.

    Only kwargs that reference local variables get captured — these
    are the tensor operands.
    """
    return {kw.arg: kw.value.id for kw in call.keywords if kw.arg is not None and isinstance(kw.value, ast.Name)}


def _literal_value(node: ast.expr, func_globals: dict[str, object]) -> tuple[bool, Any]:
    """Try to evaluate ``node`` as a Python literal (incl. arithmetic BinOps).

    Returns ``(ok, value)``. Captures plain ``Constant`` plus arithmetic
    BinOps over constants (``1/2048``, ``-eps``, ``2*pi`` etc.). Name
    references are resolved against ``func_globals`` when they point to
    numeric constants — covers kwargs like ``bias=EPS`` that reference
    a module-level constant.
    """
    try:
        return True, ast.literal_eval(node)
    except (ValueError, SyntaxError):
        pass
    if isinstance(node, ast.Name):
        value = func_globals.get(node.id)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return True, value
    if isinstance(node, ast.UnaryOp):
        ok, inner = _literal_value(node.operand, func_globals)
        if ok and isinstance(node.op, ast.USub):
            return True, -inner
        if ok and isinstance(node.op, ast.UAdd):
            return True, +inner
    if isinstance(node, ast.BinOp):
        ok_l, lhs = _literal_value(node.left, func_globals)
        ok_r, rhs = _literal_value(node.right, func_globals)
        if ok_l and ok_r:
            if isinstance(node.op, ast.Add):
                return True, lhs + rhs
            if isinstance(node.op, ast.Sub):
                return True, lhs - rhs
            if isinstance(node.op, ast.Mult):
                return True, lhs * rhs
            if isinstance(node.op, ast.Div):
                return True, lhs / rhs
    return False, None


def _extract_literal_kwargs(call: ast.Call, func_globals: dict[str, object]) -> dict[str, Any]:
    """Return ``{arg_name: literal}`` for constant-valued kwargs on ``call``.

    Handles plain ``ast.Constant``, arithmetic BinOp trees whose leaves
    are all constants (e.g. ``scale=1/2048``), and Name references that
    resolve to numeric constants in ``func_globals`` (e.g.
    ``bias=EPS`` where ``EPS = 1e-6`` lives at module scope).
    Tensor-valued (Name → ndarray) kwargs are captured separately by
    :func:`_extract_name_kwargs`.
    """
    out: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            continue
        ok, value = _literal_value(kw.value, func_globals)
        if ok:
            out[kw.arg] = value
    return out


def _extract_output_names(target: ast.expr) -> list[str]:
    """Extract output variable names from an assignment target."""
    if isinstance(target, ast.Name):
        names = [target.id]
    elif isinstance(target, ast.Tuple):
        names = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
    else:
        names = []
    return names


def _parse_op_assignment(stmt: ast.Assign, func_globals: dict[str, object]) -> _ParsedOp | None:
    """Try to parse an assignment as an NKIOp call.

    Literal kwargs are split across two ast.Call nodes:

    * Outer call — tensor operands (``data=lhs``) and call-site literals
      (e.g. ``scale=1/2048`` if the user chose to put it there).
    * Inner ``OpClass(...)`` call — op configuration literals
      (``op='square'``, ``reduce_op='add'``, ``post_op='rsqrt'``).

    We merge both into one ``op_kwargs`` dict; the outer call's Name-
    valued kwargs become ``name_kwargs`` (tensor references).

    Returns ``(op_cls, name_kwargs, op_kwargs, output_names)`` or None.
    """
    result: _ParsedOp | None = None
    op_cls = _resolve_op_class(stmt.value, func_globals) if len(stmt.targets) == 1 else None
    if op_cls is not None:
        output_names = _extract_output_names(stmt.targets[0])
        if output_names:
            if len(output_names) != len(op_cls.OUTPUT_AXES):
                raise ValueError(
                    f"Op {op_cls.NAME}: {len(output_names)} outputs assigned"
                    f" but OUTPUT_AXES has {len(op_cls.OUTPUT_AXES)} entries"
                )
            assert isinstance(stmt.value, ast.Call)
            name_kwargs = _extract_name_kwargs(stmt.value)
            outer_kwargs = _extract_literal_kwargs(stmt.value, func_globals)
            inner_call = stmt.value.func
            assert isinstance(inner_call, ast.Call)
            op_kwargs = {**_extract_literal_kwargs(inner_call, func_globals), **outer_kwargs}
            result = (op_cls, name_kwargs, op_kwargs, output_names)
    return result


def find_ops(func: Callable[..., np.ndarray]) -> tuple[list[_ParsedOp], str]:
    """Parse *func* to extract NKIOp calls and the return name.

    Returns:
        Tuple of (ops_list, return_name) where ops_list contains
        ``(op_class, name_kwargs, output_names)`` tuples.

    Raises:
        ValueError: On missing return statement.
    """
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise ValueError("Expected a function definition")

    ops: list[_ParsedOp] = []
    return_name: str | None = None

    for stmt in func_def.body:
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name):
            return_name = stmt.value.id

        if isinstance(stmt, ast.Assign):
            parsed = _parse_op_assignment(stmt, func.__globals__)
            if parsed is not None:
                ops.append(parsed)

    if return_name is None:
        raise ValueError("Math function must have a 'return <variable>' statement")
    return ops, return_name
