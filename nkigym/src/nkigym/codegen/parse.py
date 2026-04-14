"""AST-based discovery of NKIOp calls in math functions."""

import ast
import inspect
import textwrap
from collections.abc import Callable

import numpy as np

from nkigym.ops.base import NKIOp


def _extract_op_call(stmt: ast.stmt) -> tuple[ast.Call | None, str | None]:
    """Extract OpClass()(kwargs) call and output name from a statement."""
    call_node: ast.Call | None = None
    output_name: str | None = None
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        if isinstance(target, ast.Name) and isinstance(stmt.value, ast.Call):
            call_node = stmt.value
            output_name = target.id
    elif isinstance(stmt, (ast.Expr, ast.Return)) and isinstance(stmt.value, ast.Call):
        call_node = stmt.value
    return call_node, output_name


def _resolve_op_class(call_node: ast.Call, func_globals: dict[str, object]) -> type[NKIOp] | None:
    """Resolve the NKIOp subclass from an OpClass()(...) call node."""
    cls: type[NKIOp] | None = None
    if isinstance(call_node.func, ast.Call) and isinstance(call_node.func.func, ast.Name):
        candidate = func_globals.get(call_node.func.func.id)
        if isinstance(candidate, type) and issubclass(candidate, NKIOp):
            cls = candidate
    return cls


def find_ops(func: Callable[..., np.ndarray]) -> tuple[list[tuple[NKIOp, dict[str, str], str]], str]:
    """Find NKIOp subclasses and the return variable via AST inspection."""
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    func_def = tree.body[0]
    assert isinstance(func_def, ast.FunctionDef)
    ops: list[tuple[NKIOp, dict[str, str], str]] = []
    return_name: str | None = None
    for stmt in func_def.body:
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
