"""Strict structural comparison of rendered output and hand-written kernels.

The comparison preserves every statement, assertion, declaration position,
buffer name, operation argument, and nesting decision. It normalizes only the
generated function name and equivalent integer affine spellings.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable


class _Canonicalize(ast.NodeTransformer):
    """Normalize the generated function name."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Give the single compared kernel a stable name."""
        node.name = "KERNEL"
        self.generic_visit(node)
        return node


class _ConstantFold(ast.NodeTransformer):
    """Canonicalize integer affine slice arithmetic."""

    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        """Fold constant operands and constant-scaled affine expressions."""
        self.generic_visit(node)
        left, right = node.left, node.right
        result: ast.expr = node
        if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
            if isinstance(left.value, int) and isinstance(right.value, int):
                result = ast.Constant(value=_apply_binop(node.op, left.value, right.value))
        elif isinstance(node.op, ast.Mult):
            result = self._scale(node, left, right)
        return result

    def _scale(self, node: ast.BinOp, left: ast.expr, right: ast.expr) -> ast.expr:
        """Distribute or reassociate one integer constant factor."""
        inner, factor = _scaled_operands(left, right)
        result: ast.expr = node
        if inner is not None and factor is not None and isinstance(inner.op, (ast.Add, ast.Sub)):
            result = self.visit(
                ast.BinOp(
                    left=ast.BinOp(left=inner.left, op=ast.Mult(), right=factor),
                    op=inner.op,
                    right=ast.BinOp(left=inner.right, op=ast.Mult(), right=factor),
                )
            )
        elif inner is not None and factor is not None and isinstance(inner.op, ast.Mult) and _is_int_const(inner.right):
            result = self.visit(
                ast.BinOp(
                    left=inner.left, op=ast.Mult(), right=ast.BinOp(left=inner.right, op=ast.Mult(), right=factor)
                )
            )
        return result


def _scaled_operands(left: ast.expr, right: ast.expr) -> tuple[ast.BinOp | None, ast.expr | None]:
    """Return the affine expression and integer factor in a multiplication."""
    inner: ast.BinOp | None = None
    factor: ast.expr | None = None
    if isinstance(left, ast.BinOp) and _is_int_const(right):
        inner, factor = left, right
    elif isinstance(right, ast.BinOp) and _is_int_const(left):
        inner, factor = right, left
    return inner, factor


def _is_int_const(node: ast.expr) -> bool:
    """Return whether ``node`` is an integer constant."""
    return isinstance(node, ast.Constant) and isinstance(node.value, int)


def _apply_binop(op: ast.operator, left: int, right: int) -> int:
    """Evaluate an integer operator used in affine slice arithmetic."""
    if isinstance(op, ast.Add):
        result = left + right
    elif isinstance(op, ast.Sub):
        result = left - right
    elif isinstance(op, ast.Mult):
        result = left * right
    else:
        raise TypeError(f"unsupported constant-fold operator {type(op).__name__}")
    return result


def _single_function_def(module: ast.Module) -> ast.FunctionDef:
    """Return the module's sole function definition."""
    functions = [statement for statement in module.body if isinstance(statement, ast.FunctionDef)]
    if len(functions) != 1:
        raise AssertionError(f"expected exactly one function def; got {len(functions)}")
    return functions[0]


def _normalize(source: str) -> str:
    """Return a strict canonical AST dump for one kernel source."""
    function = _single_function_def(ast.parse(source))
    canonical = _Canonicalize().visit(function)
    folded = _ConstantFold().visit(canonical)
    ast.fix_missing_locations(folded)
    return ast.dump(folded, annotate_fields=True)


def assert_matches_hand(rendered_src: str, hand_fn: Callable[..., object]) -> None:
    """Assert rendered source matches a hand-written kernel."""
    got = _normalize(rendered_src)
    want = _normalize(inspect.getsource(hand_fn))
    assert got == want, f"rendered != hand kernel\n--- got ---\n{got}\n--- want ---\n{want}"


def assert_matches_render_ordered(rendered_src: str, expected_src: str) -> None:
    """Assert two kernel sources match with statement order preserved."""
    got = _normalize(rendered_src)
    want = _normalize(expected_src)
    assert got == want, f"rendered != expected\n--- got ---\n{got}\n--- want ---\n{want}"
