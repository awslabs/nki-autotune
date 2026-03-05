"""AST types and code generation helpers for loop rolling."""

import ast
import copy
from dataclasses import dataclass, field


@dataclass
class VaryingConstant:
    """A constant that varies across blocks in an arithmetic progression."""

    path: tuple[tuple[str, int | None], ...]
    stmt_offset: int
    base: int
    stride: int


@dataclass
class _LoopRun:
    """A detected repeating run of structurally identical blocks."""

    start_idx: int
    block_size: int
    trip_count: int
    varying: list[VaryingConstant] = field(default_factory=list)


_NO_RUN = _LoopRun(start_idx=0, block_size=0, trip_count=0)


def _stride_expr(base_expr: ast.expr, stride: int) -> ast.expr:
    """Wrap an expression with a stride multiplier (identity when stride=1)."""
    if stride == 1:
        result = base_expr
    else:
        result = ast.BinOp(left=base_expr, op=ast.Mult(), right=ast.Constant(value=stride))
    return result


def _make_expr(loop_var: str, base: int, stride: int) -> ast.expr:
    """Build loop-variable expression: base + i * stride with simplifications."""
    loop_name = ast.Name(id=loop_var, ctx=ast.Load())
    if stride == 0:
        result = ast.Constant(value=base)
    elif base == 0:
        result = _stride_expr(loop_name, stride)
    elif base % stride == 0:
        inner = ast.BinOp(left=loop_name, op=ast.Add(), right=ast.Constant(value=base // stride))
        result = _stride_expr(inner, stride)
    else:
        mult = ast.BinOp(left=loop_name, op=ast.Mult(), right=ast.Constant(value=stride))
        result = ast.BinOp(left=mult, op=ast.Add(), right=ast.Constant(value=base))
    return result


def _set_at_path(root: ast.AST, path: tuple[tuple[str, int | None], ...], new_node: ast.AST) -> None:
    """Replace the node at a given path with a new node."""
    node = root
    for step in path[:-1]:
        field_name, index = step
        value = getattr(node, field_name)
        if index is not None:
            node = value[index]
        else:
            node = value

    field_name, index = path[-1]
    if index is not None:
        getattr(node, field_name)[index] = new_node
    else:
        setattr(node, field_name, new_node)


def _build_for(run: _LoopRun, working_stmts: list[ast.stmt], loop_var: str) -> ast.For:
    """Build an ast.For node from a detected LoopRun."""
    template = [copy.deepcopy(s) for s in working_stmts[run.start_idx : run.start_idx + run.block_size]]

    for vc in run.varying:
        expr = _make_expr(loop_var, vc.base, vc.stride)
        _set_at_path(template[vc.stmt_offset], vc.path, expr)

    return ast.For(
        target=ast.Name(id=loop_var, ctx=ast.Store()),
        iter=ast.Call(
            func=ast.Name(id="range", ctx=ast.Load()), args=[ast.Constant(value=run.trip_count)], keywords=[]
        ),
        body=template,
        orelse=[],
    )
