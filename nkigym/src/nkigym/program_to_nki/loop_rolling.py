"""Loop rolling codegen pass.

Detects repeating statement patterns in fully-unrolled tiled functions
and rolls them into for loops. Iterates until convergence to produce
maximally nested loop structures.

This is a generic Python AST pass (str -> str) that works on any
Python function source, not tied to specific frameworks.
"""

import ast
import copy

from nkigym.program_to_nki._loop_rolling_search import (
    _LoopRun,
    _ReductionChain,
    _SearchData,
    classify_body_zones,
    find_all_runs_for_k,
    find_reduction_chain,
    prepare_search_data,
)


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
    elif stride > 0 and base % stride == 0:
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


def _rename_var(stmts: list[ast.stmt], old: str, new: str) -> None:
    """Rename all ast.Name nodes with id=old to id=new."""
    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and node.id == old:
                node.id = new


def _apply_reduction_chain(
    body: list[ast.stmt], chain: _ReductionChain, working: list[ast.stmt], loop_var: str, prologue_end: int
) -> None:
    """Apply a reduction chain roll to the body, mutating it in place."""
    acc_name = f"acc_{loop_var[2:]}"
    chain_end = chain.chain_start + chain.trip_count * chain.block_size

    """Rename last output in post-chain stmts before splicing."""
    for i in range(prologue_end + chain_end, len(body)):
        _rename_var([body[i]], chain.last_output, acc_name)

    """Build peel: deep copy with carried output renamed."""
    peel = [copy.deepcopy(s) for s in working[chain.peel_start : chain.chain_start]]
    _rename_var(peel, chain.carried_input, acc_name)

    """Build loop: reuse _build_for, then rename carried vars."""
    chain_run = _LoopRun(chain.chain_start, chain.block_size, chain.trip_count, chain.varying)
    for_node = _build_for(chain_run, working, loop_var)
    _rename_var(for_node.body, chain.carried_input, acc_name)
    _rename_var(for_node.body, chain.carried_output, acc_name)

    actual_start = prologue_end + chain.peel_start
    actual_end = prologue_end + chain_end
    body[actual_start:actual_end] = peel + [for_node]


def _count_loop_depth(tree: ast.AST) -> int:
    """Count the number of distinct i_N loop variable names in the AST."""
    loop_vars: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            name = node.target.id
            if name.startswith("i_"):
                loop_vars.add(name)
    return len(loop_vars)


def _try_spatial_roll(
    body: list[ast.stmt], working: list[ast.stmt], loop_var: str, prologue_end: int, data: _SearchData
) -> bool:
    """Try spatial (identical-block) rolling on the working zone."""
    n = len(working)
    best_runs: list[_LoopRun] = []
    best_coverage = 0
    best_max_run = 0
    for k in range(1, n // 2 + 1):
        if k * (n // k) < best_coverage:
            continue
        runs = find_all_runs_for_k(working, k, data)
        coverage = sum(r.trip_count * r.block_size for r in runs)
        max_run = max((r.trip_count * r.block_size for r in runs), default=0)
        if coverage > best_coverage or (coverage == best_coverage and max_run > best_max_run):
            best_runs = runs
            best_coverage = coverage
            best_max_run = max_run

    if best_runs:
        for r in reversed(best_runs):
            for_node = _build_for(r, working, loop_var)
            actual_start = prologue_end + r.start_idx
            actual_end = actual_start + r.trip_count * r.block_size
            body[actual_start:actual_end] = [for_node]

    return bool(best_runs)


def _try_reduction_roll(
    body: list[ast.stmt], working: list[ast.stmt], loop_var: str, prologue_end: int, data: _SearchData
) -> bool:
    """Try reduction chain rolling on the working zone."""
    chains = find_reduction_chain(working, data)
    if chains:
        _apply_reduction_chain(body, chains[0], working, loop_var, prologue_end)
    return bool(chains)


def _try_roll_in_body(body: list[ast.stmt], loop_var: str) -> bool:
    """Try to find and apply one repeating run in a statement list."""
    prologue_end, epilogue_start = classify_body_zones(body)
    working = body[prologue_end:epilogue_start]

    data = prepare_search_data(working) if working else None
    rolled = bool(data) and _try_spatial_roll(body, working, loop_var, prologue_end, data)

    if not rolled and data:
        rolled = _try_reduction_roll(body, working, loop_var, prologue_end, data)

    if not rolled:
        for stmt in body:
            if isinstance(stmt, ast.For) and _try_roll_in_body(stmt.body, loop_var):
                rolled = True
                break

    return rolled


def _roll_once(source: str) -> str:
    """Apply one loop rolling step to the source."""
    tree = ast.parse(source)

    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break

    result = source
    if func_def is not None:
        next_idx = _count_loop_depth(tree)
        loop_var = f"i_{next_idx}"
        if _try_roll_in_body(func_def.body, loop_var):
            ast.fix_missing_locations(tree)
            result = ast.unparse(tree)

    return result


def roll_loops(source: str) -> str:
    """Roll all repeating statement patterns into for loops.

    Args:
        source: Python source code string containing a function definition.

    Returns:
        Rolled source code with loops replacing repeated blocks.
    """
    while True:
        new_source = _roll_once(source)
        if new_source == source:
            break
        source = new_source
    return source
