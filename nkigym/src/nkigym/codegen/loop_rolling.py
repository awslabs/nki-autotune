"""Loop rolling codegen pass.

Detects repeating statement patterns in fully-unrolled tiled functions
and rolls them into for loops. Iterates until convergence to produce
maximally nested loop structures.

This is a generic Python AST pass (str -> str) that works on any
Python function source, not tied to specific frameworks.
"""

import ast
import copy
from dataclasses import dataclass, field


@dataclass
class VaryingConstant:
    """A constant that varies across blocks in an arithmetic progression.

    Attributes:
        path: Navigation path from statement root to the constant node.
            Each step is (field_name, index_or_none).
        stmt_offset: Index of the statement within the block.
        base: First value in the arithmetic progression.
        stride: Common difference between consecutive values.
    """

    path: tuple[tuple[str, int | None], ...]
    stmt_offset: int
    base: int
    stride: int


@dataclass
class _LoopRun:
    """A detected repeating run of structurally identical blocks.

    Attributes:
        start_idx: Starting index in the working statements list.
        block_size: Number of statements per block (K).
        trip_count: Number of consecutive matching blocks (N).
        varying: List of constants that vary across blocks.
    """

    start_idx: int
    block_size: int
    trip_count: int
    varying: list[VaryingConstant] = field(default_factory=list)


_NO_RUN = _LoopRun(start_idx=0, block_size=0, trip_count=0)


def _is_alloc_stmt(stmt: ast.stmt) -> bool:
    """Check if a statement is an np.empty allocation."""
    return (
        isinstance(stmt, ast.Assign)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and isinstance(stmt.value.func.value, ast.Name)
        and stmt.value.func.value.id == "np"
        and stmt.value.func.attr == "empty"
    )


def _classify_body_zones(body: list[ast.stmt]) -> tuple[int, int]:
    """Partition a statement list into prologue, working, epilogue zones.

    Prologue: leading np.empty allocations.
    Epilogue: trailing return statement.
    Working: everything in between.
    """
    prologue_end = 0
    for i, stmt in enumerate(body):
        if _is_alloc_stmt(stmt):
            prologue_end = i + 1
        else:
            break

    epilogue_start = len(body)
    if body and isinstance(body[-1], ast.Return):
        epilogue_start = len(body) - 1

    return prologue_end, epilogue_start


def _collect_target_mapping(stmts: list[ast.stmt]) -> dict[str, str]:
    """Map assignment and for-loop targets to positional variable names."""
    var_map: dict[str, str] = {}
    counter = 0

    def _process(stmt_list: list[ast.stmt]) -> None:
        """Recursively collect target names from a statement list."""
        nonlocal counter
        for stmt in stmt_list:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id not in var_map:
                        var_map[target.id] = f"_v{counter}"
                        counter += 1
            elif isinstance(stmt, ast.For):
                if isinstance(stmt.target, ast.Name) and stmt.target.id not in var_map:
                    var_map[stmt.target.id] = f"_v{counter}"
                    counter += 1
                _process(stmt.body)

    _process(stmts)
    return var_map


class _AstNormalizer(ast.NodeTransformer):
    """Replace local names with positional names and zero out int constants."""

    def __init__(self, var_map: dict[str, str]) -> None:
        """Initialize with a variable-to-positional-name mapping."""
        self.var_map = var_map

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Replace local variable names with positional equivalents."""
        if node.id in self.var_map:
            node.id = self.var_map[node.id]
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        """Zero out integer constants for structural comparison."""
        if isinstance(node.value, int):
            node.value = 0
        return node


def _normalize_block(stmts: list[ast.stmt]) -> str:
    """Normalize a block of statements for structural comparison.

    Renames local variables to positional names (_v0, _v1, ...) and
    replaces all int constants with 0.
    """
    stmts_copy = [copy.deepcopy(s) for s in stmts]
    var_map = _collect_target_mapping(stmts_copy)
    normalizer = _AstNormalizer(var_map)
    for i, stmt in enumerate(stmts_copy):
        stmts_copy[i] = normalizer.visit(stmt)
    return "\n".join(ast.dump(s) for s in stmts_copy)


def _collect_int_constants(node: ast.AST) -> list[tuple[tuple[tuple[str, int | None], ...], int]]:
    """Collect all int constants from an AST node in DFS order."""
    results: list[tuple[tuple[tuple[str, int | None], ...], int]] = []
    _walk_constants(node, (), results)
    return results


def _walk_constants(
    node: ast.AST,
    path: tuple[tuple[str, int | None], ...],
    results: list[tuple[tuple[tuple[str, int | None], ...], int]],
) -> None:
    """Recursively walk AST collecting int constants with their paths."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        results.append((path, node.value))
        return
    for field_name, field_value in ast.iter_fields(node):
        if isinstance(field_value, list):
            for i, item in enumerate(field_value):
                if isinstance(item, ast.AST):
                    _walk_constants(item, path + ((field_name, i),), results)
        elif isinstance(field_value, ast.AST):
            _walk_constants(field_value, path + ((field_name, None),), results)


def _is_arithmetic(values: list[int]) -> tuple[bool, int, int]:
    """Check if values form an arithmetic progression.

    Returns (is_valid, base, stride). When is_valid is False, base
    and stride are zero.
    """
    base = 0
    stride = 0
    is_valid = bool(values)
    if is_valid:
        base = values[0]
        stride = values[1] - values[0] if len(values) >= 2 else 0
        is_valid = all(v == base + i * stride for i, v in enumerate(values))
    return (is_valid, base, stride)


def _check_stmt_varying(
    blocks: list[list[ast.stmt]], stmt_offset: int, trip_count: int, varying: list[VaryingConstant]
) -> bool:
    """Check one statement position across blocks for arithmetic patterns.

    Appends any varying constants found to the varying list.
    Returns False if constant counts mismatch or any position fails
    the arithmetic progression check.
    """
    all_constants = [_collect_int_constants(blocks[bi][stmt_offset]) for bi in range(trip_count)]
    num_constants = len(all_constants[0])
    valid = all(len(bc) == num_constants for bc in all_constants[1:])

    if valid:
        for const_idx in range(num_constants):
            values = [all_constants[bi][const_idx][1] for bi in range(trip_count)]
            is_ap, base, stride = _is_arithmetic(values)
            if not is_ap:
                valid = False
                break
            if stride != 0:
                path = all_constants[0][const_idx][0]
                varying.append(VaryingConstant(path=path, stmt_offset=stmt_offset, base=base, stride=stride))

    return valid


def _extract_varying(
    working_stmts: list[ast.stmt], block_size: int, trip_count: int, start_idx: int
) -> tuple[bool, list[VaryingConstant]]:
    """Extract varying constants from a run of structurally identical blocks.

    Returns (valid, varying_list). When valid is False, the varying
    list should be ignored.
    """
    blocks = []
    for i in range(trip_count):
        offset = start_idx + i * block_size
        blocks.append(working_stmts[offset : offset + block_size])

    varying: list[VaryingConstant] = []
    valid = all(_check_stmt_varying(blocks, so, trip_count, varying) for so in range(block_size))
    return (valid, varying)


def _count_matching_blocks(
    working_stmts: list[ast.stmt], start: int, block_size: int, n: int, cache: dict[int, str]
) -> int:
    """Count consecutive structurally identical blocks starting at a position."""
    if start not in cache:
        cache[start] = _normalize_block(working_stmts[start : start + block_size])
    ref = cache[start]
    count = 1
    while start + (count + 1) * block_size <= n:
        pos = start + count * block_size
        if pos not in cache:
            cache[pos] = _normalize_block(working_stmts[pos : pos + block_size])
        if cache[pos] != ref:
            break
        count += 1
    return count


def _find_best_run(working_stmts: list[ast.stmt]) -> _LoopRun:
    """Find the repeating run with largest coverage among statements.

    Scans all block sizes K and positions P, scoring each candidate run
    as count * K (total statements covered). Blocks match when their
    normalized AST structure is identical (variables renamed to positional
    placeholders, integer constants zeroed). A candidate is valid only if
    every integer constant across its blocks forms an arithmetic progression.

    Ties are broken by discovery order (smallest K, earliest P).

    Returns _NO_RUN sentinel (trip_count=0) if no repeating pattern exists.
    """
    n = len(working_stmts)
    best = _NO_RUN
    best_coverage = 0
    for k in range(1, n // 2 + 1):
        if k * (n // k) <= best_coverage:
            continue
        cache: dict[int, str] = {}
        p = 0
        while p + 2 * k <= n:
            count = _count_matching_blocks(working_stmts, p, k, n, cache)
            if count >= 2 and count * k > best_coverage:
                valid, varying = _extract_varying(working_stmts, k, count, p)
                if valid:
                    best = _LoopRun(p, k, count, varying)
                    best_coverage = count * k
            p += 1
    return best


def _stride_expr(base_expr: ast.expr, stride: int) -> ast.expr:
    """Wrap an expression with a stride multiplier (identity when stride=1)."""
    if stride == 1:
        result = base_expr
    else:
        result = ast.BinOp(left=base_expr, op=ast.Mult(), right=ast.Constant(value=stride))
    return result


def _make_expr(loop_var: str, base: int, stride: int) -> ast.expr:
    """Create an AST expression for a varying constant.

    Simplification rules: stride=0 gives base; base=0 gives i*stride;
    base divisible by stride gives (i+offset)*stride; else i*stride+base.
    """
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
    """Build an ast.For node from a detected LoopRun.

    Deep-copies the first block as a template, replaces varying
    constants with arithmetic expressions, and wraps in a for loop.
    """
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


def _count_loop_depth(tree: ast.AST) -> int:
    """Count the number of distinct i_N loop variable names in the AST."""
    loop_vars: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            name = node.target.id
            if name.startswith("i_"):
                loop_vars.add(name)
    return len(loop_vars)


def _try_roll_in_body(body: list[ast.stmt], loop_var: str) -> bool:
    """Try to find and apply one repeating run in a statement list.

    Searches the given body for repeating patterns. If none found
    at this level, recurses into for loop bodies.
    """
    prologue_end, epilogue_start = _classify_body_zones(body)
    working = body[prologue_end:epilogue_start]

    run = _find_best_run(working)
    rolled = False
    if run.trip_count > 0:
        for_node = _build_for(run, working, loop_var)
        actual_start = prologue_end + run.start_idx
        actual_end = actual_start + run.trip_count * run.block_size
        body[actual_start:actual_end] = [for_node]
        rolled = True

    if not rolled:
        for stmt in body:
            if isinstance(stmt, ast.For) and _try_roll_in_body(stmt.body, loop_var):
                rolled = True
                break

    return rolled


def _roll_once(source: str) -> str:
    """Apply one loop rolling step to the source.

    Parses the source, finds the best repeating run anywhere in the
    AST (including inside existing for loop bodies), and replaces it
    with a for loop.
    """
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

    Iteratively detects repeating blocks and collapses them into loops
    until no more patterns remain. Produces maximally nested loops.

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
