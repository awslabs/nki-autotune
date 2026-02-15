"""Loop rolling codegen pass for NKI Gym.

Detects repeating statement patterns in fully-unrolled tiled functions
and rolls them into for loops. Iterates until convergence to produce
maximally nested loop structures.

This is a deterministic codegen pass (not a transform) that always
maximally compresses the IR.
"""

import ast
import copy
from collections.abc import Callable
from dataclasses import dataclass, field

from nkigym.utils.source import callable_to_source, source_to_callable


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


def _is_alloc_stmt(stmt: ast.stmt) -> bool:
    """Check if a statement is an nkigym.ndarray allocation.

    Args:
        stmt: AST statement node.

    Returns:
        True if the statement is an nkigym.ndarray call.
    """
    if not isinstance(stmt, ast.Assign):
        return False
    if not isinstance(stmt.value, ast.Call):
        return False
    call = stmt.value
    if not isinstance(call.func, ast.Attribute):
        return False
    if not isinstance(call.func.value, ast.Name):
        return False
    return call.func.value.id == "nkigym" and call.func.attr == "ndarray"


def _classify_body_zones(body: list[ast.stmt]) -> tuple[int, int]:
    """Partition a statement list into prologue, working, epilogue zones.

    Prologue: leading nkigym.ndarray allocations.
    Epilogue: trailing return statement.
    Working: everything in between.

    Args:
        body: List of AST statements.

    Returns:
        Tuple of (prologue_end, epilogue_start) indices.
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


def _normalize_block(stmts: list[ast.stmt]) -> str:
    """Normalize a block for structural comparison.

    Renames local variables (assignment targets, for targets) to
    positional names (_v0, _v1, ...) and replaces all int constants
    with 0. Recurses into for loop bodies.

    Args:
        stmts: List of AST statements forming the block.

    Returns:
        Normalized ast.dump string for comparison.
    """
    stmts_copy = [copy.deepcopy(s) for s in stmts]
    var_map: dict[str, str] = {}
    counter = [0]

    def _next_var() -> str:
        """Return the next positional variable name (_v0, _v1, ...)."""
        name = f"_v{counter[0]}"
        counter[0] += 1
        return name

    def _collect_targets(stmt_list: list[ast.stmt]) -> None:
        """Map assignment and for-loop targets to positional variable names."""
        for stmt in stmt_list:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id not in var_map:
                        var_map[target.id] = _next_var()
            elif isinstance(stmt, ast.For):
                if isinstance(stmt.target, ast.Name) and stmt.target.id not in var_map:
                    var_map[stmt.target.id] = _next_var()
                _collect_targets(stmt.body)

    _collect_targets(stmts_copy)

    class _Normalizer(ast.NodeTransformer):
        """Replace local names with positional names and zero out int constants."""

        def visit_Name(self, node: ast.Name) -> ast.Name:
            if node.id in var_map:
                node.id = var_map[node.id]
            return node

        def visit_Constant(self, node: ast.Constant) -> ast.Constant:
            if isinstance(node.value, int):
                node.value = 0
            return node

    normalizer = _Normalizer()
    for i, stmt in enumerate(stmts_copy):
        stmts_copy[i] = normalizer.visit(stmt)

    return "\n".join(ast.dump(s) for s in stmts_copy)


def _collect_int_constants(node: ast.AST) -> list[tuple[tuple[tuple[str, int | None], ...], int]]:
    """Collect all int constants from an AST node in DFS order.

    Args:
        node: AST node to walk.

    Returns:
        List of (path, value) pairs. Path is a tuple of navigation
        steps from the node to the constant.
    """
    results: list[tuple[tuple[tuple[str, int | None], ...], int]] = []
    _walk_constants(node, (), results)
    return results


def _walk_constants(
    node: ast.AST,
    path: tuple[tuple[str, int | None], ...],
    results: list[tuple[tuple[tuple[str, int | None], ...], int]],
) -> None:
    """Recursively walk AST collecting int constants with their paths.

    Args:
        node: Current AST node.
        path: Current navigation path from root.
        results: Accumulator for (path, value) pairs.
    """
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


def _is_arithmetic(values: list[int]) -> tuple[int, int] | None:
    """Check if values form an arithmetic progression.

    Args:
        values: List of integer values.

    Returns:
        Tuple of (base, stride) if arithmetic, None otherwise.
    """
    if len(values) < 2:
        return (values[0], 0) if values else None

    base = values[0]
    stride = values[1] - values[0]

    for i, v in enumerate(values):
        if v != base + i * stride:
            return None

    return (base, stride)


def _extract_varying(
    working_stmts: list[ast.stmt], block_size: int, trip_count: int, start_idx: int
) -> list[VaryingConstant] | None:
    """Extract varying constants from a run of structurally identical blocks.

    Walks corresponding AST nodes in parallel across all blocks. Each
    int constant position must form an arithmetic progression.

    Args:
        working_stmts: Full list of working statements.
        block_size: Number of statements per block.
        trip_count: Number of consecutive blocks.
        start_idx: Starting index in working_stmts.

    Returns:
        List of VaryingConstant for non-zero-stride positions,
        or None if any position fails the AP check.
    """
    blocks = []
    for i in range(trip_count):
        offset = start_idx + i * block_size
        blocks.append(working_stmts[offset : offset + block_size])

    varying: list[VaryingConstant] = []

    for stmt_offset in range(block_size):
        all_constants = []
        for block_idx in range(trip_count):
            constants = _collect_int_constants(blocks[block_idx][stmt_offset])
            all_constants.append(constants)

        num_constants = len(all_constants[0])
        for block_constants in all_constants[1:]:
            if len(block_constants) != num_constants:
                return None

        for const_idx in range(num_constants):
            values = [all_constants[block_idx][const_idx][1] for block_idx in range(trip_count)]
            ap = _is_arithmetic(values)
            if ap is None:
                return None

            base, stride = ap
            if stride != 0:
                path = all_constants[0][const_idx][0]
                varying.append(VaryingConstant(path=path, stmt_offset=stmt_offset, base=base, stride=stride))

    return varying


def _find_best_run(working_stmts: list[ast.stmt]) -> _LoopRun | None:
    """Find the repeating run with largest coverage among statements.

    Tries all block sizes K from largest to smallest, and for each K
    scans all starting positions for consecutive structurally identical
    blocks whose constants form arithmetic progressions.

    Args:
        working_stmts: List of AST statements to search.

    Returns:
        Best _LoopRun found, or None if no repeating pattern exists.
    """
    n = len(working_stmts)
    if n < 2:
        return None

    best: _LoopRun | None = None
    best_coverage = 0

    for k in range(1, n // 2 + 1):
        if k * (n // k) <= best_coverage:
            continue
        p = 0
        while p + 2 * k <= n:
            ref = _normalize_block(working_stmts[p : p + k])
            count = 1
            while p + (count + 1) * k <= n:
                cand = _normalize_block(working_stmts[p + count * k : p + (count + 1) * k])
                if cand != ref:
                    break
                count += 1

            if count >= 2:
                coverage = count * k
                if coverage > best_coverage:
                    varying = _extract_varying(working_stmts, k, count, p)
                    if varying is not None:
                        best = _LoopRun(p, k, count, varying)
                        best_coverage = coverage

            p += 1

    return best


def _make_expr(loop_var: str, base: int, stride: int) -> ast.expr:
    """Create an AST expression for a varying constant.

    Applies simplification rules (checked in order):
    - stride=0: just ``base``
    - base=0: ``i * stride`` (or just ``i`` when stride=1)
    - base divisible by stride: ``(i + base//stride) * stride``
    - general: ``i * stride + base``

    Args:
        loop_var: Name of the loop iteration variable.
        base: First value of the arithmetic progression.
        stride: Common difference.

    Returns:
        AST expression node.
    """
    if stride == 0:
        return ast.Constant(value=base)

    loop_name = ast.Name(id=loop_var, ctx=ast.Load())

    if base == 0:
        if stride == 1:
            return loop_name
        return ast.BinOp(left=loop_name, op=ast.Mult(), right=ast.Constant(value=stride))

    if base % stride == 0:
        offset = base // stride
        inner = ast.BinOp(left=loop_name, op=ast.Add(), right=ast.Constant(value=offset))
        if stride == 1:
            return inner
        return ast.BinOp(left=inner, op=ast.Mult(), right=ast.Constant(value=stride))

    mult = ast.BinOp(left=loop_name, op=ast.Mult(), right=ast.Constant(value=stride))
    return ast.BinOp(left=mult, op=ast.Add(), right=ast.Constant(value=base))


def _set_at_path(root: ast.AST, path: tuple[tuple[str, int | None], ...], new_node: ast.AST) -> None:
    """Replace the node at a given path with a new node.

    Navigates from root through the path steps, then replaces the
    final node with new_node.

    Args:
        root: Root AST node to navigate from.
        path: Tuple of (field_name, index_or_none) steps.
        new_node: Replacement AST node.
    """
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

    Args:
        run: Detected repeating run.
        working_stmts: Full list of working statements.
        loop_var: Name for the loop iteration variable.

    Returns:
        ast.For node containing the rolled loop.
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
    """Count the number of distinct loop variable names in the AST.

    Args:
        tree: AST to search.

    Returns:
        Number of distinct i_N loop variables found.
    """
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

    Args:
        body: List of AST statements (mutated in place).
        loop_var: Name for the new loop variable.

    Returns:
        True if a roll was applied, False otherwise.
    """
    prologue_end, epilogue_start = _classify_body_zones(body)
    working = body[prologue_end:epilogue_start]

    run = _find_best_run(working)
    if run is not None:
        for_node = _build_for(run, working, loop_var)
        actual_start = prologue_end + run.start_idx
        actual_end = actual_start + run.trip_count * run.block_size
        body[actual_start:actual_end] = [for_node]
        return True

    for stmt in body:
        if isinstance(stmt, ast.For):
            if _try_roll_in_body(stmt.body, loop_var):
                return True

    return False


def _roll_once(source: str) -> str:
    """Apply one loop rolling step to the source.

    Parses the source, finds the best repeating run anywhere in the
    AST (including inside existing for loop bodies), and replaces it
    with a for loop. Returns the source unchanged if no pattern is found.

    Args:
        source: Python source code string.

    Returns:
        New source string with one pattern rolled, or the input
        unchanged if no pattern was found.
    """
    tree = ast.parse(source)

    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break

    if func_def is None:
        return source

    next_idx = _count_loop_depth(tree)
    loop_var = f"i_{next_idx}"

    if not _try_roll_in_body(func_def.body, loop_var):
        return source

    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def roll_loops(func: Callable) -> Callable:
    """Roll all repeating statement patterns into for loops.

    Iteratively detects repeating blocks and collapses them into loops
    until no more patterns remain. Produces maximally nested loops.

    Args:
        func: nkigym function with __source__ attribute.

    Returns:
        New callable with loops replacing repeated blocks.
    """
    source = callable_to_source(func)

    while True:
        new_source = _roll_once(source)
        if new_source == source:
            break
        source = new_source

    tree = ast.parse(source)
    func_name = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            break

    if func_name is None:
        raise ValueError("No function definition found in source after loop rolling")

    return source_to_callable(source, func_name)
