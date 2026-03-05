"""Loop rolling codegen pass.

Detects repeating statement patterns in fully-unrolled tiled functions
and rolls them into for loops. Iterates until convergence to produce
maximally nested loop structures.

This is a generic Python AST pass (str -> str) that works on any
Python function source, not tied to specific frameworks.
"""

import ast
import re

import numpy as np

from ._loop_rolling_ast import _NO_RUN, VaryingConstant, _build_for, _LoopRun

_INT_CONSTANT_RE = re.compile(r"Constant\(value=\d+")
_NAME_RE = re.compile(r"id='[^']*'")


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
    """Partition a statement list into prologue, working, epilogue zones."""
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


def _normalize_block(stmts: list[ast.stmt], zeroed_cache: dict[int, str]) -> str:
    """Normalize a block for structural comparison (positional names, zeroed ints).

    Uses pre-zeroed dump strings (int constants already replaced) so only
    the variable-renaming regex is needed here.
    """
    var_map = _collect_target_mapping(stmts)
    parts = [zeroed_cache[id(s)] for s in stmts]
    raw = "\n".join(parts)
    if var_map:
        pattern = "|".join(re.escape(k) for k in sorted(var_map, key=len, reverse=True))
        raw = re.sub(f"id='({pattern})'", lambda m: f"id='{var_map[m.group(1)]}'", raw)
    return raw


def _fingerprint_stmt(stmt: ast.stmt, dump_cache: dict[int, str]) -> int:
    """Hash a statement with names and ints erased for fast rejection."""
    key = id(stmt)
    if key not in dump_cache:
        dump_cache[key] = ast.dump(stmt)
    raw = dump_cache[key]
    normalized = _INT_CONSTANT_RE.sub("Constant(value=0", _NAME_RE.sub("id='_'", raw))
    return hash(normalized)


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
    """Check if values form an arithmetic progression."""
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
    """Check one statement position across blocks for arithmetic patterns."""
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
    """Extract varying constants from a run of structurally identical blocks."""
    blocks = []
    for i in range(trip_count):
        offset = start_idx + i * block_size
        blocks.append(working_stmts[offset : offset + block_size])

    varying: list[VaryingConstant] = []
    valid = all(_check_stmt_varying(blocks, so, trip_count, varying) for so in range(block_size))
    return (valid, varying)


def _collect_assigned_names(stmts: list[ast.stmt]) -> set[str]:
    """Collect all assignment target names from a list of statements."""
    names: set[str] = set()
    for stmt in stmts:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _precompute_name_data(stmts: list[ast.stmt]) -> tuple[list[set[str]], list[frozenset[str]], list[frozenset[str]]]:
    """Precompute suffix reference unions and per-statement name sets in one pass.

    Single ast.walk pass collects both the suffix-ref sets (for scope safety)
    and per-statement assigned/referenced sets (for external-name filtering).
    Returns (suffix_refs, per_assigned, per_referenced).
    """
    n = len(stmts)
    per_assigned: list[frozenset[str]] = []
    per_referenced: list[frozenset[str]] = []
    for stmt in stmts:
        assigned: set[str] = set()
        referenced: set[str] = set()
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    assigned.add(target.id)
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                referenced.add(node.id)
        per_assigned.append(frozenset(assigned))
        per_referenced.append(frozenset(referenced))
    suffix_refs: list[set[str]] = [set() for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        suffix_refs[i] = suffix_refs[i + 1] | per_referenced[i]
    return suffix_refs, per_assigned, per_referenced


def _block_ext_names(
    per_assigned: list[frozenset[str]], per_referenced: list[frozenset[str]], start: int, block_size: int
) -> frozenset[str]:
    """Compute external names (referenced minus assigned) for a statement block."""
    assigned: set[str] = set()
    referenced: set[str] = set()
    for i in range(start, start + block_size):
        assigned |= per_assigned[i]
        referenced |= per_referenced[i]
    return frozenset(referenced - assigned)


def _ext_names_match(
    per_assigned: list[frozenset[str]],
    per_referenced: list[frozenset[str]],
    ext_cache: dict[int, frozenset[str]],
    pos: int,
    block_size: int,
) -> bool:
    """Check if consecutive blocks at pos and pos+block_size share external names."""
    if pos not in ext_cache:
        ext_cache[pos] = _block_ext_names(per_assigned, per_referenced, pos, block_size)
    cand = pos + block_size
    if cand not in ext_cache:
        ext_cache[cand] = _block_ext_names(per_assigned, per_referenced, cand, block_size)
    return ext_cache[pos] == ext_cache[cand]


def _count_matching_blocks(
    working_stmts: list[ast.stmt],
    start: int,
    block_size: int,
    n: int,
    cache: dict[int, str],
    zeroed_cache: dict[int, str],
    fingerprints: list[int],
) -> int:
    """Count consecutive structurally identical blocks starting at a position."""
    count = 1
    while start + (count + 1) * block_size <= n:
        pos = start + count * block_size
        if any(fingerprints[start + i] != fingerprints[pos + i] for i in range(block_size)):
            break
        if start not in cache:
            cache[start] = _normalize_block(working_stmts[start : start + block_size], zeroed_cache)
        if pos not in cache:
            cache[pos] = _normalize_block(working_stmts[pos : pos + block_size], zeroed_cache)
        if cache[pos] != cache[start]:
            break
        count += 1
    return count


def _block_match_positions(fps: np.ndarray, k: int) -> list[int]:
    """Find positions where K-element fingerprint blocks match their successor."""
    n = len(fps)
    num_positions = n - 2 * k + 1
    stride = fps.strides[0]
    block_a = np.lib.stride_tricks.as_strided(fps, shape=(num_positions, k), strides=(stride, stride))
    block_b = np.lib.stride_tricks.as_strided(fps[k:], shape=(num_positions, k), strides=(stride, stride))
    return np.flatnonzero(np.all(block_a == block_b, axis=1)).tolist()


def _prepare_search_data(
    working_stmts: list[ast.stmt],
) -> tuple[list[int], dict[int, str], np.ndarray, list[set[str]], list[frozenset[str]], list[frozenset[str]]]:
    """Build all precomputed data structures for the loop rolling search."""
    dump_cache: dict[int, str] = {}
    fingerprints = [_fingerprint_stmt(s, dump_cache) for s in working_stmts]
    zeroed_cache = {sid: _INT_CONSTANT_RE.sub("Constant(value=0", v) for sid, v in dump_cache.items()}
    fps = np.array(fingerprints, dtype=np.int64)
    suffix_refs, stmt_assigned, stmt_refs = _precompute_name_data(working_stmts)
    return fingerprints, zeroed_cache, fps, suffix_refs, stmt_assigned, stmt_refs


def _find_best_run(
    working_stmts: list[ast.stmt],
    fingerprints: list[int],
    zeroed_cache: dict[int, str],
    fps: np.ndarray,
    suffix_refs: list[set[str]],
    stmt_assigned: list[frozenset[str]],
    stmt_refs: list[frozenset[str]],
) -> _LoopRun:
    """Find the repeating run with largest coverage among statements."""
    n = len(working_stmts)
    best = _NO_RUN
    best_coverage = 0
    for k in range(1, n // 2 + 1):
        if k * (n // k) <= best_coverage:
            continue
        match_positions = _block_match_positions(fps, k)
        cache: dict[int, str] = {}
        ext_cache: dict[int, frozenset[str]] = {}
        for p in match_positions:
            if n - p <= best_coverage:
                break
            if not _ext_names_match(stmt_assigned, stmt_refs, ext_cache, p, k):
                continue
            count = _count_matching_blocks(working_stmts, p, k, n, cache, zeroed_cache, fingerprints)
            while count >= 2 and count * k > best_coverage:
                if _collect_assigned_names(working_stmts[p : p + count * k]) & suffix_refs[p + count * k]:
                    count -= 1
                    continue
                valid, varying = _extract_varying(working_stmts, k, count, p)
                if valid:
                    best = _LoopRun(p, k, count, varying)
                    best_coverage = count * k
                    break
                count -= 1
    return best


def _find_all_runs_for_k(
    working_stmts: list[ast.stmt],
    block_size: int,
    fingerprints: list[int],
    zeroed_cache: dict[int, str],
    fps: np.ndarray,
    suffix_refs: list[set[str]],
    stmt_assigned: list[frozenset[str]],
    stmt_refs: list[frozenset[str]],
) -> list[_LoopRun]:
    """Find all non-overlapping valid runs for a given block size."""
    n = len(working_stmts)
    match_positions = _block_match_positions(fps, block_size)
    runs: list[_LoopRun] = []
    occupied_end = 0
    cache: dict[int, str] = {}
    ext_cache: dict[int, frozenset[str]] = {}
    for p in match_positions:
        if p < occupied_end:
            continue
        if not _ext_names_match(stmt_assigned, stmt_refs, ext_cache, p, block_size):
            continue
        count = _count_matching_blocks(working_stmts, p, block_size, n, cache, zeroed_cache, fingerprints)
        while count >= 2:
            end = p + count * block_size
            if _collect_assigned_names(working_stmts[p:end]) & suffix_refs[end]:
                count -= 1
                continue
            valid, varying = _extract_varying(working_stmts, block_size, count, p)
            if valid:
                runs.append(_LoopRun(p, block_size, count, varying))
                occupied_end = end
                break
            count -= 1
    return runs


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
    """Try to find and apply one repeating run in a statement list."""
    prologue_end, epilogue_start = _classify_body_zones(body)
    working = body[prologue_end:epilogue_start]
    fingerprints, zeroed, fps, suffix_refs, assigned, refs = _prepare_search_data(working)

    run = _find_best_run(working, fingerprints, zeroed, fps, suffix_refs, assigned, refs)
    rolled = False
    if run.trip_count > 0:
        all_runs = _find_all_runs_for_k(working, run.block_size, fingerprints, zeroed, fps, suffix_refs, assigned, refs)
        for r in reversed(all_runs):
            for_node = _build_for(r, working, loop_var)
            actual_start = prologue_end + r.start_idx
            actual_end = actual_start + r.trip_count * r.block_size
            body[actual_start:actual_end] = [for_node]
        rolled = True

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
