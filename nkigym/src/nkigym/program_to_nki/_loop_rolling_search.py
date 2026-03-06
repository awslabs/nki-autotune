"""Search and detection for loop rolling.

Contains fingerprinting, normalization, block matching, spatial run
detection, and reduction chain detection. Used by loop_rolling.py
which handles orchestration and codegen.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

_INT_CONSTANT_RE = re.compile(r"Constant\(value=\d+")
_NAME_RE = re.compile(r"id='[^']*'")
_ALLOC_ATTRS = frozenset({"empty", "zeros", "ones", "full", "empty_like", "zeros_like", "ones_like"})


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


@dataclass
class _ReductionChain:
    """A detected reduction chain with peeled first iteration.

    Block 0 (peel) produces an initial value; blocks 1..N each consume
    the previous block's output via a carried variable (accumulator).
    """

    peel_start: int
    chain_start: int
    block_size: int
    trip_count: int
    varying: list[VaryingConstant]
    carried_input: str
    carried_output: str
    last_output: str


class _SearchData(NamedTuple):
    """Precomputed data structures for the loop rolling search."""

    fingerprints: list[int]
    zeroed: dict[int, str]
    fps: np.ndarray
    suffix_refs: list[set[str]]
    assigned: list[frozenset[str]]
    referenced: list[frozenset[str]]


def _is_alloc_stmt(stmt: ast.stmt) -> bool:
    """Check if a statement is a numpy allocation (empty, zeros, ones, etc.)."""
    return (
        isinstance(stmt, ast.Assign)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and isinstance(stmt.value.func.value, ast.Name)
        and stmt.value.func.value.id == "np"
        and stmt.value.func.attr in _ALLOC_ATTRS
    )


def classify_body_zones(body: list[ast.stmt]) -> tuple[int, int]:
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
    """Normalize a block for structural comparison (positional names, zeroed ints)."""
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


def _collect_int_constants(node: ast.AST) -> list[tuple[tuple[tuple[str, int | None], ...], int]]:
    """Collect all int constants from an AST node in DFS order."""
    results: list[tuple[tuple[tuple[str, int | None], ...], int]] = []
    _walk_constants(node, (), results)
    return results


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
    blocks = [working_stmts[start_idx + i * block_size : start_idx + (i + 1) * block_size] for i in range(trip_count)]
    varying: list[VaryingConstant] = []
    valid = all(_check_stmt_varying(blocks, so, trip_count, varying) for so in range(block_size))
    return (valid, varying)


def _union_assigned(per_assigned: list[frozenset[str]], start: int, end: int) -> set[str]:
    """Union all assignment target names in a statement range."""
    result: set[str] = set()
    for i in range(start, end):
        result |= per_assigned[i]
    return result


def _precompute_name_data(stmts: list[ast.stmt]) -> tuple[list[set[str]], list[frozenset[str]], list[frozenset[str]]]:
    """Precompute suffix reference unions and per-statement name sets.

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
    """Compute upward-exposed uses (referenced before assigned) for a statement block."""
    assigned_so_far: set[str] = set()
    external: set[str] = set()
    for i in range(start, start + block_size):
        external |= per_referenced[i] - assigned_so_far
        assigned_so_far |= per_assigned[i]
    return frozenset(external)


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


def prepare_search_data(working_stmts: list[ast.stmt]) -> _SearchData:
    """Build all precomputed data structures for the loop rolling search."""
    dump_cache: dict[int, str] = {}
    fingerprints = [_fingerprint_stmt(s, dump_cache) for s in working_stmts]
    zeroed_cache = {sid: _INT_CONSTANT_RE.sub("Constant(value=0", v) for sid, v in dump_cache.items()}
    fps = np.array(fingerprints, dtype=np.int64)
    suffix_refs, stmt_assigned, stmt_refs = _precompute_name_data(working_stmts)
    return _SearchData(fingerprints, zeroed_cache, fps, suffix_refs, stmt_assigned, stmt_refs)


def find_all_runs_for_k(working_stmts: list[ast.stmt], block_size: int, data: _SearchData) -> list[_LoopRun]:
    """Find all non-overlapping valid runs for a given block size."""
    n = len(working_stmts)
    match_positions = _block_match_positions(data.fps, block_size)
    runs: list[_LoopRun] = []
    occupied_end = 0
    cache: dict[int, str] = {}
    ext_cache: dict[int, frozenset[str]] = {}
    for p in match_positions:
        if p < occupied_end:
            continue
        if not _ext_names_match(data.assigned, data.referenced, ext_cache, p, block_size):
            continue
        count = _count_matching_blocks(working_stmts, p, block_size, n, cache, data.zeroed, data.fingerprints)
        while count >= 2:
            end = p + count * block_size
            if _union_assigned(data.assigned, p, end) & data.suffix_refs[end]:
                count -= 1
                continue
            valid, varying = _extract_varying(working_stmts, block_size, count, p)
            if valid:
                runs.append(_LoopRun(p, block_size, count, varying))
                occupied_end = end
                break
            count -= 1
    return runs


def _count_fp_blocks(fingerprints: list[int], start: int, block_size: int, n: int) -> int:
    """Count consecutive blocks with matching per-statement fingerprints."""
    count = 1
    while start + (count + 1) * block_size <= n:
        pos = start + count * block_size
        if any(fingerprints[start + i] != fingerprints[pos + i] for i in range(block_size)):
            break
        count += 1
    return count


def _detect_carried_chain(
    per_assigned: list[frozenset[str]], per_referenced: list[frozenset[str]], start: int, block_size: int, count: int
) -> tuple[bool, str, str, str]:
    """Detect a single carried variable across consecutive blocks.

    Returns (valid, carried_input, carried_output, last_output).
    """
    block_exts: list[frozenset[str]] = []
    block_assigns: list[set[str]] = []
    for i in range(count):
        offset = start + i * block_size
        block_exts.append(_block_ext_names(per_assigned, per_referenced, offset, block_size))
        block_assigns.append(_union_assigned(per_assigned, offset, offset + block_size))

    common = block_exts[0]
    for be in block_exts[1:]:
        common = common & be
    varying = [be - common for be in block_exts]

    valid = all(len(v) == 1 for v in varying)
    carried_reads: list[str] = []
    if valid:
        carried_reads = [next(iter(v)) for v in varying]
        valid = all(carried_reads[i] in block_assigns[i - 1] for i in range(1, count))

    carried_write_offset = -1
    if valid:
        for j in range(block_size):
            if carried_reads[1] in per_assigned[start + j]:
                carried_write_offset = j
                break
        valid = carried_write_offset >= 0

    last_output = ""
    if valid:
        last_at = per_assigned[start + (count - 1) * block_size + carried_write_offset]
        valid = len(last_at) == 1
        if valid:
            last_output = next(iter(last_at))

    c_in = carried_reads[0] if carried_reads else ""
    c_out = carried_reads[1] if len(carried_reads) > 1 else ""
    return (valid, c_in, c_out, last_output)


def _validate_chain_peel(
    fingerprints: list[int],
    per_assigned: list[frozenset[str]],
    peel_start: int,
    chain_start: int,
    block_size: int,
    carried_input: str,
) -> bool:
    """Check if block before chain is a valid peel."""
    peel_assigned = _union_assigned(per_assigned, peel_start, chain_start)
    diff = sum(1 for i in range(block_size) if fingerprints[peel_start + i] != fingerprints[chain_start + i])
    return carried_input in peel_assigned and diff <= 1


def _try_chain_at(working_stmts: list[ast.stmt], data: _SearchData, p: int, k: int) -> list[_ReductionChain]:
    """Attempt to build a reduction chain at position p with block size k."""
    n = len(working_stmts)
    count = _count_fp_blocks(data.fingerprints, p, k, n)
    peel_start = p - k
    valid = count >= 2 and peel_start >= 0

    c_in = ""
    c_out = ""
    last = ""
    if valid:
        valid, c_in, c_out, last = _detect_carried_chain(data.assigned, data.referenced, p, k, count)
        valid = valid and _validate_chain_peel(data.fingerprints, data.assigned, peel_start, p, k, c_in)

    varying: list[VaryingConstant] = []
    if valid:
        valid, varying = _extract_varying(working_stmts, k, count, p)

    if valid:
        chain_end = p + count * k
        leaking = (_union_assigned(data.assigned, peel_start, chain_end) & data.suffix_refs[chain_end]) - {last}
        valid = not leaking

    result: list[_ReductionChain] = []
    if valid:
        result = [_ReductionChain(peel_start, p, k, count, varying, c_in, c_out, last)]
    return result


def find_reduction_chain(working_stmts: list[ast.stmt], data: _SearchData) -> list[_ReductionChain]:
    """Find the best reduction chain across all block sizes.

    Returns a list with at most one chain (empty if none found).
    """
    n = len(working_stmts)
    best: list[_ReductionChain] = []
    best_coverage = 0

    for k in range(1, n // 2 + 1):
        if k * (n // k) <= best_coverage:
            continue
        match_positions = _block_match_positions(data.fps, k)
        for p in match_positions:
            chain = _try_chain_at(working_stmts, data, p, k)
            if chain:
                coverage = (chain[0].trip_count + 1) * chain[0].block_size
                if coverage > best_coverage:
                    best = chain
                    best_coverage = coverage

    return best
