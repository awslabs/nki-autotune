"""Operand merge analysis and transform for tiled compute graphs.

Identifies adjacent statement groups that operate on contiguous slices of
the same tensor and can be merged into a single wider operation. This
reduces the total number of load/store/compute statements, improving
hardware utilization.

Example::

    merge = OperandMergeTransform()
    opportunities = merge.analyze(tiled_func)
    for opp in opportunities:
        tiled_func = merge.transform(tiled_func, opp)
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from nkigym.transforms.base import Transform
from nkigym.utils.source import exec_source_to_func, get_source

logger = logging.getLogger(__name__)

TILE_LIMITS: dict[str, dict[str, int]] = {
    "nc_matmul": {"M": 128, "K": 128, "N": 512},
    "activation": {"partition": 128},
    "tensor_scalar": {},
    "tensor_tensor": {},
}
"""Hardware tile size limits per operation type.

Each entry maps an operation name to its maximum tile dimensions.
Merged slices must not exceed these limits on any dimension.

- ``nc_matmul``: free dimension on moving side (N) max 512, stationary
  (M) max 128, contraction (K) max 128.
- ``activation``: partition dimension max 128. Free dimension is limited
  by SBUF partition size but is left unchecked here.
- ``tensor_scalar``: broadcast (P, 1) -- no free-dimension limit.
- ``tensor_tensor``: constrained by operand matching rules, no additional
  numeric limit.
"""


@dataclass
class MergeOpportunity:
    """A single merge opportunity found by ``analyze()``.

    Attributes:
        op_type: The operation type string (e.g., ``"nc_matmul"``,
            ``"load"``, ``"activation"``, ``"tensor_scalar"``,
            ``"tensor_tensor"``).
        stmt_a: The AST statement index (position in function body)
            of the first operation.
        stmt_b: The AST statement index of the second operation
            (to be absorbed).
        differing_operand_idx: Index of the operand that differs between
            the two ops (0-based among the operands). For loads this is
            the slice dimension index that differs.
        merged_slice: A ``(start, stop)`` tuple for the merged
            free-dimension range of the differing operand after
            merging.
        description: Human-readable description for logging.
    """

    op_type: str
    stmt_a: int
    stmt_b: int
    differing_operand_idx: int
    merged_slice: tuple[int, int]
    description: str


def _parse_slice(node: ast.Slice) -> tuple[int, int]:
    """Extract ``(start, stop)`` from an AST Slice node.

    The tiled IR always uses ``ast.Constant`` for start and stop values.

    Args:
        node: An AST Slice node with constant start/stop.

    Returns:
        Tuple of ``(start, stop)`` integers.

    Raises:
        ValueError: If start or stop is not an ``ast.Constant``.
    """
    if not isinstance(node.lower, ast.Constant) or not isinstance(node.upper, ast.Constant):
        raise ValueError(f"Expected constant slice bounds, got {ast.dump(node)}")
    return (node.lower.value, node.upper.value)


def _parse_subscript_slices(node: ast.Subscript) -> list[tuple[int, int]]:
    """Extract all dimension slices from a subscript expression.

    Handles both single-dimension (``tensor[0:128]``) and
    multi-dimension (``tensor[0:128, 128:256]``) subscripts.

    Args:
        node: An AST Subscript node.

    Returns:
        List of ``(start, stop)`` tuples, one per dimension.

    Raises:
        ValueError: If the slice structure is unexpected.
    """
    slice_node = node.slice
    if isinstance(slice_node, ast.Tuple):
        return [_parse_slice(elt) for elt in slice_node.elts]
    if isinstance(slice_node, ast.Slice):
        return [_parse_slice(slice_node)]
    raise ValueError(f"Unexpected subscript slice type: {ast.dump(slice_node)}")


def _get_nkigym_func_name(call_node: ast.Call) -> str | None:
    """Extract the function name from a ``nkigym.X(...)`` call.

    Args:
        call_node: An AST Call node.

    Returns:
        The function name (e.g., ``"nc_matmul"``, ``"ndarray"``),
        or ``None`` if the call is not to ``nkigym.<name>``.
    """
    func = call_node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "nkigym":
        return func.attr
    return None


def _extract_arg_info(arg: ast.expr) -> dict:
    """Extract information from a call argument.

    For Name nodes, returns the variable name. For Subscript nodes,
    returns the variable name and its slices.

    Args:
        arg: An AST expression node (argument to a call).

    Returns:
        Dict with ``"name"`` and optionally ``"slices"`` keys.
    """
    if isinstance(arg, ast.Name):
        return {"name": arg.id, "slices": None}
    if isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name):
        return {"name": arg.value.id, "slices": _parse_subscript_slices(arg)}
    return {"name": None, "slices": None}


def _classify_stmt(stmt: ast.stmt, idx: int) -> dict:
    """Classify a single AST statement from the tiled IR.

    Determines the statement type and extracts relevant fields.

    Args:
        stmt: An AST statement node.
        idx: The statement's index in the function body.

    Returns:
        Dict with keys:
            - ``"type"``: one of ``"alloc"``, ``"load"``, ``"op"``,
              ``"store"``, ``"aug_assign"``, ``"return"``, ``"other"``
            - ``"idx"``: the statement index
            - Additional keys depending on type (see below).

        For ``"load"``:
            ``"var"``, ``"source_tensor"``, ``"slices"``

        For ``"alloc"``:
            ``"var"``

        For ``"op"``:
            ``"var"``, ``"op_name"``, ``"args"``, ``"kwargs"``

        For ``"store"``:
            ``"target"``, ``"target_slices"``, ``"value_var"``

        For ``"aug_assign"``:
            ``"var"``, ``"op_name"``, ``"args"``, ``"kwargs"``
    """
    result: dict = {"type": "other", "idx": idx}

    if isinstance(stmt, ast.Return):
        result["type"] = "return"
        return result

    if isinstance(stmt, ast.AugAssign):
        result["type"] = "aug_assign"
        if isinstance(stmt.target, ast.Name):
            result["var"] = stmt.target.id
        if isinstance(stmt.value, ast.Call):
            op_name = _get_nkigym_func_name(stmt.value)
            if op_name:
                result["op_name"] = op_name
                result["args"] = [_extract_arg_info(a) for a in stmt.value.args]
                result["kwargs"] = {kw.arg: kw.value for kw in stmt.value.keywords if kw.arg is not None}
        return result

    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return result

    target = stmt.targets[0]
    value = stmt.value

    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
        result["type"] = "store"
        result["target"] = target.value.id
        result["target_slices"] = _parse_subscript_slices(target)
        if isinstance(value, ast.Name):
            result["value_var"] = value.id
        return result

    if not isinstance(target, ast.Name):
        return result

    var_name = target.id

    if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
        result["type"] = "load"
        result["var"] = var_name
        result["source_tensor"] = value.value.id
        result["slices"] = _parse_subscript_slices(value)
        return result

    if isinstance(value, ast.Call):
        op_name = _get_nkigym_func_name(value)
        if op_name == "ndarray":
            result["type"] = "alloc"
            result["var"] = var_name
            return result
        if op_name:
            result["type"] = "op"
            result["var"] = var_name
            result["op_name"] = op_name
            result["args"] = [_extract_arg_info(a) for a in value.args]
            result["kwargs"] = {kw.arg: kw.value for kw in value.keywords if kw.arg is not None}
            return result

    return result


def _parse_function_body(func: Callable) -> tuple[str, list[str], dict[str, list[dict]]]:
    """Parse a tiled function's source into classified statements.

    Args:
        func: A tiled function (with ``__source__`` attribute).

    Returns:
        Tuple of ``(func_name, param_names, classified)`` where
        ``classified`` is a dict keyed by statement type
        (``"load"``, ``"op"``, ``"store"``, ``"aug_assign"``, etc.)
        plus ``"all"`` for the flat positional list.

    Raises:
        ValueError: If no function definition is found in the source.
    """
    source = get_source(func)
    tree = ast.parse(source)

    func_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    if not func_defs:
        raise ValueError("No function definition found in source")

    func_def = func_defs[0]
    func_name = func_def.name
    param_names = [arg.arg for arg in func_def.args.args]
    all_stmts = [_classify_stmt(stmt, idx) for idx, stmt in enumerate(func_def.body)]

    by_type: dict[str, list[dict]] = {"all": all_stmts}
    for s in all_stmts:
        by_type.setdefault(s["type"], []).append(s)

    return func_name, param_names, by_type


def _greedy_pair_opportunities(
    groups: dict[Any, list[dict]],
    check_pair: Callable[[dict, dict], MergeOpportunity | None],
    sort_key: Callable[[dict], Any] | None = None,
) -> list[MergeOpportunity]:
    """Greedily pair grouped statements into merge opportunities.

    Iterates each group and pairs statements using a first-match greedy
    strategy. Once a statement is paired it is excluded from further
    consideration within the same ``analyze()`` call.

    Args:
        groups: Mapping from group key to list of classified statement dicts.
        check_pair: Callable that receives two candidate statements and
            returns a ``MergeOpportunity`` if they can be merged, or
            ``None`` to skip.
        sort_key: Optional key function to sort each group before pairing.

    Returns:
        List of ``MergeOpportunity`` objects produced by ``check_pair``.
    """
    used_stmts: set[int] = set()
    opportunities: list[MergeOpportunity] = []

    for members in groups.values():
        if len(members) < 2:
            continue

        ordered = sorted(members, key=sort_key) if sort_key else members

        for i in range(len(ordered)):
            if ordered[i]["idx"] in used_stmts:
                continue

            for j in range(i + 1, len(ordered)):
                if ordered[j]["idx"] in used_stmts:
                    continue

                opp = check_pair(ordered[i], ordered[j])
                if opp is None:
                    continue

                opportunities.append(opp)
                used_stmts.add(opp.stmt_a)
                used_stmts.add(opp.stmt_b)
                break

    return opportunities


def _find_merge_opportunities(classified: dict[str, list[dict]]) -> list[MergeOpportunity]:
    """Find all merge opportunities (load and op) in classified statements.

    Scans loads for adjacent pairs that can be merged into wider loads,
    then scans ops for pairs that differ on exactly one subscripted
    argument with adjacent slices.

    Args:
        classified: Dict keyed by statement type from
            ``_parse_function_body()``.

    Returns:
        List of ``MergeOpportunity`` objects for both load and op
        merging.
    """
    all_stmts = classified["all"]
    loads = classified.get("load", [])
    stores = classified.get("store", [])
    aug_assigns = classified.get("aug_assign", [])

    opportunities: list[MergeOpportunity] = []

    if len(loads) >= 2:
        load_groups: dict[tuple, list[dict]] = {}
        for load in loads:
            slices = load["slices"]
            if len(slices) < 2:
                continue
            partition_key = (load["source_tensor"], slices[0])
            load_groups.setdefault(partition_key, []).append(load)

        def check_load_pair(load_a: dict, load_b: dict) -> MergeOpportunity | None:
            adj = _check_adjacent_slices(load_a["slices"], load_b["slices"])
            if adj is None:
                return None

            dim, merged = adj

            stmt_first = load_a if load_a["idx"] < load_b["idx"] else load_b
            stmt_second = load_b if load_a["idx"] < load_b["idx"] else load_a

            if not _check_dependency_safe(stmt_first["idx"], stmt_second["idx"], stmt_second["var"], all_stmts):
                return None

            return MergeOpportunity(
                op_type="load",
                stmt_a=stmt_first["idx"],
                stmt_b=stmt_second["idx"],
                differing_operand_idx=dim,
                merged_slice=merged,
                description=(
                    f"Merge load {stmt_first['var']} and {stmt_second['var']} "
                    f"from {stmt_first['source_tensor']}[dim {dim}: "
                    f"{load_a['slices'][dim]} + {load_b['slices'][dim]} -> {merged}]"
                ),
            )

        opportunities.extend(
            _greedy_pair_opportunities(load_groups, check_load_pair, sort_key=lambda ld: ld["slices"][-1][0])
        )

    mergeable_ops = {"nc_matmul", "tensor_tensor", "activation", "tensor_scalar"}
    ops = [s for s in classified.get("op", []) if s.get("op_name") in mergeable_ops]
    if len(ops) >= 2:
        op_groups: dict[str, list[dict]] = {}
        for op in ops:
            op_groups.setdefault(op["op_name"], []).append(op)

        def check_op_pair(op_a: dict, op_b: dict) -> MergeOpportunity | None:
            args_a = op_a.get("args", [])
            args_b = op_b.get("args", [])

            if len(args_a) != len(args_b):
                return None

            if not _kwargs_equal(op_a.get("kwargs", {}), op_b.get("kwargs", {})):
                return None

            differing_args = [
                k
                for k in range(len(args_a))
                if _resolve_arg_signature(args_a[k], loads) != _resolve_arg_signature(args_b[k], loads)
            ]
            if len(differing_args) != 1:
                return None

            diff_idx = differing_args[0]
            arg_a = args_a[diff_idx]
            arg_b = args_b[diff_idx]

            slices_a = arg_a["slices"]
            slices_b = arg_b["slices"]

            if slices_a is None:
                load_a = _resolve_arg_to_load(arg_a, loads)
                if load_a:
                    slices_a = load_a["slices"]
            if slices_b is None:
                load_b = _resolve_arg_to_load(arg_b, loads)
                if load_b:
                    slices_b = load_b["slices"]

            if slices_a is None or slices_b is None:
                return None

            adj = _check_adjacent_slices(slices_a, slices_b)
            if adj is None:
                return None

            dim, merged = adj
            op_name = op_a["op_name"]

            if op_name == "nc_matmul":
                if not _check_nc_matmul_limits(diff_idx, merged):
                    return None

            store_a = _find_store_for_var(stores, op_a["var"])
            store_b = _find_store_for_var(stores, op_b["var"])
            if store_a and store_b:
                s_adj = _check_adjacent_slices(store_a["target_slices"], store_b["target_slices"])
                if s_adj is None:
                    return None

            if _has_aug_assign_accumulation(op_a, aug_assigns) or _has_aug_assign_accumulation(op_b, aug_assigns):
                return None

            stmt_first = op_a if op_a["idx"] < op_b["idx"] else op_b
            stmt_second = op_b if op_a["idx"] < op_b["idx"] else op_a

            if not _check_dependency_safe(stmt_first["idx"], stmt_second["idx"], stmt_second.get("var", ""), all_stmts):
                return None

            return MergeOpportunity(
                op_type=op_name,
                stmt_a=stmt_first["idx"],
                stmt_b=stmt_second["idx"],
                differing_operand_idx=diff_idx,
                merged_slice=merged,
                description=(
                    f"Merge {op_name} at stmt {stmt_first['idx']} and "
                    f"{stmt_second['idx']} on arg {diff_idx} "
                    f"[dim {dim}: {slices_a[dim]} + "
                    f"{slices_b[dim]} -> {merged}]"
                ),
            )

        opportunities.extend(_greedy_pair_opportunities(op_groups, check_op_pair))

    return opportunities


def _check_dependency_safe(idx_lo: int, idx_hi: int, absorbed_var: str, all_stmts: list[dict]) -> bool:
    """Check that merging two statements does not violate data dependencies.

    The absorbed statement's variable must not be used by any statement
    between index ``idx_lo`` and ``idx_hi`` (exclusive on both ends).

    Args:
        idx_lo: Statement index of the kept (earlier) statement.
        idx_hi: Statement index of the absorbed (later) statement.
        absorbed_var: Variable name produced by the absorbed statement.
        all_stmts: Full flat list of classified statements.

    Returns:
        True if merging is safe, False otherwise.
    """
    for stmt in all_stmts:
        if stmt["idx"] <= idx_lo or stmt["idx"] >= idx_hi:
            continue
        if _stmt_uses_var(stmt, absorbed_var):
            return False

    return True


def _stmt_uses_var(stmt: dict, var: str) -> bool:
    """Check if a classified statement uses a given variable.

    Args:
        stmt: A classified statement dict.
        var: The variable name to check for.

    Returns:
        True if the statement references the variable.
    """
    stype = stmt["type"]

    if stype in ("op", "aug_assign"):
        return any(arg.get("name") == var for arg in stmt.get("args", []))

    if stype == "store":
        return stmt.get("value_var") == var

    if stype == "load":
        return stmt.get("source_tensor") == var

    return False


def _resolve_arg_signature(arg_info: dict, loads: list[dict]) -> tuple:
    """Create a hashable signature resolving Name args through loads.

    If the argument is a bare Name referencing a load, uses the load's
    ``(source_tensor, slices)`` as the signature for comparison.

    Args:
        arg_info: Dict from ``_extract_arg_info``.
        loads: Classified load statements.

    Returns:
        A hashable tuple representing the resolved argument identity.
    """
    if arg_info["slices"] is not None:
        return (arg_info["name"], tuple(arg_info["slices"]))
    load = _resolve_arg_to_load(arg_info, loads)
    if load is not None:
        return ("load", load["source_tensor"], tuple(load["slices"]))
    return (arg_info["name"],)


def _kwargs_equal(kw_a: dict, kw_b: dict) -> bool:
    """Check if two keyword argument dicts are structurally equal.

    Compares AST node dumps for value equality since AST nodes
    are not directly comparable.

    Args:
        kw_a: First keyword argument dict (arg_name -> ast node).
        kw_b: Second keyword argument dict (arg_name -> ast node).

    Returns:
        True if all keyword arguments match.
    """
    if set(kw_a.keys()) != set(kw_b.keys()):
        return False
    return all(ast.dump(kw_a[key]) == ast.dump(kw_b[key]) for key in kw_a)


def _check_adjacent_slices(
    slices_a: list[tuple[int, int]], slices_b: list[tuple[int, int]]
) -> tuple[int, tuple[int, int]] | None:
    """Check if two slice lists differ on exactly one dimension with adjacent ranges.

    Args:
        slices_a: Slice list from the first subscript argument.
        slices_b: Slice list from the second subscript argument.

    Returns:
        Tuple of ``(differing_dim_index, merged_slice)`` if adjacent,
        or ``None`` if not mergeable.
    """
    if len(slices_a) != len(slices_b):
        return None

    differing_dims = []
    for d in range(len(slices_a)):
        if slices_a[d] != slices_b[d]:
            differing_dims.append(d)

    if len(differing_dims) != 1:
        return None

    dim = differing_dims[0]
    sa_start, sa_stop = slices_a[dim]
    sb_start, sb_stop = slices_b[dim]

    if sa_stop == sb_start:
        return (dim, (sa_start, sb_stop))
    if sb_stop == sa_start:
        return (dim, (sb_start, sa_stop))
    return None


def _check_nc_matmul_limits(differing_arg_idx: int, merged_slice: tuple[int, int]) -> bool:
    """Check that a merged nc_matmul does not exceed hardware tile limits.

    For nc_matmul(lhs[K,M], rhs[K,N]):
    - If differing arg is rhs (idx 1), the merged N dimension must be <= 512.
    - If differing arg is lhs (idx 0), the merged M dimension must be <= 128.

    Args:
        differing_arg_idx: Which positional arg differs (0=lhs, 1=rhs).
        merged_slice: The ``(start, stop)`` of the merged range.

    Returns:
        True if within hardware limits.
    """
    limits = TILE_LIMITS["nc_matmul"]
    merged_size = merged_slice[1] - merged_slice[0]

    if differing_arg_idx == 1:
        return merged_size <= limits["N"]
    if differing_arg_idx == 0:
        return merged_size <= limits["M"]
    return True


def _has_aug_assign_accumulation(op: dict, aug_assigns: list[dict]) -> bool:
    """Check if an op's result variable is the target of an aug_assign.

    When an op has ``+=`` accumulations (from reduction tiling), merging the
    initial assignment without also merging the accumulations would produce
    shape mismatches.

    Args:
        op: Classified dict for an op statement.
        aug_assigns: Classified aug_assign statements.

    Returns:
        True if the op's result variable is accumulated via aug_assign.
    """
    var = op.get("var")
    if not var:
        return False
    return any(s.get("var") == var for s in aug_assigns)


def _resolve_arg_to_load(arg_info: dict, loads: list[dict]) -> dict | None:
    """Resolve a Name argument to its defining load statement.

    Args:
        arg_info: Argument dict from ``_extract_arg_info``.
        loads: Classified load statements.

    Returns:
        The load statement dict, or ``None`` if the arg does not
        reference a load variable.
    """
    name = arg_info.get("name")
    if name is None:
        return None
    for s in loads:
        if s["var"] == name:
            return s
    return None


def _find_store_for_var(stores: list[dict], var: str) -> dict | None:
    """Find the store statement that writes a variable to output.

    Args:
        stores: Classified store statements.
        var: Variable name to look for.

    Returns:
        The store statement dict, or ``None``.
    """
    for s in stores:
        if s.get("value_var") == var:
            return s
    return None


def _update_subscript_dim(node: ast.Subscript, dim: int, merged: tuple[int, int]) -> None:
    """Update a single dimension of a subscript's slice range.

    Args:
        node: An AST Subscript node to modify in-place.
        dim: The dimension index to update.
        merged: ``(start, stop)`` for the new slice range.
    """
    slice_node = node.slice
    if isinstance(slice_node, ast.Tuple):
        sl = slice_node.elts[dim]
        if isinstance(sl, ast.Slice):
            sl.lower = ast.Constant(value=merged[0])
            sl.upper = ast.Constant(value=merged[1])
    elif isinstance(slice_node, ast.Slice) and dim == 0:
        slice_node.lower = ast.Constant(value=merged[0])
        slice_node.upper = ast.Constant(value=merged[1])


def _relative_slices(abs_slices: list[tuple[int, int]], merge_dim: int, merged_start: int) -> list[tuple[int, int]]:
    """Convert absolute source-tensor slices to relative loaded-tensor slices.

    After a load ``var = source[s0:e0, s1:e1]``, the loaded tensor ``var``
    has shape ``[e0-s0, e1-s1]`` and indices start from 0. Consumers that
    subscript into ``var`` must use relative indices, not the absolute
    source-tensor indices.

    Args:
        abs_slices: Absolute slices from the classified load statement.
        merge_dim: The dimension being merged (widened).
        merged_start: The start of the merged range on ``merge_dim``.

    Returns:
        List of ``(start, stop)`` tuples relative to the loaded tensor.
    """
    return [
        (s - merged_start, e - merged_start) if d == merge_dim else (0, e - s) for d, (s, e) in enumerate(abs_slices)
    ]


def _build_slices_node(slices: list[tuple[int, int]]) -> ast.expr:
    """Build an AST slice expression from a list of ``(start, stop)`` tuples.

    Args:
        slices: List of ``(start, stop)`` tuples, one per dimension.

    Returns:
        An AST node suitable for use as a Subscript slice: a single
        ``ast.Slice`` if one dimension, or an ``ast.Tuple`` of slices.
    """
    slice_nodes = [ast.Slice(lower=ast.Constant(value=s), upper=ast.Constant(value=e)) for s, e in slices]
    if len(slice_nodes) == 1:
        return slice_nodes[0]
    return ast.Tuple(elts=slice_nodes, ctx=ast.Load())


def _subscript_bare_consumers(stmt: ast.stmt, var: str, slices: list[tuple[int, int]]) -> None:
    """Replace bare Name references to ``var`` in Call args with Subscript expressions.

    Walks the AST statement and replaces every bare ``ast.Name(id=var)``
    that appears as a positional argument in a ``Call`` node with
    ``ast.Subscript(value=ast.Name(id=var), slice=<slices>)``.

    Args:
        stmt: An AST statement node to modify in-place.
        var: The variable name to subscript.
        slices: The original slice ranges to use for the subscript.
    """
    for node in ast.walk(stmt):
        if not isinstance(node, ast.Call):
            continue
        for idx, arg in enumerate(node.args):
            if isinstance(arg, ast.Name) and arg.id == var:
                node.args[idx] = ast.Subscript(
                    value=ast.Name(id=var, ctx=ast.Load()), slice=_build_slices_node(slices), ctx=ast.Load()
                )


def _rename_vars_in_stmt(stmt: ast.stmt, renames: dict[str, str]) -> None:
    """Rename all variable references in a statement.

    Args:
        stmt: An AST statement node to modify in-place.
        renames: Mapping from old variable names to new names.
    """
    if not renames:
        return
    for node in ast.walk(stmt):
        if isinstance(node, ast.Name) and node.id in renames:
            node.id = renames[node.id]


class OperandMergeTransform(Transform):
    """Transform that merges adjacent operations on contiguous tensor slices.

    Identifies pairs of statements that perform the same operation but on
    adjacent slices of a tensor, differing on exactly one operand dimension.
    These can be combined into a single wider operation without exceeding
    hardware tile limits.

    ``analyze()`` returns a list of ``MergeOpportunity`` objects.
    ``transform()`` applies one opportunity at a time.

    Example::

        merge = OperandMergeTransform()
        opportunities = merge.analyze(tiled_func)
        for opp in opportunities:
            tiled_func = merge.transform(tiled_func, opp)
    """

    name = "operand_merge"

    def analyze(self, func: Callable) -> list[MergeOpportunity]:
        """Analyze a tiled function to find merge opportunities.

        Parses the IR source, groups statements by their structure, and
        identifies pairs that operate on contiguous slices differing on
        exactly one dimension.

        Args:
            func: A tiled function (with ``__source__`` attribute) to analyze.

        Returns:
            List of ``MergeOpportunity`` objects, each representing a
            single merge that can be passed to ``transform()``.
        """
        _, _, classified = _parse_function_body(func)
        return _find_merge_opportunities(classified)

    def transform(self, func: Callable, option: MergeOpportunity) -> Callable[..., np.ndarray]:
        """Apply a single merge opportunity.

        Merges the absorbed statement into the kept statement by widening
        the slice on the differing dimension and removing the absorbed
        statement and its associated load/store chain.

        Args:
            func: A tiled function to transform.
            option: A single ``MergeOpportunity`` from ``analyze()``.

        Returns:
            New callable with the merged operation applied.
        """
        func_name, _, classified = _parse_function_body(func)
        all_stmts = classified["all"]
        loads = classified.get("load", [])
        stores = classified.get("store", [])

        source = get_source(func)
        tree = ast.parse(source)

        func_def = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))

        stmt_a = all_stmts[option.stmt_a]
        stmt_b = all_stmts[option.stmt_b]

        to_remove: set[int] = {option.stmt_b}
        renames: dict[str, str] = {stmt_b["var"]: stmt_a["var"]}

        subscript_map: dict[str, list[tuple[int, int]]] = {}
        if option.op_type == "load":
            merge_dim = option.differing_operand_idx
            merged_start = option.merged_slice[0]
            subscript_map[stmt_a["var"]] = _relative_slices(stmt_a["slices"], merge_dim, merged_start)
            subscript_map[stmt_b["var"]] = _relative_slices(stmt_b["slices"], merge_dim, merged_start)
        else:
            arg_b = stmt_b["args"][option.differing_operand_idx]
            if arg_b["slices"] is None:
                load_b = _resolve_arg_to_load(arg_b, loads)
                if load_b:
                    to_remove.add(load_b["idx"])
            store_b = _find_store_for_var(stores, stmt_b["var"])
            if store_b:
                to_remove.add(store_b["idx"])

        store_a = _find_store_for_var(stores, stmt_a["var"])
        store_b_info = _find_store_for_var(stores, stmt_b["var"]) if option.op_type != "load" else None

        widen_load_idx: int | None = None
        widen_load_dim: int | None = None
        if option.op_type != "load":
            arg_a_info = stmt_a["args"][option.differing_operand_idx]
            arg_b_info = stmt_b["args"][option.differing_operand_idx]
            if arg_a_info["slices"] is None:
                load_a = _resolve_arg_to_load(arg_a_info, loads)
                load_b = _resolve_arg_to_load(arg_b_info, loads)
                if load_a and load_b:
                    adj = _check_adjacent_slices(load_a["slices"], load_b["slices"])
                    if adj:
                        widen_load_idx = load_a["idx"]
                        widen_load_dim = adj[0]

        new_body = []
        for i, body_stmt in enumerate(func_def.body):
            if i in to_remove:
                continue

            if i == option.stmt_a:
                self._widen_kept_stmt(body_stmt, option, all_stmts)

            if widen_load_idx is not None and i == widen_load_idx and widen_load_dim is not None:
                if isinstance(body_stmt, ast.Assign) and isinstance(body_stmt.value, ast.Subscript):
                    _update_subscript_dim(body_stmt.value, widen_load_dim, option.merged_slice)

            if store_a and i == store_a["idx"] and store_b_info:
                self._widen_kept_store(body_stmt, store_a, store_b_info)

            for var, slices in subscript_map.items():
                _subscript_bare_consumers(body_stmt, var, slices)

            _rename_vars_in_stmt(body_stmt, renames)
            new_body.append(body_stmt)

        func_def.body = new_body
        ast.fix_missing_locations(tree)
        new_source = ast.unparse(tree)
        logger.debug("Merged source:\n%s", new_source)
        return exec_source_to_func(new_source, func_name)

    def _widen_kept_stmt(self, body_stmt: ast.stmt, option: MergeOpportunity, all_stmts: list[dict]) -> None:
        """Widen the kept statement's slice to the merged range.

        Args:
            body_stmt: The AST statement node to modify.
            option: The merge opportunity.
            all_stmts: Flat list of all classified statements.
        """
        if option.op_type == "load":
            if isinstance(body_stmt, ast.Assign) and isinstance(body_stmt.value, ast.Subscript):
                _update_subscript_dim(body_stmt.value, option.differing_operand_idx, option.merged_slice)
            return

        if not isinstance(body_stmt, ast.Assign) or not isinstance(body_stmt.value, ast.Call):
            return

        diff_arg = body_stmt.value.args[option.differing_operand_idx]
        stmt_a = all_stmts[option.stmt_a]
        stmt_b = all_stmts[option.stmt_b]

        if isinstance(diff_arg, ast.Subscript):
            arg_a_info = stmt_a["args"][option.differing_operand_idx]
            arg_b_info = stmt_b["args"][option.differing_operand_idx]
            slices_a = arg_a_info["slices"]
            slices_b = arg_b_info["slices"]
            if slices_a and slices_b:
                for d in range(len(slices_a)):
                    if slices_a[d] != slices_b[d]:
                        _update_subscript_dim(diff_arg, d, option.merged_slice)
                        break

    @staticmethod
    def _widen_kept_store(body_stmt: ast.stmt, store_a: dict, store_b: dict) -> None:
        """Widen the kept store's target slice.

        Args:
            body_stmt: The AST store statement to modify.
            store_a: Classified dict for the kept store.
            store_b: Classified dict for the absorbed store.
        """
        if not isinstance(body_stmt, ast.Assign):
            return
        target = body_stmt.targets[0]
        if not isinstance(target, ast.Subscript):
            return

        s_a = store_a["target_slices"]
        s_b = store_b["target_slices"]
        for d in range(len(s_a)):
            if s_a[d] != s_b[d]:
                merged_store = (min(s_a[d][0], s_b[d][0]), max(s_a[d][1], s_b[d][1]))
                _update_subscript_dim(target, d, merged_store)
                break


_default_transform = OperandMergeTransform()


def analyze_operand_merge(func: Callable) -> list[MergeOpportunity]:
    """Find operand merge opportunities in a tiled function.

    Convenience wrapper around ``OperandMergeTransform.analyze()``.

    Args:
        func: A tiled function with ``__source__`` attribute.

    Returns:
        List of ``MergeOpportunity`` objects.
    """
    return _default_transform.analyze(func)
