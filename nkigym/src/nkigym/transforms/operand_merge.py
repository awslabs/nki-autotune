"""Operand merge analysis and transform for tiled compute graphs.

Identifies adjacent statement groups that operate on contiguous slices of
the same tensor and can be merged into a single wider operation. This
reduces the total number of load/store/compute statements, improving
hardware utilization.

The algorithm is uniform across all op types. Each ``GymOp`` declares its
inputs/outputs with named axes and a ``tile_limits`` dict. The merge
algorithm uses these to determine which dimensions can widen and by how
much.

Example::

    merge = OperandMergeTransform()
    while True:
        opportunities = merge.analyze_ir(program)
        if not opportunities:
            break
        program = merge.transform_ir(program, opportunities[0])
"""

from typing import Any

from nkigym.ir import GymProgram, GymStatement, TensorRef
from nkigym.transforms.base import Transform
from nkigym.transforms.merge_utils import (
    MergeOpportunity,
    StmtIndex,
    _axis_dim,
    _check_adjacent_slices,
    _remap_stmt_refs,
    _rename_stmt_refs,
    _replace_stmts,
    _widen_slice,
)


def _stmt_group_key(stmt: GymStatement, index: StmtIndex) -> tuple:
    """Compute a grouping key for merge candidate grouping.

    Statements with the same key are potential merge pairs. Uses canonical
    names (resolving through loads) so that statements referencing identical
    loads are grouped together.

    Args:
        stmt: A GymStatement to compute the key for.
        index: Pre-built statement index for load resolution.

    Returns:
        Hashable grouping key.
    """
    config = tuple((k, v) for k, v in stmt.kwargs if not isinstance(v, TensorRef) and k != "acc")
    tensor_names = tuple(
        (k, index.canonical_name(v)) for k, v in stmt.kwargs if isinstance(v, TensorRef) and k != "acc"
    )
    return (stmt.op, config, tensor_names)


def _acc_compatible(stmt_a: GymStatement, stmt_b: GymStatement) -> bool:
    """Check that acc kwargs match (both absent or identical refs).

    Args:
        stmt_a: First statement.
        stmt_b: Second statement.

    Returns:
        True if acc kwargs are compatible for merging.
    """
    acc_a = dict(stmt_a.kwargs).get("acc")
    acc_b = dict(stmt_b.kwargs).get("acc")
    return acc_a == acc_b


def _collect_diffs(
    stmt_a: GymStatement, stmt_b: GymStatement, index: StmtIndex
) -> list[tuple[int, int, int, tuple[int, int]]]:
    """Collect differing TensorRef kwargs between two statements.

    For same-name refs, compares direct slices for adjacency.
    For different-name refs, resolves through loads and requires identical
    resolved slices (equivalent operands, not a diff).

    Args:
        stmt_a: First statement.
        stmt_b: Second statement.
        index: Pre-built statement index for load resolution.

    Returns:
        List of (kwarg_idx, tensor_pos, dim, merged_range) tuples.
        Empty list if the pair is invalid or has no diffs.
    """
    tensor_pos = 0
    diffs: list[tuple[int, int, int, tuple[int, int]]] = []
    valid = True
    for i, ((ka, va), (_, vb)) in enumerate(zip(stmt_a.kwargs, stmt_b.kwargs)):
        if not isinstance(va, TensorRef) or ka == "acc":
            continue
        if va.name == vb.name:
            adj = _check_adjacent_slices(va.slices, vb.slices)
            if adj[0] >= 0:
                diffs.append((i, tensor_pos, adj[0], adj[1]))
            elif va.slices != vb.slices:
                valid = False
        elif index.resolve_operand_slices(va) != index.resolve_operand_slices(vb):
            valid = False
        tensor_pos += 1
    return diffs if valid else []


def _diffs_to_axis(diffs: list[tuple[int, int, int, tuple[int, int]]], op_cls: type) -> str:
    """Map diffs to named axes and validate same-axis + tile limits.

    Args:
        diffs: List of (kwarg_idx, tensor_pos, dim, merged_range) from
            ``_collect_diffs``.
        op_cls: The GymOp subclass for the statements.

    Returns:
        The merge axis name, or empty string if invalid.
    """
    axes: set[str] = set()
    ok = True
    for _, tensor_pos, dim, merged in diffs:
        axis = op_cls.inputs[tensor_pos].axes[dim]
        if not isinstance(axis, str):
            ok = False
            break
        merged_size = merged[1] - merged[0]
        if not op_cls.can_merge_operand_dim(tensor_pos, dim, merged_size):
            ok = False
            break
        axes.add(axis)
    result = ""
    if ok and len(axes) == 1:
        result = axes.pop()
    return result


def _check_pair(
    stmt_a: GymStatement, stmt_b: GymStatement, idx_a: int, idx_b: int, index: StmtIndex
) -> list[MergeOpportunity]:
    """Validate whether two statements can be merged.

    Checks acc compatibility, collects operand diffs, validates same-axis
    mapping and tile limits.

    Args:
        stmt_a: First statement.
        stmt_b: Second statement.
        idx_a: Statement index of stmt_a.
        idx_b: Statement index of stmt_b.
        index: Pre-built statement index.

    Returns:
        Single-element list with the opportunity, or empty list.
    """
    result: list[MergeOpportunity] = []
    if _acc_compatible(stmt_a, stmt_b):
        diffs = _collect_diffs(stmt_a, stmt_b, index)
        merge_axis = _diffs_to_axis(diffs, stmt_a.op) if diffs else ""
        if merge_axis:
            first = min(idx_a, idx_b)
            second = max(idx_a, idx_b)
            kwarg_idx, _, dim, merged = diffs[0]
            result.append(
                MergeOpportunity(
                    stmt_a=first,
                    stmt_b=second,
                    differing_operand_idx=kwarg_idx,
                    differing_dim=dim,
                    merged_slice=merged,
                    merge_axis=merge_axis,
                    description=f"Merge {stmt_a.op.op_name} at {first} and {second} on axis {merge_axis}",
                )
            )
    return result


def _find_merge_opportunities(index: StmtIndex) -> list[MergeOpportunity]:
    """Find all merge opportunities in a single pass.

    Groups statements by op type, config kwargs, and canonical tensor
    names. Within each group, checks pairs for adjacent operand slices
    that map to the same named axis within tile limits.

    Args:
        index: Pre-built statement index.

    Returns:
        List of MergeOpportunity objects.
    """
    groups: dict[tuple, list[tuple[int, GymStatement]]] = {}
    for i, stmt in enumerate(index.stmts):
        key = _stmt_group_key(stmt, index)
        groups.setdefault(key, []).append((i, stmt))

    used: set[int] = set()
    opportunities: list[MergeOpportunity] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            if members[i][0] in used:
                continue
            for j in range(i + 1, len(members)):
                if members[j][0] in used:
                    continue
                opps = _check_pair(members[i][1], members[j][1], members[i][0], members[j][0], index)
                if opps:
                    opportunities.extend(opps)
                    used.add(opps[0].stmt_a)
                    used.add(opps[0].stmt_b)
                    break

    return opportunities


def _widen_kwarg(ref_a: TensorRef, ref_b: TensorRef, merge_dim: int) -> TensorRef:
    """Widen a single TensorRef kwarg on the merge dimension.

    If the ref shape matches its slice extents (tile ref), the shape
    is updated to the new extents. Otherwise (buffer ref), the shape
    is preserved.

    Args:
        ref_a: TensorRef from the kept statement.
        ref_b: TensorRef from the absorbed statement.
        merge_dim: Dimension to widen.

    Returns:
        Widened TensorRef.
    """
    widened = (
        min(ref_a.slices[merge_dim][0], ref_b.slices[merge_dim][0]),
        max(ref_a.slices[merge_dim][1], ref_b.slices[merge_dim][1]),
    )
    new_slices = _widen_slice(ref_a.slices, merge_dim, widened)
    old_extents = tuple(e - s for s, e in ref_a.slices)
    new_shape = tuple(e - s for s, e in new_slices) if ref_a.shape == old_extents else ref_a.shape
    return TensorRef(ref_a.name, new_shape, new_slices)


def _widen_output(stmt_a: GymStatement, stmt_b: GymStatement, merge_axis: str) -> TensorRef:
    """Compute the widened output TensorRef for the merged statement.

    For SSA merges (different output names), the new shape is the sum
    of both shapes on the merge dimension. For buffer merges (same
    output name), the shape is preserved and slices are widened.

    Args:
        stmt_a: Kept statement.
        stmt_b: Absorbed statement.
        merge_axis: Named axis being merged.

    Returns:
        Widened output TensorRef.
    """
    output_dim = _axis_dim(stmt_a.op.outputs[0].axes, merge_axis)
    out_a = stmt_a.output
    result = out_a
    if output_dim >= 0 and out_a.name != stmt_b.output.name:
        merged_extent = out_a.shape[output_dim] + stmt_b.output.shape[output_dim]
        new_shape = tuple(merged_extent if d == output_dim else s for d, s in enumerate(out_a.shape))
        result = TensorRef(out_a.name, new_shape, tuple((0, s) for s in new_shape))
    elif output_dim >= 0:
        out_b = stmt_b.output
        widened = (
            min(out_a.slices[output_dim][0], out_b.slices[output_dim][0]),
            max(out_a.slices[output_dim][1], out_b.slices[output_dim][1]),
        )
        result = TensorRef(out_a.name, out_a.shape, _widen_slice(out_a.slices, output_dim, widened))
    return result


def _widen_on_axis(
    stmt_a: GymStatement, stmt_b: GymStatement, merge_axis: str
) -> tuple[tuple[tuple[str, Any], ...], TensorRef]:
    """Widen the kept statement's kwargs and output on the merge axis.

    For each TensorRef kwarg (excluding acc) where the merge axis appears,
    widens the slices on that dimension. Only widens when both refs
    reference the same variable name.

    Args:
        stmt_a: Kept statement.
        stmt_b: Absorbed statement.
        merge_axis: Named axis being merged.

    Returns:
        Tuple of (new_kwargs, new_output).
    """
    op_cls = stmt_a.op
    new_kwargs = list(stmt_a.kwargs)
    tensor_pos = 0
    for i, (k, v) in enumerate(stmt_a.kwargs):
        if not isinstance(v, TensorRef) or k == "acc":
            continue
        merge_dim = _axis_dim(op_cls.inputs[tensor_pos].axes, merge_axis)
        ref_b = stmt_b.kwargs[i][1]
        if merge_dim >= 0 and v.name == ref_b.name:
            new_kwargs[i] = (k, _widen_kwarg(v, ref_b, merge_dim))
        tensor_pos += 1
    return tuple(new_kwargs), _widen_output(stmt_a, stmt_b, merge_axis)


def _build_subscript_map(
    stmt_a: GymStatement, stmt_b: GymStatement, option: MergeOpportunity
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Build a subscript map for consumer remapping after merge.

    Maps each original output variable to its relative slice within the
    wider merged output. Only needed for SSA merges (different output
    names). The offset is derived from the differing input kwarg's
    slice positions.

    Args:
        stmt_a: Kept statement.
        stmt_b: Absorbed statement.
        option: The merge opportunity.

    Returns:
        Dict mapping variable names to relative output slices,
        or empty dict if not an SSA merge.
    """
    result: dict[str, tuple[tuple[int, int], ...]] = {}
    output_dim = _axis_dim(stmt_a.op.outputs[0].axes, option.merge_axis)
    if stmt_a.output.name != stmt_b.output.name and output_dim >= 0:
        diff_idx = option.differing_operand_idx
        ref_a = stmt_a.kwargs[diff_idx][1]
        ref_b = stmt_b.kwargs[diff_idx][1]
        merged_start = min(ref_a.slices[option.differing_dim][0], ref_b.slices[option.differing_dim][0])

        for out_ref, in_ref in ((stmt_a.output, ref_a), (stmt_b.output, ref_b)):
            offset = in_ref.slices[option.differing_dim][0] - merged_start
            result[out_ref.name] = tuple(
                (offset, offset + out_ref.shape[d]) if d == output_dim else (0, out_ref.shape[d])
                for d in range(len(out_ref.shape))
            )
    return result


def _apply_merge(program: GymProgram, option: MergeOpportunity) -> GymProgram:
    """Apply a single merge opportunity to the program.

    Widens the kept statement, removes the absorbed statement, and
    rewrites downstream consumers via subscript remapping and variable
    renaming.

    Args:
        program: GymProgram to transform.
        option: A MergeOpportunity from ``_find_merge_opportunities``.

    Returns:
        New GymProgram with the merge applied.
    """
    stmts = program.stmts
    stmt_a = stmts[option.stmt_a]
    stmt_b = stmts[option.stmt_b]

    subscript_map = _build_subscript_map(stmt_a, stmt_b, option)
    renames: dict[str, str] = {}
    if stmt_a.output.name != stmt_b.output.name:
        renames[stmt_b.output.name] = stmt_a.output.name

    new_stmts: list[GymStatement] = []
    for i, stmt in enumerate(stmts):
        if i == option.stmt_b:
            continue
        kwargs = stmt.kwargs
        output = stmt.output
        if i == option.stmt_a:
            kwargs, output = _widen_on_axis(stmt_a, stmt_b, option.merge_axis)
        elif subscript_map:
            kwargs, output = _remap_stmt_refs(kwargs, output, subscript_map)
        if renames:
            kwargs, output = _rename_stmt_refs(kwargs, output, renames)
        new_stmts.append(GymStatement(stmt.op, kwargs, output))

    return _replace_stmts(program, tuple(new_stmts))


class OperandMergeTransform(Transform):
    """Transform that merges adjacent operations on contiguous tensor slices.

    ``analyze_ir()`` returns a list of ``MergeOpportunity`` objects.
    ``transform_ir()`` applies one opportunity at a time.

    Example::

        merge = OperandMergeTransform()
        while True:
            opportunities = merge.analyze_ir(program)
            if not opportunities:
                break
            program = merge.transform_ir(program, opportunities[0])
    """

    name = "operand_merge"

    def analyze_ir(self, program: GymProgram) -> list[MergeOpportunity]:
        """Analyze a program to find merge opportunities.

        Args:
            program: GymProgram tuple.

        Returns:
            List of MergeOpportunity objects.
        """
        index = StmtIndex.build(program.stmts)
        return _find_merge_opportunities(index)

    def transform_ir(self, program: GymProgram, option: MergeOpportunity) -> GymProgram:
        """Apply a single merge opportunity.

        Args:
            program: GymProgram to transform.
            option: A single MergeOpportunity from ``analyze_ir()``.

        Returns:
            New GymProgram with the merged operation applied.
        """
        return _apply_merge(program, option)
