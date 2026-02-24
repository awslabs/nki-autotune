"""Operand merge analysis and transform for tiled compute graphs.

Identifies adjacent statement groups that operate on contiguous slices of
the same tensor and can be merged into a single wider operation. This
reduces the total number of load/store/compute statements, improving
hardware utilization.

Operates on the GymProgram IR. Hardware tile limits are delegated
to ``GymOp.can_merge_operand_dim()`` rather than a hardcoded table.

Example::

    merge = OperandMergeTransform()
    while True:
        opportunities = merge.analyze_ir(program)
        if not opportunities:
            break
        program = merge.transform_ir(program, opportunities[0])
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from nkigym.ir import GymProgram, GymStatement, TensorRef
from nkigym.ops import GymOp
from nkigym.transforms.base import Transform

TILING_OPS = frozenset({"np_slice", "np_store", "np_empty"})


def _is_compute(stmt: GymStatement) -> bool:
    """Check if a statement is a compute operation (not a tiling op).

    Args:
        stmt: A GymStatement.

    Returns:
        True if the statement is not a tiling infrastructure op.
    """
    return stmt.op not in TILING_OPS


def _get_kwarg_ref(stmt: GymStatement, key: str) -> TensorRef:
    """Get a TensorRef kwarg by key from a statement.

    Args:
        stmt: A GymStatement.
        key: The keyword argument name.

    Returns:
        The TensorRef value associated with the key.

    Raises:
        ValueError: If no kwarg with the given key is found.
    """
    for k, v in stmt.kwargs:
        if k == key:
            return v
    raise ValueError(f"{stmt.op} statement missing '{key}' kwarg")


def _is_identity(ref: TensorRef) -> bool:
    """Check if a TensorRef has identity slices covering the full shape.

    Args:
        ref: Tensor reference to check.

    Returns:
        True if slices start at 0 and span each dimension's full size.
    """
    return all(s == 0 for s, _ in ref.slices) and tuple(e - s for s, e in ref.slices) == ref.shape


def _check_adjacent_slices(
    slices_a: tuple[tuple[int, int], ...], slices_b: tuple[tuple[int, int], ...]
) -> tuple[int, tuple[int, int]] | None:
    """Check if two slice tuples differ on exactly one dimension with adjacent ranges.

    Args:
        slices_a: Slice tuple from the first operand.
        slices_b: Slice tuple from the second operand.

    Returns:
        Tuple of (differing_dim_index, merged_slice) if adjacent,
        or None if not mergeable.
    """
    if len(slices_a) != len(slices_b):
        return None

    differing_dims = [d for d in range(len(slices_a)) if slices_a[d] != slices_b[d]]
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


def _widen_slice(slices: tuple[tuple[int, int], ...], dim: int, merged: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Widen a single dimension in a slice tuple.

    Args:
        slices: Original slice tuple.
        dim: Dimension to widen.
        merged: New (start, stop) for that dimension.

    Returns:
        New slice tuple with the widened dimension.
    """
    return (*slices[:dim], merged, *slices[dim + 1 :])


def _relative_slices(
    abs_slices: tuple[tuple[int, int], ...], merge_dim: int, merged_start: int
) -> tuple[tuple[int, int], ...]:
    """Convert absolute source-tensor slices to relative loaded-tensor slices.

    Args:
        abs_slices: Absolute slices from the load statement.
        merge_dim: The dimension being merged (widened).
        merged_start: The start of the merged range on merge_dim.

    Returns:
        Tuple of (start, stop) pairs relative to the loaded tensor.
    """
    return tuple(
        (s - merged_start, e - merged_start) if d == merge_dim else (0, e - s) for d, (s, e) in enumerate(abs_slices)
    )


def _offset_slices(
    rel_slices: tuple[tuple[int, int], ...], existing_slices: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, int], ...]:
    """Offset existing subscript slices by relative position.

    Args:
        rel_slices: Relative slices from _relative_slices.
        existing_slices: Existing subscript slices to offset.

    Returns:
        New slices with offset applied.
    """
    return tuple((rel_s + sub_s, rel_s + sub_e) for (rel_s, _), (sub_s, sub_e) in zip(rel_slices, existing_slices))


def _find_differing_dim(slices_a: tuple[tuple[int, int], ...], slices_b: tuple[tuple[int, int], ...]) -> int | None:
    """Find the first dimension where two slice tuples differ.

    Args:
        slices_a: First slice tuple.
        slices_b: Second slice tuple.

    Returns:
        Dimension index, or None if all dimensions match.
    """
    for d in range(len(slices_a)):
        if slices_a[d] != slices_b[d]:
            return d
    return None


@dataclass
class StmtIndex:
    """Pre-computed index over program statements for efficient lookups.

    Built once per analysis/transform pass, replacing repeated linear scans
    with O(1) dict lookups.

    Attributes:
        stmts: The original program statements.
        loads_by_output: Maps output variable name to load statement index.
        stores_by_src: Maps source variable name to store statement index.
        var_usage: Maps variable name to sorted list of statement indices
            that reference it.
    """

    stmts: tuple[GymStatement, ...]
    loads_by_output: dict[str, int]
    stores_by_src: dict[str, int]
    var_usage: dict[str, list[int]]

    @classmethod
    def build(cls, stmts: tuple[GymStatement, ...]) -> "StmtIndex":
        """Build all indexes in a single pass over statements.

        Args:
            stmts: Program statements tuple.

        Returns:
            Populated StmtIndex.
        """
        loads_by_output: dict[str, int] = {}
        stores_by_src: dict[str, int] = {}
        usage: dict[str, set[int]] = {}

        for i, stmt in enumerate(stmts):
            if stmt.op == "np_slice":
                loads_by_output[stmt.output.name] = i
            elif stmt.op == "np_store":
                src_ref = _get_kwarg_ref(stmt, "src")
                stores_by_src[src_ref.name] = i

            for _, value in stmt.kwargs:
                if isinstance(value, TensorRef):
                    usage.setdefault(value.name, set()).add(i)
            usage.setdefault(stmt.output.name, set()).add(i)

        return cls(
            stmts=stmts,
            loads_by_output=loads_by_output,
            stores_by_src=stores_by_src,
            var_usage={var: sorted(indices) for var, indices in usage.items()},
        )

    def load_src_slices(self, var_name: str) -> tuple[tuple[int, int], ...] | None:
        """Get the source slices for a load that produces var_name.

        Args:
            var_name: Variable name that might be a loaded tensor.

        Returns:
            The source slices from the load, or None if not a load variable.
        """
        idx = self.loads_by_output.get(var_name)
        if idx is None:
            return None
        return _get_kwarg_ref(self.stmts[idx], "src").slices

    def resolve_operand_slices(self, ref: TensorRef) -> tuple[tuple[int, int], ...] | None:
        """Resolve operand slices to source-tensor coordinates.

        Composes the ref's slices with the load's source slices so
        that the result is always in source-tensor coordinates. This
        is necessary because after a load merge, an identity-looking
        ref may actually be a subscript into a wider load.

        Args:
            ref: TensorRef for the operand.

        Returns:
            Source-tensor coordinates, or None if not a loaded variable.
        """
        load_src = self.load_src_slices(ref.name)
        if load_src is None:
            return ref.slices
        return tuple((ls + rs, ls + re) for (ls, _), (rs, re) in zip(load_src, ref.slices))

    def operand_signature(self, ref: TensorRef) -> tuple:
        """Create a hashable signature for an operand, resolving through loads.

        Args:
            ref: TensorRef for the operand.

        Returns:
            Hashable tuple for comparison.
        """
        if not _is_identity(ref):
            return (ref.name, ref.slices)
        idx = self.loads_by_output.get(ref.name)
        if idx is not None:
            src_ref = _get_kwarg_ref(self.stmts[idx], "src")
            return ("load", src_ref.name, src_ref.slices)
        return (ref.name,)


@dataclass
class MergeOpportunity:
    """A single merge opportunity found by ``analyze_ir()``.

    Attributes:
        op_type: The operation type string (e.g., ``"nc_matmul"``,
            ``"load"``).
        stmt_a: The statement index of the first operation.
        stmt_b: The statement index of the second operation
            (to be absorbed).
        differing_operand_idx: Index of the operand that differs between
            the two ops. For loads this is the slice dimension index.
        differing_dim: The dimension index within the operand's slices
            that differs between the two operations.
        merged_slice: A ``(start, stop)`` tuple for the merged
            free-dimension range of the differing operand.
        description: Human-readable description for logging.
    """

    op_type: str
    stmt_a: int
    stmt_b: int
    differing_operand_idx: int
    differing_dim: int
    merged_slice: tuple[int, int]
    description: str


def _find_load_merge_opportunities(index: StmtIndex) -> list[MergeOpportunity]:
    """Find load merge opportunities.

    Groups loads by (source_tensor, partition_slice) and checks adjacent
    pairs within each group for mergeable free-dimension ranges.

    Args:
        index: Pre-built statement index.

    Returns:
        List of MergeOpportunity for loads.
    """
    stmts = index.stmts
    loads = [(i, stmt) for i, stmt in enumerate(stmts) if stmt.op == "np_slice"]
    if len(loads) < 2:
        return []

    load_groups: dict[tuple, list[tuple[int, GymStatement]]] = {}
    for idx, stmt in loads:
        src_ref = _get_kwarg_ref(stmt, "src")
        if len(src_ref.slices) < 2:
            continue
        partition_key = (src_ref.name, src_ref.slices[0])
        load_groups.setdefault(partition_key, []).append((idx, stmt))

    used: set[int] = set()
    opportunities: list[MergeOpportunity] = []

    for members in load_groups.values():
        if len(members) < 2:
            continue

        ordered = sorted(members, key=lambda m: _get_kwarg_ref(m[1], "src").slices[-1][0])

        for i in range(len(ordered)):
            if ordered[i][0] in used:
                continue
            for j in range(i + 1, len(ordered)):
                if ordered[j][0] in used:
                    continue

                idx_a, stmt_a = ordered[i]
                idx_b, stmt_b = ordered[j]
                src_slices_a = _get_kwarg_ref(stmt_a, "src").slices
                src_slices_b = _get_kwarg_ref(stmt_b, "src").slices

                adj = _check_adjacent_slices(src_slices_a, src_slices_b)
                if adj is None:
                    continue

                dim, merged = adj
                first_idx = min(idx_a, idx_b)
                second_idx = max(idx_a, idx_b)
                first_stmt = stmt_a if idx_a <= idx_b else stmt_b
                second_stmt = stmt_b if idx_a <= idx_b else stmt_a

                opportunities.append(
                    MergeOpportunity(
                        op_type="load",
                        stmt_a=first_idx,
                        stmt_b=second_idx,
                        differing_operand_idx=dim,
                        differing_dim=dim,
                        merged_slice=merged,
                        description=(
                            f"Merge load {first_stmt.output.name} and "
                            f"{second_stmt.output.name} "
                            f"[dim {dim}: {src_slices_a[dim]} + "
                            f"{src_slices_b[dim]} -> {merged}]"
                        ),
                    )
                )
                used.add(first_idx)
                used.add(second_idx)
                break

    return opportunities


def _find_compute_merge_opportunities(index: StmtIndex) -> list[MergeOpportunity]:
    """Find compute op merge opportunities.

    Groups compute statements by op type and checks pairs within each
    group for mergeable operand dimensions.

    Args:
        index: Pre-built statement index.

    Returns:
        List of MergeOpportunity for compute ops.
    """
    stmts = index.stmts
    compute_stmts = [(i, stmt) for i, stmt in enumerate(stmts) if _is_compute(stmt)]
    if len(compute_stmts) < 2:
        return []

    op_groups: dict[str, list[tuple[int, GymStatement]]] = {}
    for entry in compute_stmts:
        op_groups.setdefault(entry[1].op, []).append(entry)

    used: set[int] = set()
    opportunities: list[MergeOpportunity] = []

    for members in op_groups.values():
        if len(members) < 2:
            continue

        for i in range(len(members)):
            if members[i][0] in used:
                continue
            for j in range(i + 1, len(members)):
                if members[j][0] in used:
                    continue

                opp = _check_compute_pair(members[i], members[j], index)
                if opp is not None:
                    opportunities.append(opp)
                    used.add(opp.stmt_a)
                    used.add(opp.stmt_b)
                    break

    return opportunities


def _check_compute_pair(
    entry_a: tuple[int, GymStatement], entry_b: tuple[int, GymStatement], index: StmtIndex
) -> MergeOpportunity | None:
    """Check if two compute statements can be merged.

    Verifies same op type, matching non-tensor config kwargs, exactly one
    differing tensor operand with adjacent slices within hardware limits,
    compatible stores, and safe dependencies for both kept and absorbed vars.

    Args:
        entry_a: (index, stmt) for first compute.
        entry_b: (index, stmt) for second compute.
        index: Pre-built statement index.

    Returns:
        MergeOpportunity or None.
    """
    idx_a, stmt_a = entry_a
    idx_b, stmt_b = entry_b

    if stmt_a.op != stmt_b.op:
        return None

    tensor_indices_a = [i for i, (_, v) in enumerate(stmt_a.kwargs) if isinstance(v, TensorRef)]
    tensor_indices_b = [i for i, (_, v) in enumerate(stmt_b.kwargs) if isinstance(v, TensorRef)]
    config_a = tuple((k, v) for k, v in stmt_a.kwargs if not isinstance(v, TensorRef) and k != "acc")
    config_b = tuple((k, v) for k, v in stmt_b.kwargs if not isinstance(v, TensorRef) and k != "acc")
    if config_a != config_b:
        return None

    if len(tensor_indices_a) != len(tensor_indices_b):
        return None

    differing_args: list[int] = []
    for ti_a, ti_b in zip(tensor_indices_a, tensor_indices_b):
        _, ref_a = stmt_a.kwargs[ti_a]
        _, ref_b = stmt_b.kwargs[ti_b]
        if index.operand_signature(ref_a) != index.operand_signature(ref_b):
            differing_args.append(ti_a)

    if len(differing_args) != 1:
        return None

    diff_idx = differing_args[0]
    operand_pos = tensor_indices_a.index(diff_idx)

    _, ref_a = stmt_a.kwargs[diff_idx]
    _, ref_b = stmt_b.kwargs[diff_idx]

    if ref_a.name != ref_b.name:
        return None

    slices_a = index.resolve_operand_slices(ref_a)
    slices_b = index.resolve_operand_slices(ref_b)

    if slices_a is None or slices_b is None:
        return None

    adj = _check_adjacent_slices(slices_a, slices_b)
    if adj is None:
        return None

    dim, merged = adj
    merged_size = merged[1] - merged[0]

    op_instance = GymOp.get(stmt_a.op)()
    if not op_instance.can_merge_operand_dim(operand_pos, dim, merged_size):
        return None

    trial_args: list[object] = []
    for i, (k, v) in enumerate(stmt_a.kwargs):
        if k == "acc":
            continue
        if isinstance(v, TensorRef):
            if i == diff_idx:
                shape = tuple(merged_size if d == dim else (e - s) for d, (s, e) in enumerate(v.slices))
            else:
                shape = tuple(e - s for s, e in v.slices)
            trial_args.append(np.zeros(shape, dtype=np.float32))
        elif isinstance(v, str):
            try:
                trial_args.append(float(v))
            except ValueError:
                continue
        else:
            trial_args.append(v)
    try:
        op_instance.simulate(*trial_args)
    except (ValueError, TypeError):
        return None

    first_idx = min(idx_a, idx_b)
    second_idx = max(idx_a, idx_b)

    return MergeOpportunity(
        op_type=stmt_a.op,
        stmt_a=first_idx,
        stmt_b=second_idx,
        differing_operand_idx=diff_idx,
        differing_dim=dim,
        merged_slice=merged,
        description=(
            f"Merge {stmt_a.op} at stmt {first_idx} and "
            f"{second_idx} on arg {diff_idx} "
            f"[dim {dim}: {slices_a[dim]} + {slices_b[dim]} -> {merged}]"
        ),
    )


def _replace_stmts(program: GymProgram, new_stmts: tuple[GymStatement, ...]) -> GymProgram:
    """Create a new GymProgram with replaced statements.

    Args:
        program: Original program.
        new_stmts: New statement tuple.

    Returns:
        New GymProgram with updated statements.
    """
    return GymProgram(
        program.name, program.params, program.input_shapes, new_stmts, program.return_var, program.output_dtype
    )


def _remap_ref(ref: TensorRef, subscript_map: dict[str, tuple[tuple[int, int], ...]]) -> TensorRef:
    """Remap a TensorRef using a subscript offset map.

    Offsets the ref's slices by the relative position in the map.
    The ref's shape is preserved, so the consumer reads the same
    amount of data from the correct offset within the wider variable.

    Args:
        ref: TensorRef to remap.
        subscript_map: Mapping from variable name to relative slices.

    Returns:
        Remapped TensorRef, or original if not in map.
    """
    if ref.name not in subscript_map:
        return ref
    rel = subscript_map[ref.name]
    new_slices = _offset_slices(rel, ref.slices)
    return TensorRef(ref.name, ref.shape, new_slices)


def _remap_stmt_refs(
    kwargs: tuple[tuple[str, Any], ...], output: TensorRef, subscript_map: dict[str, tuple[tuple[int, int], ...]]
) -> tuple[tuple[tuple[str, Any], ...], TensorRef]:
    """Rewrite TensorRef slices in kwargs and output using a subscript offset map.

    Args:
        kwargs: Original kwargs tuple.
        output: Original output TensorRef.
        subscript_map: Mapping from variable name to relative slices.

    Returns:
        Tuple of (new_kwargs, new_output).
    """
    new_kwargs = tuple(
        (key, _remap_ref(value, subscript_map)) if isinstance(value, TensorRef) else (key, value)
        for key, value in kwargs
    )
    return new_kwargs, _remap_ref(output, subscript_map)


def _rename_ref(ref: TensorRef, renames: dict[str, str]) -> TensorRef:
    """Apply variable renames to a TensorRef.

    Args:
        ref: TensorRef to rename.
        renames: Mapping from old variable names to new names.

    Returns:
        Renamed TensorRef, or original if not in map.
    """
    if ref.name in renames:
        return TensorRef(renames[ref.name], ref.shape, ref.slices)
    return ref


def _rename_stmt_refs(
    kwargs: tuple[tuple[str, Any], ...], output: TensorRef, renames: dict[str, str]
) -> tuple[tuple[tuple[str, Any], ...], TensorRef]:
    """Apply variable renames to kwargs and output.

    Args:
        kwargs: Original kwargs tuple.
        output: Original output TensorRef.
        renames: Mapping from old variable names to new names.

    Returns:
        Tuple of (new_kwargs, new_output).
    """
    new_kwargs = tuple(
        (key, _rename_ref(value, renames)) if isinstance(value, TensorRef) else (key, value) for key, value in kwargs
    )
    return new_kwargs, _rename_ref(output, renames)


def _widen_load_stmt(
    kwargs: tuple[tuple[str, Any], ...], output: TensorRef, dim: int, merged: tuple[int, int]
) -> tuple[tuple[tuple[str, Any], ...], TensorRef]:
    """Widen the source slices and destination shape of a load statement.

    Args:
        kwargs: Original kwargs tuple.
        output: Original output TensorRef.
        dim: Dimension to widen.
        merged: New (start, stop) for that dimension.

    Returns:
        Tuple of (new_kwargs, new_output).
    """
    new_kwargs_list: list[tuple[str, Any]] = []
    for key, value in kwargs:
        if key == "src" and isinstance(value, TensorRef):
            new_src_slices = _widen_slice(value.slices, dim, merged)
            new_kwargs_list.append((key, TensorRef(value.name, value.shape, new_src_slices)))
        else:
            new_kwargs_list.append((key, value))

    new_src = next(v for k, v in new_kwargs_list if k == "src")
    new_dst_shape = tuple(merged[1] - merged[0] if d == dim else (e - s) for d, (s, e) in enumerate(new_src.slices))
    new_dst_slices = tuple((0, s) for s in new_dst_shape)
    new_output = TensorRef(output.name, new_dst_shape, new_dst_slices)

    return tuple(new_kwargs_list), new_output


def _compute_output_dim(stmt: GymStatement, diff_idx: int, diff_dim: int) -> int | None:
    """Find which output dimension corresponds to a differing input dimension.

    Uses the op's axis mapping to determine which output dimension widens
    when an input operand is widened on ``diff_dim``.

    Args:
        stmt: The compute statement.
        diff_idx: Index of the differing kwarg.
        diff_dim: Dimension index within that operand's slices.

    Returns:
        Output dimension index, or None if no mapping exists.
    """
    op_cls = GymOp.get(stmt.op)
    operand_pos = sum(1 for i, (_, v) in enumerate(stmt.kwargs) if isinstance(v, TensorRef) and i < diff_idx)
    input_axes = op_cls.inputs[operand_pos].axes if operand_pos < len(op_cls.inputs) else ()
    output_axes = op_cls.outputs[0].axes
    axis_name = input_axes[diff_dim] if diff_dim < len(input_axes) else None
    if not isinstance(axis_name, str):
        return None
    for od, ax in enumerate(output_axes):
        if ax == axis_name:
            return od
    return None


def _widen_compute_stmt(
    kwargs: tuple[tuple[str, Any], ...], output: TensorRef, option: MergeOpportunity, stmts: tuple[GymStatement, ...]
) -> tuple[tuple[tuple[str, Any], ...], TensorRef]:
    """Widen the kept compute statement's differing operand and output.

    Args:
        kwargs: Original kwargs of the kept stmt.
        output: Original output TensorRef of the kept stmt.
        option: The merge opportunity.
        stmts: All program statements.

    Returns:
        Tuple of (new_kwargs, new_output).
    """
    diff_idx = option.differing_operand_idx
    dim = option.differing_dim
    key, ref = kwargs[diff_idx]

    new_slices = _widen_slice(ref.slices, dim, option.merged_slice)
    new_shape = tuple(e - s for s, e in new_slices)
    new_ref = TensorRef(ref.name, new_shape, new_slices)
    new_kwargs = list(kwargs)
    new_kwargs[diff_idx] = (key, new_ref)

    output_dim = _compute_output_dim(stmts[option.stmt_a], diff_idx, dim)
    if output_dim is not None:
        merged_extent = option.merged_slice[1] - option.merged_slice[0]
        out_slices = list(output.slices)
        out_slices[output_dim] = (0, merged_extent)
        new_out_shape = list(output.shape)
        new_out_shape[output_dim] = merged_extent
        output = TensorRef(output.name, tuple(new_out_shape), tuple(out_slices))

    return tuple(new_kwargs), output


def _widen_store(
    kwargs: tuple[tuple[str, Any], ...], output: TensorRef, store_a_stmt: GymStatement, store_b_stmt: GymStatement
) -> tuple[tuple[tuple[str, Any], ...], TensorRef]:
    """Widen the kept store's target and value slices.

    Finds the differing dimension between the two original store
    destinations, then widens both dst and src in the current kwargs.

    Args:
        kwargs: Current store kwargs.
        output: Current store output.
        store_a_stmt: Original store A statement.
        store_b_stmt: Original store B statement.

    Returns:
        Tuple of (new_kwargs, new_output).
    """
    target_a = _get_kwarg_ref(store_a_stmt, "dst").slices
    target_b = _get_kwarg_ref(store_b_stmt, "dst").slices

    dim = _find_differing_dim(target_a, target_b)
    if dim is None:
        return kwargs, output

    merged_target = (min(target_a[dim][0], target_b[dim][0]), max(target_a[dim][1], target_b[dim][1]))

    new_kwargs: list[tuple[str, Any]] = []
    new_dst = output
    for key, value in kwargs:
        if key == "dst" and isinstance(value, TensorRef):
            new_dst_slices = _widen_slice(value.slices, dim, merged_target)
            new_dst = TensorRef(value.name, value.shape, new_dst_slices)
            new_kwargs.append((key, new_dst))
        elif key == "src" and isinstance(value, TensorRef) and value.slices:
            vs = value.slices[dim][0]
            merged_value = (vs, vs + (merged_target[1] - merged_target[0]))
            new_src_slices = _widen_slice(value.slices, dim, merged_value)
            new_src_shape = tuple(e - s for s, e in new_src_slices)
            new_kwargs.append((key, TensorRef(value.name, new_src_shape, new_src_slices)))
        else:
            new_kwargs.append((key, value))

    return tuple(new_kwargs), new_dst


def _apply_load_merge(program: GymProgram, option: MergeOpportunity) -> GymProgram:
    """Apply a load merge opportunity.

    Widens the kept load, removes the absorbed load, and rewrites
    downstream consumers to use subscript offsets into the wider load.

    Args:
        program: GymProgram to transform.
        option: The load merge opportunity.

    Returns:
        New GymProgram.
    """
    stmts = program.stmts
    stmt_a = stmts[option.stmt_a]
    stmt_b = stmts[option.stmt_b]

    merge_dim = option.differing_operand_idx
    merged_start = option.merged_slice[0]

    var_a = stmt_a.output.name
    var_b = stmt_b.output.name

    src_a = _get_kwarg_ref(stmt_a, "src")
    src_b = _get_kwarg_ref(stmt_b, "src")

    subscript_map: dict[str, tuple[tuple[int, int], ...]] = {
        var_a: _relative_slices(src_a.slices, merge_dim, merged_start),
        var_b: _relative_slices(src_b.slices, merge_dim, merged_start),
    }

    renames = {var_b: var_a}

    new_stmts: list[GymStatement] = []
    for i, stmt in enumerate(stmts):
        if i == option.stmt_b:
            continue

        kwargs = stmt.kwargs
        output = stmt.output

        if i == option.stmt_a:
            kwargs, output = _widen_load_stmt(kwargs, output, merge_dim, option.merged_slice)
        else:
            kwargs, output = _remap_stmt_refs(kwargs, output, subscript_map)

        kwargs, output = _rename_stmt_refs(kwargs, output, renames)
        new_stmts.append(GymStatement(stmt.op, kwargs, output))

    return _replace_stmts(program, tuple(new_stmts))


def _build_compute_subscript_map(
    stmt_a: GymStatement, stmt_b: GymStatement, option: MergeOpportunity, index: StmtIndex
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Build a subscript map for consumer remapping after compute merge.

    Maps each original output variable to its relative slice within the
    wider merged output, so downstream consumers are remapped to read the
    correct portion.

    Args:
        stmt_a: Kept compute statement.
        stmt_b: Absorbed compute statement.
        option: The merge opportunity.
        index: Pre-built statement index.

    Returns:
        Dict mapping variable names to relative output slices,
        or empty dict if no output dimension is widened.
    """
    output_dim = _compute_output_dim(stmt_a, option.differing_operand_idx, option.differing_dim)
    if output_dim is None:
        return {}

    diff_idx = option.differing_operand_idx
    _, ref_a = stmt_a.kwargs[diff_idx]
    _, ref_b = stmt_b.kwargs[diff_idx]
    slices_a = index.resolve_operand_slices(ref_a)
    slices_b = index.resolve_operand_slices(ref_b)
    if slices_a is None or slices_b is None:
        return {}

    merged_start = option.merged_slice[0]
    diff_dim = option.differing_dim

    def _output_rel(original_shape: tuple[int, ...], operand_slice: tuple[int, int]) -> tuple[tuple[int, int], ...]:
        """Compute relative output slices for one variable."""
        result: list[tuple[int, int]] = []
        for d in range(len(original_shape)):
            if d == output_dim:
                offset = operand_slice[0] - merged_start
                result.append((offset, offset + original_shape[d]))
            else:
                result.append((0, original_shape[d]))
        return tuple(result)

    return {
        stmt_a.output.name: _output_rel(stmt_a.output.shape, slices_a[diff_dim]),
        stmt_b.output.name: _output_rel(stmt_b.output.shape, slices_b[diff_dim]),
    }


def _apply_compute_merge(program: GymProgram, option: MergeOpportunity) -> GymProgram:
    """Apply a compute op merge opportunity.

    Widens the kept compute statement, removes the absorbed compute,
    remaps downstream consumers (including store src refs) via subscript
    offsets, and renames absorbed variable references. Does not merge
    stores — store merging is a separate atomic step.

    Args:
        program: GymProgram to transform.
        option: A compute MergeOpportunity.

    Returns:
        New GymProgram.
    """
    stmts = program.stmts
    index = StmtIndex.build(stmts)

    stmt_a = stmts[option.stmt_a]
    stmt_b = stmts[option.stmt_b]

    var_a = stmt_a.output.name
    var_b = stmt_b.output.name
    renames = {var_b: var_a}

    to_remove: set[int] = {option.stmt_b}

    subscript_map = _build_compute_subscript_map(stmt_a, stmt_b, option, index)

    new_stmts: list[GymStatement] = []
    for i, stmt in enumerate(stmts):
        if i in to_remove:
            continue

        kwargs = stmt.kwargs
        output = stmt.output

        if i == option.stmt_a:
            kwargs, output = _widen_compute_stmt(kwargs, output, option, stmts)
        elif subscript_map:
            kwargs, output = _remap_stmt_refs(kwargs, output, subscript_map)

        kwargs, output = _rename_stmt_refs(kwargs, output, renames)
        new_stmts.append(GymStatement(stmt.op, kwargs, output))

    return _replace_stmts(program, tuple(new_stmts))


def _find_store_merge_opportunities(index: StmtIndex) -> list[MergeOpportunity]:
    """Find store merge opportunities.

    Groups stores by destination tensor and checks adjacent pairs for
    mergeable dimension ranges. Both the dst and src must reference the
    same variable with adjacent slices on the same dimension, and the
    merged partition dim (dim 0) must be <= 128.

    Args:
        index: Pre-built statement index.

    Returns:
        List of MergeOpportunity for stores.
    """
    stmts = index.stmts
    stores = [(i, stmt) for i, stmt in enumerate(stmts) if stmt.op == "np_store"]
    if len(stores) < 2:
        return []

    store_groups: dict[str, list[tuple[int, GymStatement]]] = {}
    for idx, stmt in stores:
        dst_ref = _get_kwarg_ref(stmt, "dst")
        store_groups.setdefault(dst_ref.name, []).append((idx, stmt))

    used: set[int] = set()
    opportunities: list[MergeOpportunity] = []

    for members in store_groups.values():
        if len(members) < 2:
            continue

        for i in range(len(members)):
            if members[i][0] in used:
                continue
            for j in range(i + 1, len(members)):
                if members[j][0] in used:
                    continue

                idx_a, stmt_a = members[i]
                idx_b, stmt_b = members[j]

                src_a = _get_kwarg_ref(stmt_a, "src")
                src_b = _get_kwarg_ref(stmt_b, "src")
                if src_a.name != src_b.name:
                    continue

                dst_a = _get_kwarg_ref(stmt_a, "dst")
                dst_b = _get_kwarg_ref(stmt_b, "dst")

                adj = _check_adjacent_slices(dst_a.slices, dst_b.slices)
                if adj is None:
                    continue

                src_adj = _check_adjacent_slices(src_a.slices, src_b.slices)
                if src_adj is None:
                    continue

                dim, merged = adj
                merged_size = merged[1] - merged[0]
                if dim == 0 and merged_size > 128:
                    continue

                first_idx = min(idx_a, idx_b)
                second_idx = max(idx_a, idx_b)

                opportunities.append(
                    MergeOpportunity(
                        op_type="store",
                        stmt_a=first_idx,
                        stmt_b=second_idx,
                        differing_operand_idx=dim,
                        differing_dim=dim,
                        merged_slice=merged,
                        description=(
                            f"Merge store at stmt {first_idx} and "
                            f"{second_idx} into {dst_a.name} "
                            f"[dim {dim}: {dst_a.slices[dim]} + "
                            f"{dst_b.slices[dim]} -> {merged}]"
                        ),
                    )
                )
                used.add(first_idx)
                used.add(second_idx)
                break

    return opportunities


def _apply_store_merge(program: GymProgram, option: MergeOpportunity) -> GymProgram:
    """Apply a store merge opportunity.

    Widens the kept store statement and removes the absorbed store.

    Args:
        program: GymProgram to transform.
        option: A store MergeOpportunity.

    Returns:
        New GymProgram.
    """
    stmts = program.stmts
    store_a = stmts[option.stmt_a]
    store_b = stmts[option.stmt_b]

    new_stmts: list[GymStatement] = []
    for i, stmt in enumerate(stmts):
        if i == option.stmt_b:
            continue

        if i == option.stmt_a:
            kwargs, output = _widen_store(stmt.kwargs, stmt.output, store_a, store_b)
            new_stmts.append(GymStatement(stmt.op, kwargs, output))
        else:
            new_stmts.append(stmt)

    return _replace_stmts(program, tuple(new_stmts))


class OperandMergeTransform(Transform):
    """Transform that merges adjacent operations on contiguous tensor slices.

    Identifies pairs of statements that perform the same operation but on
    adjacent slices of a tensor, differing on exactly one operand dimension.
    These can be combined into a single wider operation without exceeding
    hardware tile limits (checked via ``GymOp.can_merge_operand_dim()``).

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
        opportunities: list[MergeOpportunity] = []
        opportunities.extend(_find_load_merge_opportunities(index))
        opportunities.extend(_find_compute_merge_opportunities(index))
        opportunities.extend(_find_store_merge_opportunities(index))
        return opportunities

    def transform_ir(self, program: GymProgram, option: MergeOpportunity) -> GymProgram:
        """Apply a single merge opportunity.

        Args:
            program: GymProgram to transform.
            option: A single MergeOpportunity from ``analyze_ir()``.

        Returns:
            New GymProgram with the merged operation applied.
        """
        if option.op_type == "load":
            return _apply_load_merge(program, option)
        if option.op_type == "store":
            return _apply_store_merge(program, option)
        return _apply_compute_merge(program, option)
