"""Shared utilities for operand merge: index, slice ops, and ref operations."""

from dataclasses import dataclass
from typing import Any

from nkigym.ir import GymProgram, GymStatement, TensorRef
from nkigym.ops.tiling_ops import LoadOp, StoreOp


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
    raise ValueError(f"{stmt.op.op_name} statement missing '{key}' kwarg")


def _check_adjacent_slices(
    slices_a: tuple[tuple[int, int], ...], slices_b: tuple[tuple[int, int], ...]
) -> tuple[int, tuple[int, int]]:
    """Check if two slice tuples differ on exactly one dimension with adjacent ranges.

    Args:
        slices_a: Slice tuple from the first operand.
        slices_b: Slice tuple from the second operand.

    Returns:
        Tuple of (differing_dim_index, merged_slice) if adjacent,
        or ``(-1, (0, 0))`` sentinel if not mergeable.
    """
    result: tuple[int, tuple[int, int]] = (-1, (0, 0))
    if len(slices_a) == len(slices_b):
        differing_dims = [d for d in range(len(slices_a)) if slices_a[d] != slices_b[d]]
        if len(differing_dims) == 1:
            dim = differing_dims[0]
            sa_start, sa_stop = slices_a[dim]
            sb_start, sb_stop = slices_b[dim]
            if sa_stop == sb_start:
                result = (dim, (sa_start, sb_stop))
            elif sb_stop == sa_start:
                result = (dim, (sb_start, sa_stop))
    return result


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


def _offset_slices(
    rel_slices: tuple[tuple[int, int], ...], existing_slices: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, int], ...]:
    """Offset existing subscript slices by relative position.

    Args:
        rel_slices: Relative slices providing the base offset per dimension.
        existing_slices: Existing subscript slices to offset.

    Returns:
        New slices with offset applied.
    """
    return tuple((rel_s + sub_s, rel_s + sub_e) for (rel_s, _), (sub_s, sub_e) in zip(rel_slices, existing_slices))


def _axis_dim(axes: tuple[str | int, ...], axis_name: str) -> int:
    """Find the dimension index for a named axis.

    Args:
        axes: Axis tuple from a Tensor descriptor.
        axis_name: Named axis to find.

    Returns:
        Dimension index, or ``-1`` if the axis is not present.
    """
    result = -1
    for d, ax in enumerate(axes):
        if ax == axis_name:
            result = d
            break
    return result


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
            if stmt.op is LoadOp:
                loads_by_output[stmt.output.name] = i
            elif stmt.op is StoreOp:
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

    def load_src_slices(self, var_name: str) -> tuple[tuple[int, int], ...]:
        """Get the source slices for a load that produces var_name.

        Args:
            var_name: Variable name that might be a loaded tensor.

        Returns:
            The source slices from the load, or empty tuple if not a load.
        """
        idx = self.loads_by_output.get(var_name, -1)
        result: tuple[tuple[int, int], ...] = ()
        if idx >= 0:
            result = _get_kwarg_ref(self.stmts[idx], "src").slices
        return result

    def resolve_operand_slices(self, ref: TensorRef) -> tuple[tuple[int, int], ...]:
        """Resolve operand slices to source-tensor coordinates.

        Composes the ref's slices with the load's source slices so
        that the result is always in source-tensor coordinates.

        Args:
            ref: TensorRef for the operand.

        Returns:
            Source-tensor coordinates.
        """
        load_src = self.load_src_slices(ref.name)
        result = ref.slices
        if load_src:
            result = tuple((ls + rs, ls + re) for (ls, _), (rs, re) in zip(load_src, ref.slices))
        return result

    def canonical_name(self, ref: TensorRef) -> tuple:
        """Create a hashable canonical name, resolving through loads.

        For loaded variables, returns ``(source_name, source_slices)``.
        For non-loaded variables, returns ``(variable_name,)``.

        Args:
            ref: TensorRef for the operand.

        Returns:
            Hashable canonical form.
        """
        load_idx = self.loads_by_output.get(ref.name, -1)
        result: tuple = (ref.name,)
        if load_idx >= 0:
            src = _get_kwarg_ref(self.stmts[load_idx], "src")
            result = (src.name, src.slices)
        return result


@dataclass
class MergeOpportunity:
    """A single merge opportunity found by ``analyze_ir()``.

    Attributes:
        stmt_a: The statement index of the first (kept) operation.
        stmt_b: The statement index of the second operation (absorbed).
        differing_operand_idx: Kwarg index of the first differing TensorRef.
        differing_dim: The dimension index within that kwarg's slices
            that differs between the two operations.
        merged_slice: A ``(start, stop)`` tuple for the merged range.
        merge_axis: Named axis all diffs map to (e.g., ``"N"``, ``"F"``).
        description: Human-readable description for logging.
    """

    stmt_a: int
    stmt_b: int
    differing_operand_idx: int
    differing_dim: int
    merged_slice: tuple[int, int]
    merge_axis: str
    description: str


def _replace_stmts(program: GymProgram, new_stmts: tuple[GymStatement, ...]) -> GymProgram:
    """Create a new GymProgram with replaced statements.

    Args:
        program: Original program.
        new_stmts: New statement tuple.

    Returns:
        New GymProgram with updated statements.
    """
    return GymProgram(program.name, program.kwargs, new_stmts, program.return_var, program.output_dtype)


def _remap_ref(ref: TensorRef, subscript_map: dict[str, tuple[tuple[int, int], ...]]) -> TensorRef:
    """Remap a TensorRef using a subscript offset map.

    Offsets the ref's slices by the relative position in the map.
    The ref's shape is preserved.

    Args:
        ref: TensorRef to remap.
        subscript_map: Mapping from variable name to relative slices.

    Returns:
        Remapped TensorRef, or original if not in map.
    """
    result = ref
    if ref.name in subscript_map:
        rel = subscript_map[ref.name]
        new_slices = _offset_slices(rel, ref.slices)
        result = TensorRef(ref.name, ref.shape, new_slices)
    return result


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
    result = ref
    if ref.name in renames:
        result = TensorRef(renames[ref.name], ref.shape, ref.slices)
    return result


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
