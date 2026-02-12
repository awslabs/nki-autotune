"""Operand merge analysis and transform for tiled compute graphs.

Identifies adjacent statement groups that operate on contiguous slices of
the same tensor and can be merged into a single wider operation. This
reduces the total number of load/store/compute statements, improving
hardware utilization.

Operates on the tuple-based program IR. Hardware tile limits are delegated
to ``op.can_merge_dim()`` rather than a hardcoded table.

Example::

    merge = OperandMergeTransform()
    opportunities = merge.analyze_ir(program)
    for opp in opportunities:
        program = merge.transform_ir(program, opp)
"""

from dataclasses import dataclass

from nkigym.ir import Operand, Program, Statement
from nkigym.ops import AllocOp, ElementwiseOp, LoadOp, NKIOp, StoreOp
from nkigym.transforms.base import Transform


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
        merged_slice: A ``(start, stop)`` tuple for the merged
            free-dimension range of the differing operand.
        description: Human-readable description for logging.
    """

    op_type: str
    stmt_a: int
    stmt_b: int
    differing_operand_idx: int
    merged_slice: tuple[int, int]
    description: str


class OperandMergeTransform(Transform):
    """Transform that merges adjacent operations on contiguous tensor slices.

    Identifies pairs of statements that perform the same operation but on
    adjacent slices of a tensor, differing on exactly one operand dimension.
    These can be combined into a single wider operation without exceeding
    hardware tile limits (checked via ``op.can_merge_dim()``).

    ``analyze_ir()`` returns a list of ``MergeOpportunity`` objects.
    ``transform_ir()`` applies one opportunity at a time.

    Example::

        merge = OperandMergeTransform()
        opportunities = merge.analyze_ir(program)
        for opp in opportunities:
            program = merge.transform_ir(program, opp)
    """

    name = "operand_merge"

    @staticmethod
    def _check_adjacent_slices(
        slices_a: tuple[tuple[int, int], ...], slices_b: tuple[tuple[int, int], ...]
    ) -> tuple[int, tuple[int, int]] | None:
        """Check if two slice tuples differ on exactly one dimension with adjacent ranges.

        Args:
            slices_a: Slice tuple from the first operand.
            slices_b: Slice tuple from the second operand.

        Returns:
            Tuple of ``(differing_dim_index, merged_slice)`` if adjacent,
            or ``None`` if not mergeable.
        """
        if len(slices_a) != len(slices_b):
            return None

        differing_dims: list[int] = []
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

    @staticmethod
    def _resolve_load_slices(var_name: str, stmts: tuple[Statement, ...]) -> tuple[tuple[int, int], ...] | None:
        """Resolve a variable name to its load source slices.

        Args:
            var_name: Variable name that might be a loaded tensor.
            stmts: All program statements.

        Returns:
            The source slices from the load, or None if not a load variable.
        """
        for op, operands in stmts:
            if isinstance(op, LoadOp) and operands[1][0] == var_name:
                return operands[0][1]
        return None

    @staticmethod
    def _find_load_idx(var_name: str, stmts: tuple[Statement, ...]) -> int | None:
        """Find the statement index of a load that produces a variable.

        Args:
            var_name: Variable name.
            stmts: All program statements.

        Returns:
            Statement index or None.
        """
        for i, (op, operands) in enumerate(stmts):
            if isinstance(op, LoadOp) and operands[1][0] == var_name:
                return i
        return None

    @staticmethod
    def _find_store_idx(var_name: str, stmts: tuple[Statement, ...]) -> int | None:
        """Find the statement index of a store that reads a variable.

        Args:
            var_name: Source variable name for the store.
            stmts: All program statements.

        Returns:
            Statement index or None.
        """
        for i, (op, operands) in enumerate(stmts):
            if isinstance(op, StoreOp) and operands[0][0] == var_name:
                return i
        return None

    @staticmethod
    def _get_store_for_var(var_name: str, stmts: tuple[Statement, ...]) -> tuple[int, tuple[Operand, ...]] | None:
        """Get the store statement for a computed variable.

        Args:
            var_name: Variable name produced by a compute op.
            stmts: All program statements.

        Returns:
            Tuple of (index, operands) or None.
        """
        for i, (op, operands) in enumerate(stmts):
            if isinstance(op, StoreOp) and operands[0][0] == var_name:
                return (i, operands)
        return None

    @staticmethod
    def _build_var_usage(stmts: tuple[Statement, ...]) -> dict[str, list[int]]:
        """Build index mapping variable names to all statement indices where they appear as operands.

        Args:
            stmts: All program statements.

        Returns:
            Dict mapping variable name to sorted list of statement indices
            that reference it (including the statement that defines it).
        """
        usage: dict[str, set[int]] = {}
        for i, (op, operands) in enumerate(stmts):
            for var, _ in operands:
                usage.setdefault(var, set()).add(i)
        return {var: sorted(indices) for var, indices in usage.items()}

    @staticmethod
    def _check_dependency_safe(idx_lo: int, idx_hi: int, absorbed_var: str, var_usage: dict[str, list[int]]) -> bool:
        """Check that merging two statements does not violate data dependencies.

        Args:
            idx_lo: Statement index of the kept (earlier) statement.
            idx_hi: Statement index of the absorbed (later) statement.
            absorbed_var: Variable name produced by the absorbed statement.
            var_usage: Pre-built usage index.

        Returns:
            True if merging is safe.
        """
        indices = var_usage.get(absorbed_var)
        if indices is None:
            return True
        for idx in indices:
            if idx <= idx_lo:
                continue
            if idx >= idx_hi:
                break
            return False
        return True

    @staticmethod
    def _resolve_operand_slices(operand: Operand, stmts: tuple[Statement, ...]) -> tuple[tuple[int, int], ...] | None:
        """Resolve operand slices, following through loads if needed.

        Args:
            operand: (var_name, slices) pair.
            stmts: All program statements.

        Returns:
            The effective slices, or None if unresolvable.
        """
        var, slices = operand
        if slices:
            return slices
        return OperandMergeTransform._resolve_load_slices(var, stmts)

    @staticmethod
    def _operand_signature(operand: Operand, stmts: tuple[Statement, ...]) -> tuple:
        """Create a hashable signature for an operand, resolving through loads.

        Args:
            operand: (var_name, slices) pair.
            stmts: All program statements.

        Returns:
            Hashable tuple for comparison.
        """
        var, slices = operand
        if slices:
            return (var, slices)
        load_slices = OperandMergeTransform._resolve_load_slices(var, stmts)
        if load_slices is not None:
            for op, operands in stmts:
                if isinstance(op, LoadOp) and operands[1][0] == var:
                    return ("load", operands[0][0], operands[0][1])
        return (var,)

    @staticmethod
    def _has_accumulation(var_name: str, stmts: tuple[Statement, ...], compute_vars: set[str]) -> bool:
        """Check if a compute variable is also the target of an accumulation.

        Args:
            var_name: Variable name produced by a compute op.
            stmts: All program statements.
            compute_vars: Set of variable names defined by compute ops so far.

        Returns:
            True if the variable is accumulated via a later compute stmt.
        """
        seen_first = False
        for op, operands in stmts:
            if isinstance(op, (LoadOp, StoreOp, AllocOp)):
                continue
            dst_var = operands[-1][0]
            if dst_var == var_name:
                if seen_first:
                    return True
                seen_first = True
        return False

    @staticmethod
    def _widen_slice(
        slices: tuple[tuple[int, int], ...], dim: int, merged: tuple[int, int]
    ) -> tuple[tuple[int, int], ...]:
        """Widen a single dimension in a slice tuple.

        Args:
            slices: Original slice tuple.
            dim: Dimension to widen.
            merged: New (start, stop) for that dimension.

        Returns:
            New slice tuple with the widened dimension.
        """
        return (*slices[:dim], merged, *slices[dim + 1 :])

    @staticmethod
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
            (s - merged_start, e - merged_start) if d == merge_dim else (0, e - s)
            for d, (s, e) in enumerate(abs_slices)
        )

    @staticmethod
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

    @staticmethod
    def _apply_subscript_map(
        operands: tuple[Operand, ...], subscript_map: dict[str, tuple[tuple[int, int], ...]]
    ) -> tuple[Operand, ...]:
        """Rewrite operand slices using a subscript offset map.

        For operands whose variable is in the map:
        - If the operand has no slices (bare name), use the relative slices directly.
        - If the operand has slices, offset them by the relative position.

        Args:
            operands: Original operands tuple.
            subscript_map: Mapping from variable name to relative slices.

        Returns:
            New operands tuple with rewritten slices.
        """
        new_ops: list[Operand] = []
        for var, slices in operands:
            if var in subscript_map:
                rel = subscript_map[var]
                if not slices:
                    new_ops.append((var, rel))
                else:
                    new_ops.append((var, OperandMergeTransform._offset_slices(rel, slices)))
            else:
                new_ops.append((var, slices))
        return tuple(new_ops)

    @staticmethod
    def _find_merge_opportunities(stmts: tuple[Statement, ...]) -> list[MergeOpportunity]:
        """Find all merge opportunities in a program's statements.

        Args:
            stmts: Program statements tuple.

        Returns:
            List of MergeOpportunity objects.
        """
        var_usage = OperandMergeTransform._build_var_usage(stmts)
        opportunities: list[MergeOpportunity] = []

        compute_vars: set[str] = set()
        for op, operands in stmts:
            if not isinstance(op, (LoadOp, StoreOp, AllocOp)):
                compute_vars.add(operands[-1][0])

        opportunities.extend(OperandMergeTransform._find_load_merge_opportunities(stmts, var_usage))
        opportunities.extend(OperandMergeTransform._find_compute_merge_opportunities(stmts, var_usage, compute_vars))

        return opportunities

    @staticmethod
    def _find_load_merge_opportunities(
        stmts: tuple[Statement, ...], var_usage: dict[str, list[int]]
    ) -> list[MergeOpportunity]:
        """Find load merge opportunities.

        Args:
            stmts: Program statements.
            var_usage: Pre-built usage index.

        Returns:
            List of MergeOpportunity for loads.
        """
        loads: list[tuple[int, tuple[Operand, ...]]] = []
        for i, (op, operands) in enumerate(stmts):
            if isinstance(op, LoadOp):
                loads.append((i, operands))

        if len(loads) < 2:
            return []

        load_groups: dict[tuple, list[tuple[int, tuple[Operand, ...]]]] = {}
        for idx, operands in loads:
            src_var, src_slices = operands[0]
            if len(src_slices) < 2:
                continue
            partition_key = (src_var, src_slices[0])
            load_groups.setdefault(partition_key, []).append((idx, operands))

        used: set[int] = set()
        opportunities: list[MergeOpportunity] = []

        for members in load_groups.values():
            if len(members) < 2:
                continue

            ordered = sorted(members, key=lambda m: m[1][0][1][-1][0])

            for i in range(len(ordered)):
                if ordered[i][0] in used:
                    continue
                for j in range(i + 1, len(ordered)):
                    if ordered[j][0] in used:
                        continue

                    idx_a, ops_a = ordered[i]
                    idx_b, ops_b = ordered[j]
                    src_slices_a = ops_a[0][1]
                    src_slices_b = ops_b[0][1]

                    adj = OperandMergeTransform._check_adjacent_slices(src_slices_a, src_slices_b)
                    if adj is None:
                        continue

                    dim, merged = adj

                    first_idx = min(idx_a, idx_b)
                    second_idx = max(idx_a, idx_b)
                    first_ops = ops_a if idx_a <= idx_b else ops_b
                    second_ops = ops_b if idx_a <= idx_b else ops_a

                    if not OperandMergeTransform._check_dependency_safe(
                        first_idx, second_idx, second_ops[1][0], var_usage
                    ):
                        continue

                    opportunities.append(
                        MergeOpportunity(
                            op_type="load",
                            stmt_a=first_idx,
                            stmt_b=second_idx,
                            differing_operand_idx=dim,
                            merged_slice=merged,
                            description=(
                                f"Merge load {first_ops[1][0]} and {second_ops[1][0]} "
                                f"[dim {dim}: {src_slices_a[dim]} + {src_slices_b[dim]} -> {merged}]"
                            ),
                        )
                    )
                    used.add(first_idx)
                    used.add(second_idx)
                    break

        return opportunities

    @staticmethod
    def _find_compute_merge_opportunities(
        stmts: tuple[Statement, ...], var_usage: dict[str, list[int]], compute_vars: set[str]
    ) -> list[MergeOpportunity]:
        """Find compute op merge opportunities.

        Args:
            stmts: Program statements.
            var_usage: Pre-built usage index.
            compute_vars: Set of compute variable names.

        Returns:
            List of MergeOpportunity for compute ops.
        """
        compute_stmts: list[tuple[int, NKIOp, tuple[Operand, ...]]] = []
        for i, (op, operands) in enumerate(stmts):
            if isinstance(op, (LoadOp, StoreOp, AllocOp)):
                continue
            compute_stmts.append((i, op, operands))

        if len(compute_stmts) < 2:
            return []

        op_groups: dict[str, list[tuple[int, NKIOp, tuple[Operand, ...]]]] = {}
        for entry in compute_stmts:
            op_groups.setdefault(entry[1].op_name, []).append(entry)

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

                    opp = OperandMergeTransform._check_compute_pair(
                        members[i], members[j], stmts, var_usage, compute_vars
                    )
                    if opp is not None:
                        opportunities.append(opp)
                        used.add(opp.stmt_a)
                        used.add(opp.stmt_b)
                        break

        return opportunities

    @staticmethod
    def _check_compute_pair(
        entry_a: tuple[int, NKIOp, tuple[Operand, ...]],
        entry_b: tuple[int, NKIOp, tuple[Operand, ...]],
        stmts: tuple[Statement, ...],
        var_usage: dict[str, list[int]],
        compute_vars: set[str],
    ) -> MergeOpportunity | None:
        """Check if two compute statements can be merged.

        Args:
            entry_a: (index, op, operands) for first compute.
            entry_b: (index, op, operands) for second compute.
            stmts: All program statements.
            var_usage: Pre-built usage index.
            compute_vars: Set of compute variable names.

        Returns:
            MergeOpportunity or None.
        """
        idx_a, op_a, operands_a = entry_a
        idx_b, op_b, operands_b = entry_b

        if isinstance(op_a, ElementwiseOp) or isinstance(op_b, ElementwiseOp):
            if op_a != op_b:
                return None

        input_ops_a = operands_a[:-1]
        input_ops_b = operands_b[:-1]

        if len(input_ops_a) != len(input_ops_b):
            return None

        differing_args: list[int] = []
        for k in range(len(input_ops_a)):
            if OperandMergeTransform._operand_signature(
                input_ops_a[k], stmts
            ) != OperandMergeTransform._operand_signature(input_ops_b[k], stmts):
                differing_args.append(k)

        if len(differing_args) != 1:
            return None

        diff_idx = differing_args[0]

        slices_a = OperandMergeTransform._resolve_operand_slices(input_ops_a[diff_idx], stmts)
        slices_b = OperandMergeTransform._resolve_operand_slices(input_ops_b[diff_idx], stmts)

        if slices_a is None or slices_b is None:
            return None

        adj = OperandMergeTransform._check_adjacent_slices(slices_a, slices_b)
        if adj is None:
            return None

        dim, merged = adj
        merged_size = merged[1] - merged[0]

        if not op_a.can_merge_operand_dim(diff_idx, dim, merged_size):
            return None

        var_a = operands_a[-1][0]
        var_b = operands_b[-1][0]

        store_a = OperandMergeTransform._get_store_for_var(var_a, stmts)
        store_b = OperandMergeTransform._get_store_for_var(var_b, stmts)
        if store_a and store_b:
            s_adj = OperandMergeTransform._check_adjacent_slices(store_a[1][1][1], store_b[1][1][1])
            if s_adj is None:
                return None

        if OperandMergeTransform._has_accumulation(
            var_a, stmts, compute_vars
        ) or OperandMergeTransform._has_accumulation(var_b, stmts, compute_vars):
            return None

        first_idx = min(idx_a, idx_b)
        second_idx = max(idx_a, idx_b)
        second_var = var_b if idx_a <= idx_b else var_a

        if not OperandMergeTransform._check_dependency_safe(first_idx, second_idx, second_var, var_usage):
            return None

        return MergeOpportunity(
            op_type=op_a.op_name,
            stmt_a=first_idx,
            stmt_b=second_idx,
            differing_operand_idx=diff_idx,
            merged_slice=merged,
            description=(
                f"Merge {op_a.op_name} at stmt {first_idx} and "
                f"{second_idx} on arg {diff_idx} "
                f"[dim {dim}: {slices_a[dim]} + {slices_b[dim]} -> {merged}]"
            ),
        )

    def analyze_ir(self, program: Program) -> list[MergeOpportunity]:
        """Analyze a program to find merge opportunities.

        Args:
            program: Program tuple (name, params, stmts, return_var).

        Returns:
            List of MergeOpportunity objects.
        """
        stmts = program.stmts
        return self._find_merge_opportunities(stmts)

    def transform_ir(self, program: Program, option: MergeOpportunity) -> Program:
        """Apply a single merge opportunity.

        Args:
            program: Program tuple to transform.
            option: A single MergeOpportunity from ``analyze_ir()``.

        Returns:
            New program tuple with the merged operation applied.
        """
        name, params, stmts, return_var, preamble = program

        stmt_a_op, stmt_a_operands = stmts[option.stmt_a]
        stmt_b_op, stmt_b_operands = stmts[option.stmt_b]

        to_remove: set[int] = {option.stmt_b}
        renames: dict[str, str] = {}

        if option.op_type == "load":
            return self._apply_load_merge(name, params, stmts, return_var, preamble, option)

        var_a = stmt_a_operands[-1][0]
        var_b = stmt_b_operands[-1][0]
        renames[var_b] = var_a

        diff_idx = option.differing_operand_idx
        arg_b = stmt_b_operands[diff_idx]
        if not arg_b[1]:
            load_idx = self._find_load_idx(arg_b[0], stmts)
            if load_idx is not None:
                to_remove.add(load_idx)

        store_b_idx = self._find_store_idx(var_b, stmts)
        if store_b_idx is not None:
            to_remove.add(store_b_idx)

        widen_load_idx: int | None = None
        widen_load_dim: int | None = None
        arg_a = stmt_a_operands[diff_idx]
        if not arg_a[1]:
            load_a_idx = self._find_load_idx(arg_a[0], stmts)
            load_b_idx = self._find_load_idx(arg_b[0], stmts) if not arg_b[1] else None
            if load_a_idx is not None and load_b_idx is not None:
                load_a_src = stmts[load_a_idx][1][0][1]
                load_b_src = stmts[load_b_idx][1][0][1]
                adj = self._check_adjacent_slices(load_a_src, load_b_src)
                if adj:
                    widen_load_idx = load_a_idx
                    widen_load_dim = adj[0]

        store_a_info = self._get_store_for_var(var_a, stmts)
        store_b_info = self._get_store_for_var(var_b, stmts)

        new_stmts: list[Statement] = []
        for i, (op, operands) in enumerate(stmts):
            if i in to_remove:
                continue

            if i == option.stmt_a:
                operands = self._widen_compute_stmt(operands, option, stmts)

            if widen_load_idx is not None and i == widen_load_idx and widen_load_dim is not None:
                src_var, src_slices = operands[0]
                new_src_slices = self._widen_slice(src_slices, widen_load_dim, option.merged_slice)
                dst_var, dst_slices = operands[1]
                new_dst_shape = tuple(stop - start for start, stop in new_src_slices)
                new_dst_slices = tuple((0, s) for s in new_dst_shape)
                operands = ((src_var, new_src_slices), (dst_var, new_dst_slices))

            if store_a_info and i == store_a_info[0] and store_b_info:
                operands = self._widen_store(operands, store_a_info[1], store_b_info[1])

            if renames:
                operands = tuple((renames.get(var, var), slices) for var, slices in operands)

            new_stmts.append((op, operands))

        return Program(name, params, tuple(new_stmts), return_var, preamble)

    def _apply_load_merge(
        self,
        prog_name: str,
        params: tuple[str, ...],
        stmts: tuple[Statement, ...],
        return_var: str,
        preamble: str,
        option: MergeOpportunity,
    ) -> Program:
        """Apply a load merge opportunity.

        Args:
            prog_name: Program name.
            params: Program parameters.
            stmts: Current statements.
            return_var: Return variable name.
            preamble: Original function preamble.
            option: The load merge opportunity.

        Returns:
            New program tuple.
        """
        stmt_a_op, stmt_a_operands = stmts[option.stmt_a]
        stmt_b_op, stmt_b_operands = stmts[option.stmt_b]

        merge_dim = option.differing_operand_idx
        merged_start = option.merged_slice[0]

        var_a = stmt_a_operands[1][0]
        var_b = stmt_b_operands[1][0]

        subscript_map: dict[str, tuple[tuple[int, int], ...]] = {
            var_a: self._relative_slices(stmt_a_operands[0][1], merge_dim, merged_start),
            var_b: self._relative_slices(stmt_b_operands[0][1], merge_dim, merged_start),
        }

        renames = {var_b: var_a}
        to_remove = {option.stmt_b}

        new_stmts: list[Statement] = []
        for i, (op, operands) in enumerate(stmts):
            if i in to_remove:
                continue

            if i == option.stmt_a:
                src_var, src_slices = operands[0]
                new_src_slices = self._widen_slice(src_slices, merge_dim, option.merged_slice)
                dst_var, dst_slices = operands[1]
                new_dst_shape = tuple(stop - start for start, stop in new_src_slices)
                new_dst_slices = tuple((0, s) for s in new_dst_shape)
                operands = ((src_var, new_src_slices), (dst_var, new_dst_slices))
            else:
                operands = self._apply_subscript_map(operands, subscript_map)

            operands = tuple((renames.get(var, var), slices) for var, slices in operands)
            new_stmts.append((op, operands))

        return Program(prog_name, params, tuple(new_stmts), return_var, preamble)

    def _widen_compute_stmt(
        self, operands: tuple[Operand, ...], option: MergeOpportunity, stmts: tuple[Statement, ...]
    ) -> tuple[Operand, ...]:
        """Widen the kept compute statement's differing operand.

        Args:
            operands: Original operands of the kept stmt.
            option: The merge opportunity.
            stmts: All program statements.

        Returns:
            New operands tuple with widened slice.
        """
        diff_idx = option.differing_operand_idx
        var, slices = operands[diff_idx]

        if slices:
            orig_a = stmts[option.stmt_a][1][diff_idx][1]
            orig_b = stmts[option.stmt_b][1][diff_idx][1]
            for d in range(len(orig_a)):
                if orig_a[d] != orig_b[d]:
                    new_slices = self._widen_slice(slices, d, option.merged_slice)
                    new_operands = list(operands)
                    new_operands[diff_idx] = (var, new_slices)
                    return tuple(new_operands)

        return operands

    def _widen_store(
        self,
        operands: tuple[Operand, ...],
        store_a_operands: tuple[Operand, ...],
        store_b_operands: tuple[Operand, ...],
    ) -> tuple[Operand, ...]:
        """Widen the kept store's target and value slices.

        Args:
            operands: Current store operands.
            store_a_operands: Original operands of store A.
            store_b_operands: Original operands of store B.

        Returns:
            New operands with widened slices.
        """
        src_var, src_slices = operands[0]
        dst_var, dst_slices = operands[1]
        target_a = store_a_operands[1][1]
        target_b = store_b_operands[1][1]

        for d in range(len(target_a)):
            if target_a[d] != target_b[d]:
                merged_target = (min(target_a[d][0], target_b[d][0]), max(target_a[d][1], target_b[d][1]))
                new_dst_slices = self._widen_slice(dst_slices, d, merged_target)

                if src_slices:
                    vs, ve = src_slices[d]
                    merged_value = (vs, vs + (merged_target[1] - merged_target[0]))
                    new_src_slices = self._widen_slice(src_slices, d, merged_value)
                    return ((src_var, new_src_slices), (dst_var, new_dst_slices))
                return ((src_var, src_slices), (dst_var, new_dst_slices))

        return operands
