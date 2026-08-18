"""Eliminate identity fills superseded by a reduction's first write."""

from __future__ import annotations

from dataclasses import dataclass, replace

from nkigym.ir import Expr, KernelIR, Var, substitute, to_affine
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode
from nkigym.ops.base import BilinearReductionContract, InitializerContract, ReductionContract
from nkigym.search.state_facts import operation_facts
from nkigym.transforms.base import (
    Transform,
    TransformLegalityError,
    TransformOption,
    copy_for_rewrite,
    intersects_software_pipeline,
    software_pipeline_overlap_nodes,
)
from nkigym.transforms.helper.canonical_rewrite import finalize_rewrite, owning_block, single_leaf
from nkigym.transforms.helper.tree_ops import _replace_in_parent_children


@dataclass(frozen=True)
class EliminateIdentityInitializerOption(TransformOption):
    """Identify one removable initializer and its reduction block."""

    initializer_block_nid: int
    reduction_block_nid: int
    tensor: str
    initializer_leaf_nid: int | None = None


@dataclass(frozen=True)
class _InitializerMatch:
    """Resolved leaves and execution subtree for one elimination option."""

    option: EliminateIdentityInitializerOption
    initializer_leaf_nid: int
    reduction_leaf_nid: int
    initializer_execution_nid: int
    output_operand: str
    reduction_axis: str


class EliminateIdentityInitializer(Transform[EliminateIdentityInitializerOption]):
    """Remove an identity fill after first-write overwrite is explicit."""

    def analyze(self, ir: KernelIR) -> list[EliminateIdentityInitializerOption]:
        """Return every identity initializer with proven overwrite semantics."""
        facts = operation_facts(ir)
        if not facts.has_initializer or not facts.has_reduction:
            return []
        options: list[EliminateIdentityInitializerOption] = []
        overlap_nodes = software_pipeline_overlap_nodes(ir)
        for initializer_leaf_nid in ir.tree.preorder():
            if not isinstance(ir.tree.data(initializer_leaf_nid), ISANode):
                continue
            option = self._candidate_option(ir, initializer_leaf_nid)
            if option is not None and self._resolve(ir, option, explicit=True, overlap_nodes=overlap_nodes) is not None:
                options.append(option)
        return options

    def apply(self, ir: KernelIR, option: EliminateIdentityInitializerOption) -> KernelIR:
        """Recheck, copy, remove the initializer block, and rebuild metadata."""
        match = self._resolve(ir, option, explicit=True)
        if match is None:
            raise TransformLegalityError(f"illegal EliminateIdentityInitializer option: {option}")
        new_ir = copy_for_rewrite(ir)
        copied_match = self._resolve(new_ir, option, explicit=True)
        if copied_match is None:
            raise AssertionError(f"EliminateIdentityInitializer option disappeared after deepcopy: {option}")
        self._remove_initializer(new_ir, copied_match)
        return new_ir

    def _candidate_option(self, ir: KernelIR, initializer_leaf_nid: int) -> EliminateIdentityInitializerOption | None:
        """Construct an option when the next tensor touch is a reduction."""
        option: EliminateIdentityInitializerOption | None = None
        if initializer_leaf_nid in ir.tree.graph and isinstance(ir.tree.data(initializer_leaf_nid), ISANode):
            initializer = ir.tree.isa(initializer_leaf_nid)
            contract = initializer.op_cls.algebraic_contract(initializer.kwargs)
            if isinstance(contract, InitializerContract):
                region = initializer.operand_bindings.get(contract.output_operand)
                if region is not None:
                    ordered = ir.dependency.touches_by_tensor.get(region.tensor, ())
                    if initializer_leaf_nid in ordered:
                        index = ordered.index(initializer_leaf_nid)
                        if index + 1 < len(ordered):
                            reduction_leaf_nid = ordered[index + 1]
                            option = EliminateIdentityInitializerOption(
                                initializer_block_nid=owning_block(ir.tree, initializer_leaf_nid),
                                reduction_block_nid=owning_block(ir.tree, reduction_leaf_nid),
                                tensor=region.tensor,
                                initializer_leaf_nid=initializer_leaf_nid,
                            )
        return option

    def _resolve(
        self,
        ir: KernelIR,
        option: EliminateIdentityInitializerOption,
        explicit: bool,
        overlap_nodes: frozenset[int] | None = None,
    ) -> _InitializerMatch | None:
        """Resolve a one-step identity/reduction pair before or after marking."""
        result: _InitializerMatch | None = None
        initializer_block_nid = option.initializer_block_nid
        reduction_block_nid = option.reduction_block_nid
        if initializer_block_nid not in ir.tree.graph or reduction_block_nid not in ir.tree.graph:
            return result
        if not isinstance(ir.tree.data(initializer_block_nid), BlockNode) or not isinstance(
            ir.tree.data(reduction_block_nid), BlockNode
        ):
            return result
        if intersects_software_pipeline(ir, (initializer_block_nid, reduction_block_nid), overlap_nodes):
            return result
        initializer_leaf_nid = option.initializer_leaf_nid
        if initializer_leaf_nid is None:
            initializer_leaf_nid = single_leaf(ir.tree, initializer_block_nid)
        reduction_leaf_nid = single_leaf(ir.tree, reduction_block_nid)
        if (
            initializer_leaf_nid is None
            or initializer_leaf_nid not in ir.tree.graph
            or owning_block(ir.tree, initializer_leaf_nid) != initializer_block_nid
            or reduction_leaf_nid is None
        ):
            return result
        initializer = ir.tree.isa(initializer_leaf_nid)
        reduction = ir.tree.isa(reduction_leaf_nid)
        initializer_contract = initializer.op_cls.algebraic_contract(initializer.kwargs)
        reduction_contract = reduction.op_cls.algebraic_contract(reduction.kwargs)
        reduction_fields = self._reduction_fields(reduction_contract)
        if not isinstance(initializer_contract, InitializerContract) or reduction_fields is None:
            return result
        output_operand, reduction_axis, identity = reduction_fields
        initializer_region = initializer.operand_bindings.get(initializer_contract.output_operand)
        reduction_region = reduction.operand_bindings.get(output_operand)
        if (
            initializer_region is None
            or reduction_region is None
            or initializer_region.tensor != option.tensor
            or reduction_region != initializer_region
            or initializer_contract.value != identity
            or not reduction.op_cls.first_write_overwrites(output_operand, reduction.kwargs)
        ):
            return result
        initializer_execution_nid = self._initializer_execution(ir, initializer_block_nid, initializer_leaf_nid)
        buffer_owner = self._buffer_owner(ir, option.tensor)
        if (
            not self._matching_output_domains(
                ir, initializer_leaf_nid, reduction_leaf_nid, initializer_region, reduction_axis
            )
            or buffer_owner is None
            or buffer_owner not in ir.tree.ancestors(initializer_leaf_nid)
            or buffer_owner not in ir.tree.ancestors(reduction_leaf_nid)
            or not self._first_touches_match(ir, option.tensor, initializer_leaf_nid, reduction_leaf_nid)
        ):
            return result
        reduction_block = ir.tree.block(reduction_block_nid)
        rmw = output_operand in reduction.op_cls.rmw_operands(reduction.kwargs)
        configured = reduction.kwargs.get("accumulate") == (reduction_axis,)
        state_matches = (
            (configured if explicit else "accumulate" not in reduction.kwargs)
            and rmw
            and reduction_region in reduction_block.reads
        )
        if not state_matches:
            return result
        result = _InitializerMatch(
            option=option,
            initializer_leaf_nid=initializer_leaf_nid,
            reduction_leaf_nid=reduction_leaf_nid,
            initializer_execution_nid=initializer_execution_nid,
            output_operand=output_operand,
            reduction_axis=reduction_axis,
        )
        return result

    def _reduction_fields(self, contract: object) -> tuple[str, str, float] | None:
        """Return output, reduction axis, and identity for supported contracts."""
        result: tuple[str, str, float] | None = None
        if isinstance(contract, (BilinearReductionContract, ReductionContract)):
            result = (contract.output_operand, contract.reduction_axis, contract.combinator.identity)
        return result

    def _enclosing_block(self, ir: KernelIR, nid: int) -> int | None:
        """Return the nearest block above one loop."""
        return next(
            (
                ancestor
                for ancestor in reversed(ir.tree.ancestors(nid))
                if isinstance(ir.tree.data(ancestor), BlockNode)
            ),
            None,
        )

    def _simple_initializer_loop_nest(self, ir: KernelIR, block_nid: int, leaf_nid: int) -> bool:
        """Return whether one initializer is wrapped only by loops."""
        return all(
            descendant == leaf_nid or isinstance(ir.tree.data(descendant), ForNode)
            for descendant in ir.tree.descendants(block_nid)
        )

    def _initializer_execution(self, ir: KernelIR, block_nid: int, leaf_nid: int) -> int:
        """Return the loop child that executes only the selected initializer."""
        return block_nid if self._simple_initializer_loop_nest(ir, block_nid, leaf_nid) else leaf_nid

    def _matching_output_domains(
        self,
        ir: KernelIR,
        initializer_leaf_nid: int,
        reduction_leaf_nid: int,
        region: BufferRegion,
        reduction_axis: str,
    ) -> bool:
        """Return whether initializer and reduction cover the same output tiles."""
        initializer_form = self._output_domain_form(ir, initializer_leaf_nid, region)
        reduction_form = self._output_domain_form(ir, reduction_leaf_nid, region)
        if initializer_form is None or initializer_form != reduction_form:
            return False
        initializer_loops = self._ancestor_loops(ir, initializer_leaf_nid)
        reduction_loops = self._ancestor_loops(ir, reduction_leaf_nid)
        output_variables = self._region_variables(region)
        initializer_repeats = [loop for loop in initializer_loops.values() if loop.loop_var not in output_variables]
        reduction_repeats = [loop for loop in reduction_loops.values() if loop.loop_var not in output_variables]
        reduction_block = ir.tree.block(owning_block(ir.tree, reduction_leaf_nid))
        concrete_axis = reduction_block.axis_map.get(reduction_axis, reduction_axis)
        return not initializer_repeats and all(
            self._loop_binds_axis(reduction_block, loop.loop_var, concrete_axis) for loop in reduction_repeats
        )

    def _output_domain_form(
        self, ir: KernelIR, leaf_nid: int, region: BufferRegion
    ) -> tuple[tuple[int, ...], tuple[tuple[Expr, Expr], ...]] | None:
        """Return an alpha-normalized output iteration domain."""
        loops = self._ancestor_loops(ir, leaf_nid)
        variables = self._region_variables(region)
        selected = [loop for loop in loops.values() if loop.loop_var in variables]
        if len(selected) != len(variables):
            return None
        substitutions: dict[str, Expr] = {
            loop.loop_var: Var(name=f"_output_loop_{index}") for index, loop in enumerate(selected)
        }
        ranges = tuple(
            (substitute(lower, substitutions), substitute(width, substitutions)) for lower, width in region.ranges
        )
        return tuple(loop.extent for loop in selected), ranges

    def _ancestor_loops(self, ir: KernelIR, leaf_nid: int) -> dict[int, ForNode]:
        """Return materialized loops enclosing one ISA leaf in execution order."""
        return {nid: loop for nid in ir.tree.ancestors(leaf_nid) if isinstance((loop := ir.tree.data(nid)), ForNode)}

    def _region_variables(self, region: BufferRegion) -> frozenset[str]:
        """Return loop variables used by one output region."""
        return frozenset(
            variable
            for lower, width in region.ranges
            for expression in (lower, width)
            for variable in to_affine(expression)
            if variable is not None
        )

    def _loop_binds_axis(self, block: BlockNode, loop_var: str, axis: str) -> bool:
        """Return whether one materialized loop contributes to ``axis``."""
        return any(
            iter_var.axis == axis and loop_var in to_affine(value)
            for iter_var, value in zip(block.iter_vars, block.iter_values)
        )

    def _buffer_owner(self, ir: KernelIR, tensor: str) -> int | None:
        """Return the unique block declaring ``tensor``."""
        owners = [
            block_nid
            for block_nid in ir.tree.blocks()
            if any(buffer.name == tensor for buffer in ir.tree.block(block_nid).alloc_buffers)
        ]
        return owners[0] if len(owners) == 1 else None

    def _first_touches_match(
        self, ir: KernelIR, tensor: str, initializer_leaf_nid: int, reduction_leaf_nid: int
    ) -> bool:
        """Return whether the initializer and reduction are the first two touches."""
        touches = set(ir.dependency.touches_by_tensor.get(tensor, ()))
        preorder = [nid for nid in ir.tree.preorder() if nid in touches]
        return len(preorder) >= 2 and preorder[:2] == [initializer_leaf_nid, reduction_leaf_nid]

    def _remove_initializer(self, ir: KernelIR, match: _InitializerMatch) -> None:
        """Remove one identity fill already superseded by an explicit overwrite."""
        block_nid = match.option.initializer_block_nid
        execution_nid = match.initializer_execution_nid
        parent = ir.tree.parent(execution_nid)
        if parent is None:
            raise AssertionError(f"initializer execution {execution_nid} has no parent")
        removed = {execution_nid, *ir.tree.descendants(execution_nid)}
        _replace_in_parent_children(ir.tree, parent, [execution_nid], [])
        ir.tree.graph.remove_nodes_from(removed)
        if execution_nid != block_nid:
            self._prune_empty_loops(ir, parent, block_nid)
            block = ir.tree.block(block_nid)
            ir.tree.graph.nodes[block_nid]["data"] = replace(
                block, iter_vars=(), iter_values=(), reads=(), writes=(), axis_map={}
            )
        finalize_rewrite(ir)

    def _prune_empty_loops(self, ir: KernelIR, nid: int, stop_nid: int) -> None:
        """Remove empty initializer-only loops below the owning block."""
        current = nid
        while current != stop_nid and isinstance(ir.tree.data(current), ForNode) and not ir.tree.children(current):
            parent = ir.tree.parent(current)
            if parent is None:
                raise AssertionError(f"empty initializer loop {current} has no parent")
            ir.tree.graph.remove_node(current)
            current = parent


__all__ = ["EliminateIdentityInitializer", "EliminateIdentityInitializerOption"]
