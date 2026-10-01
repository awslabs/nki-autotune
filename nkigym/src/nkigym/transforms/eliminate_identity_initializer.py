"""Eliminate identity fills superseded by a reduction's first write."""

from __future__ import annotations

from dataclasses import dataclass, replace

from nkigym.ir import Add, Const, Expr, FloorDiv, KernelIR, Mod, Mul, Var, substitute, to_affine
from nkigym.ir.arith.analyzer import Analyzer
from nkigym.ir.arith.expr import affine_terms, expr_variables
from nkigym.ir.program_sharding import PROGRAM_SHARDS_ANNOTATION, configured_program_shards
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode
from nkigym.ops.base import BilinearReductionContract, InitializerContract, ReductionContract
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


def _write_region(ir: KernelIR, node: ISANode, operand: str) -> BufferRegion | None:
    """Recover a rectangular PSUM view instead of its conservative allocation bounds."""
    region = node.operand_bindings.get(operand)
    pattern = node.access_patterns.get(operand)
    if region is not None and pattern is not None:
        buffer = ir.buffer(region.tensor)
        pitch = buffer.per_tile_physical_shape()[-1]
        if (
            buffer.location != "psum"
            or buffer.versions != 1
            or buffer.logical_tile_count() != 1
            or len(pattern.pattern) != 2
            or pattern.pattern[0][0] != Const(value=pitch)
            or pattern.pattern[1][0] != Const(value=1)
        ):
            return None
        region = BufferRegion(
            tensor=region.tensor,
            ranges=(
                (FloorDiv(left=pattern.offset, right=Const(value=pitch)), pattern.pattern[0][1]),
                (Mod(left=pattern.offset, right=Const(value=pitch)), pattern.pattern[1][1]),
            ),
        )
    return region


def _overwrite_is_bank_aligned(ir: KernelIR, node: ISANode, operand: str) -> bool:
    """Reject resets within a bank while preserving distinct logical tile slots."""
    if node.op_cls.NAME != "nc_matmul":
        return True
    region = _write_region(ir, node, operand)
    if region is None or ir.buffer(region.tensor).location != "psum":
        return False
    return _bank_aligned_free_offset(region.ranges[1][0])


def _bank_aligned_free_offset(offset: Expr) -> bool:
    """Prove alignment within one logical PSUM tile, in float32 elements."""
    analyzer = Analyzer()
    remainder = analyzer.simplify(Mod(left=offset, right=Const(value=512)))
    return remainder == Const(value=0) or all(
        coefficient % 512 == 0 for coefficient in affine_terms(analyzer.simplify(offset)).values()
    )


def _explicit_matmul_overwrite(node: ISANode) -> bool:
    """Identify an explicit reset rather than compiler-inferred accumulation."""
    accumulation = node.kwargs.get("accumulate")
    return node.op_cls.NAME == "nc_matmul" and (accumulation is False or isinstance(accumulation, tuple))


class EliminateIdentityInitializer(Transform[EliminateIdentityInitializerOption]):
    """Remove an identity fill after first-write overwrite is explicit."""

    def analyze(self, ir: KernelIR) -> list[EliminateIdentityInitializerOption]:
        """Return every identity initializer with proven overwrite semantics."""
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
        initializer_region = _write_region(ir, initializer, initializer_contract.output_operand)
        reduction_region = _write_region(ir, reduction, output_operand)
        if (
            initializer_region is None
            or reduction_region is None
            or initializer_region.tensor != option.tensor
            or reduction_region.tensor != option.tensor
            or initializer_contract.value != identity
            or not reduction.op_cls.first_write_overwrites(output_operand, reduction.kwargs)
            or not _overwrite_is_bank_aligned(ir, reduction, output_operand)
        ):
            return result
        initializer_execution_nid = self._initializer_execution(ir, initializer_block_nid, initializer_leaf_nid)
        buffer_owner = self._buffer_owner(ir, option.tensor)
        if (
            not self._matching_output_domains(
                ir, initializer_leaf_nid, reduction_leaf_nid, (initializer_region, reduction_region), reduction_axis
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
            and reduction.operand_bindings[output_operand] in reduction_block.reads
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
        regions: tuple[BufferRegion, BufferRegion],
        reduction_axis: str,
    ) -> bool:
        """Match output tiles and reset frequency within shared enclosing loops."""
        initializer_loops = self._ancestor_loops(ir, initializer_leaf_nid)
        reduction_loops = self._ancestor_loops(ir, reduction_leaf_nid)
        shared_loops = frozenset(initializer_loops.keys() & reduction_loops.keys())
        initializer_form = self._output_domain_form(ir, initializer_leaf_nid, regions[0], shared_loops)
        reduction_form = self._output_domain_form(ir, reduction_leaf_nid, regions[1], shared_loops)
        if initializer_form is None or initializer_form != reduction_form:
            return False
        initializer_repeats = [
            loop
            for nid, loop in initializer_loops.items()
            if nid not in shared_loops and loop.loop_var not in self._region_variables(regions[0])
        ]
        reduction_repeats = [
            loop
            for nid, loop in reduction_loops.items()
            if nid not in shared_loops and loop.loop_var not in self._region_variables(regions[1])
        ]
        reduction_block = ir.tree.block(owning_block(ir.tree, reduction_leaf_nid))
        concrete_axis = reduction_block.axis_map.get(reduction_axis, reduction_axis)
        shared_reduction = any(
            self._loop_binds_axis(reduction_block, reduction_loops[nid].loop_var, concrete_axis) for nid in shared_loops
        )
        return (
            not initializer_repeats
            and not shared_reduction
            and all(self._loop_binds_axis(reduction_block, loop.loop_var, concrete_axis) for loop in reduction_repeats)
        )

    def _output_domain_form(
        self, ir: KernelIR, leaf_nid: int, region: BufferRegion, shared_loops: frozenset[int]
    ) -> tuple[tuple[int, ...], tuple[tuple[Expr, Expr], ...]] | None:
        """Normalize private output loops while retaining shared loop identities."""
        loops = self._ancestor_loops(ir, leaf_nid)
        variables = self._region_variables(region)
        selected = [(nid, loop) for nid, loop in loops.items() if loop.loop_var in variables]
        if len(selected) != len(variables) or len({loop.loop_var for loop in loops.values()}) != len(loops):
            return None
        substitutions: dict[str, Expr] = {
            loop.loop_var: Var(name=f"_shared_loop_{nid}" if nid in shared_loops else f"_output_loop_{index}")
            for index, (nid, loop) in enumerate(selected)
        }
        analyzer = Analyzer()
        ranges = tuple(
            (analyzer.simplify(substitute(lower, substitutions)), analyzer.simplify(substitute(width, substitutions)))
            for lower, width in region.ranges
        )
        return tuple(loop.extent for _, loop in selected), ranges

    def _ancestor_loops(self, ir: KernelIR, leaf_nid: int) -> dict[int, ForNode]:
        """Return materialized loops enclosing one ISA leaf in execution order."""
        return {nid: loop for nid in ir.tree.ancestors(leaf_nid) if isinstance((loop := ir.tree.data(nid)), ForNode)}

    def _region_variables(self, region: BufferRegion) -> frozenset[str]:
        """Return loop variables used by one output region."""
        return frozenset(
            variable
            for lower, width in region.ranges
            for expression in (lower, width)
            for variable in expr_variables(expression)
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
        shards = configured_program_shards(ir)
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
        root = ir.tree.block(ir.tree.root)
        annotations = dict(root.annotations)
        annotations[PROGRAM_SHARDS_ANNOTATION] = {
            loop_nid: programs for loop_nid, programs in shards.items() if loop_nid in ir.tree.graph
        }
        ir.tree.graph.nodes[ir.tree.root]["data"] = replace(root, annotations=annotations)
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
