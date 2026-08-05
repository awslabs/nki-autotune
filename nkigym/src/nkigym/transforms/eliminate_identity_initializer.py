"""Eliminate identity fills superseded by a reduction's first write."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from nkigym.ir import KernelIR, to_affine
from nkigym.ir.tree import BlockNode, ForNode
from nkigym.ops.base import BilinearReductionContract, InitializerContract, ReductionContract
from nkigym.transforms._canonical_rewrite import finalize_rewrite, owning_block, single_leaf
from nkigym.transforms._tree_ops import _replace_in_parent_children
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class EliminateIdentityInitializerOption(TransformOption):
    """Identify one removable initializer and its reduction block."""

    initializer_block_nid: int
    reduction_block_nid: int
    tensor: str


@dataclass(frozen=True)
class _InitializerMatch:
    """Resolved leaves and reset loop for one elimination option."""

    option: EliminateIdentityInitializerOption
    initializer_leaf_nid: int
    reduction_leaf_nid: int
    reset_loop_nid: int
    output_operand: str


class EliminateIdentityInitializer(Transform[EliminateIdentityInitializerOption]):
    """Remove an identity fill when a one-step reduction overwrites first."""

    def analyze(self, ir: KernelIR) -> list[EliminateIdentityInitializerOption]:
        """Return every identity initializer with proven overwrite semantics."""
        options: list[EliminateIdentityInitializerOption] = []
        for initializer_block_nid in ir.tree.blocks():
            option = self._candidate_option(ir, initializer_block_nid)
            if option is not None and self._resolve(ir, option) is not None:
                options.append(option)
        return options

    def apply(self, ir: KernelIR, option: EliminateIdentityInitializerOption) -> KernelIR:
        """Recheck, copy, remove the initializer block, and rebuild metadata."""
        match = self._resolve(ir, option)
        if match is None:
            raise TransformLegalityError(f"illegal EliminateIdentityInitializer option: {option}")
        new_ir = copy.deepcopy(ir)
        copied_match = self._resolve(new_ir, option)
        if copied_match is None:
            raise AssertionError(f"EliminateIdentityInitializer option disappeared after deepcopy: {option}")
        self._remove_initializer(new_ir, copied_match)
        return new_ir

    def _candidate_option(self, ir: KernelIR, initializer_block_nid: int) -> EliminateIdentityInitializerOption | None:
        """Construct an option when the next tensor touch is a reduction."""
        option: EliminateIdentityInitializerOption | None = None
        initializer_leaf_nid = single_leaf(ir.tree, initializer_block_nid)
        if initializer_leaf_nid is not None:
            initializer = ir.tree.isa(initializer_leaf_nid)
            contract = initializer.op_cls.algebraic_contract(initializer.kwargs)
            if isinstance(contract, InitializerContract):
                region = initializer.operand_bindings.get(contract.output_operand)
                if region is not None:
                    touches = set(ir.dependency.touches_by_tensor.get(region.tensor, ()))
                    ordered = [nid for nid in ir.tree.preorder() if nid in touches]
                    if initializer_leaf_nid in ordered:
                        index = ordered.index(initializer_leaf_nid)
                        if index + 1 < len(ordered):
                            reduction_leaf_nid = ordered[index + 1]
                            option = EliminateIdentityInitializerOption(
                                initializer_block_nid=initializer_block_nid,
                                reduction_block_nid=owning_block(ir.tree, reduction_leaf_nid),
                                tensor=region.tensor,
                            )
        return option

    def _resolve(self, ir: KernelIR, option: EliminateIdentityInitializerOption) -> _InitializerMatch | None:
        """Resolve a fresh one-step reduction whose first write replaces identity."""
        result: _InitializerMatch | None = None
        initializer_block_nid = option.initializer_block_nid
        reduction_block_nid = option.reduction_block_nid
        if initializer_block_nid not in ir.tree.graph or reduction_block_nid not in ir.tree.graph:
            return result
        if not isinstance(ir.tree.data(initializer_block_nid), BlockNode) or not isinstance(
            ir.tree.data(reduction_block_nid), BlockNode
        ):
            return result
        initializer_leaf_nid = single_leaf(ir.tree, initializer_block_nid)
        reduction_leaf_nid = single_leaf(ir.tree, reduction_block_nid)
        if initializer_leaf_nid is None or reduction_leaf_nid is None:
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
            or output_operand not in reduction.op_cls.rmw_operands(reduction.kwargs)
            or not reduction.op_cls.first_write_overwrites(output_operand, reduction.kwargs)
            or self._has_reduction_loop(ir, reduction_block_nid, reduction_leaf_nid, reduction_axis)
        ):
            return result
        reset_loop_nid = ir.tree.parent(initializer_block_nid)
        allocation_block_nid = ir.tree.parent(reset_loop_nid) if reset_loop_nid is not None else None
        if (
            reset_loop_nid is None
            or not isinstance(ir.tree.data(reset_loop_nid), ForNode)
            or reset_loop_nid not in ir.tree.ancestors(reduction_leaf_nid)
            or self._has_local_loop(ir, initializer_block_nid)
            or allocation_block_nid is None
            or not isinstance(ir.tree.data(allocation_block_nid), BlockNode)
            or self._buffer_owner(ir, option.tensor) != allocation_block_nid
            or not self._initializer_precedes_reduction(ir, reset_loop_nid, initializer_block_nid, reduction_leaf_nid)
            or not self._all_touches_are_scoped(
                ir, option.tensor, reset_loop_nid, initializer_leaf_nid, reduction_leaf_nid
            )
        ):
            return result
        result = _InitializerMatch(
            option=option,
            initializer_leaf_nid=initializer_leaf_nid,
            reduction_leaf_nid=reduction_leaf_nid,
            reset_loop_nid=reset_loop_nid,
            output_operand=output_operand,
        )
        return result

    def _reduction_fields(self, contract: object) -> tuple[str, str, float] | None:
        """Return output, reduction axis, and identity for supported contracts."""
        result: tuple[str, str, float] | None = None
        if isinstance(contract, (BilinearReductionContract, ReductionContract)):
            result = (contract.output_operand, contract.reduction_axis, contract.combinator.identity)
        return result

    def _has_reduction_loop(self, ir: KernelIR, block_nid: int, leaf_nid: int, reduction_axis: str) -> bool:
        """Return whether the reduction axis executes more than one ISA call."""
        block = ir.tree.block(block_nid)
        concrete_axis = block.axis_map.get(reduction_axis)
        binding_vars: set[str] = set()
        if concrete_axis is not None:
            values = [
                value for iter_var, value in zip(block.iter_vars, block.iter_values) if iter_var.axis == concrete_axis
            ]
            if len(values) == 1:
                binding_vars = {name for name in to_affine(values[0]) if name is not None}
        return any(
            isinstance((node := ir.tree.data(nid)), ForNode)
            and node.loop_var in binding_vars
            and block_nid in ir.tree.ancestors(nid)
            for nid in ir.tree.ancestors(leaf_nid)
        )

    def _has_local_loop(self, ir: KernelIR, block_nid: int) -> bool:
        """Return whether an initializer block contains a loop."""
        return any(isinstance(ir.tree.data(nid), ForNode) for nid in ir.tree.descendants(block_nid))

    def _buffer_owner(self, ir: KernelIR, tensor: str) -> int | None:
        """Return the unique block declaring ``tensor``."""
        owners = [
            block_nid
            for block_nid in ir.tree.blocks()
            if any(buffer.name == tensor for buffer in ir.tree.block(block_nid).alloc_buffers)
        ]
        return owners[0] if len(owners) == 1 else None

    def _initializer_precedes_reduction(
        self, ir: KernelIR, loop_nid: int, initializer_block_nid: int, reduction_leaf_nid: int
    ) -> bool:
        """Return whether the initializer is before the reduction path child."""
        path_child = reduction_leaf_nid
        for ancestor in reversed(ir.tree.ancestors(reduction_leaf_nid)):
            if ancestor == loop_nid:
                break
            path_child = ancestor
        children = ir.tree.children(loop_nid)
        return (
            initializer_block_nid in children
            and path_child in children
            and children.index(initializer_block_nid) < children.index(path_child)
        )

    def _all_touches_are_scoped(
        self, ir: KernelIR, tensor: str, loop_nid: int, initializer_leaf_nid: int, reduction_leaf_nid: int
    ) -> bool:
        """Return whether the reduction becomes the first touch of each fresh tile."""
        touches = set(ir.dependency.touches_by_tensor.get(tensor, ()))
        preorder = [nid for nid in ir.tree.preorder() if nid in touches]
        scoped = all(loop_nid in ir.tree.ancestors(nid) for nid in touches)
        return (
            scoped and len(preorder) >= 2 and preorder[0] == initializer_leaf_nid and preorder[1] == reduction_leaf_nid
        )

    def _remove_initializer(self, ir: KernelIR, match: _InitializerMatch) -> None:
        """Remove the fill and make the single reduction write explicit."""
        block_nid = match.option.initializer_block_nid
        parent = ir.tree.parent(block_nid)
        if parent != match.reset_loop_nid:
            raise AssertionError(f"initializer block {block_nid} moved out of reset loop")
        reduction = ir.tree.isa(match.reduction_leaf_nid)
        output_region = reduction.operand_bindings[match.output_operand]
        kwargs = reduction.op_cls.with_first_write_overwrite(match.output_operand, reduction.kwargs)
        ir.tree.graph.nodes[match.reduction_leaf_nid]["data"] = replace(reduction, kwargs=kwargs)
        reduction_block_nid = match.option.reduction_block_nid
        reduction_block = ir.tree.block(reduction_block_nid)
        reads = tuple(region for region in reduction_block.reads if region != output_region)
        if len(reads) + 1 != len(reduction_block.reads):
            raise AssertionError(f"reduction block {reduction_block_nid} does not read its destination exactly once")
        ir.tree.graph.nodes[reduction_block_nid]["data"] = replace(reduction_block, reads=reads)
        removed = {block_nid, *ir.tree.descendants(block_nid)}
        _replace_in_parent_children(ir.tree, match.reset_loop_nid, [block_nid], [])
        ir.tree.graph.remove_nodes_from(removed)
        finalize_rewrite(ir)


__all__ = ["EliminateIdentityInitializer", "EliminateIdentityInitializerOption"]
