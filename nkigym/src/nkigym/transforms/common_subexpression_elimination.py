"""Eliminate repeated pure pointwise expressions in one execution scope."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode
from nkigym.ops.base import PointwiseContract
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption
from nkigym.transforms.helper.canonical_rewrite import finalize_rewrite, remove_buffers, single_leaf
from nkigym.transforms.helper.tree_ops import _replace_in_parent_children
from nkigym.transforms.helper.value_graph import contract_input_operands


@dataclass(frozen=True)
class CommonSubexpressionEliminationOption(TransformOption):
    """Identify one canonical and one redundant pointwise block."""

    canonical_block_nid: int
    redundant_block_nid: int


@dataclass(frozen=True)
class _CommonSubexpressionMatch:
    """Resolved duplicate outputs and owning leaves."""

    option: CommonSubexpressionEliminationOption
    canonical_leaf_nid: int
    redundant_leaf_nid: int
    canonical_output: BufferRegion
    redundant_output: BufferRegion


class CommonSubexpressionElimination(Transform[CommonSubexpressionEliminationOption]):
    """Share identical contract-declared pointwise expressions."""

    def analyze(self, ir: KernelIR) -> list[CommonSubexpressionEliminationOption]:
        """Return repeated pure pointwise blocks within each direct child list."""
        options: list[CommonSubexpressionEliminationOption] = []
        for parent_nid in ir.tree.preorder():
            blocks = [nid for nid in ir.tree.children(parent_nid) if isinstance(ir.tree.data(nid), BlockNode)]
            for index, canonical_nid in enumerate(blocks):
                for redundant_nid in blocks[index + 1 :]:
                    option = CommonSubexpressionEliminationOption(
                        canonical_block_nid=canonical_nid, redundant_block_nid=redundant_nid
                    )
                    if self._resolve(ir, option) is not None:
                        options.append(option)
        return options

    def apply(self, ir: KernelIR, option: CommonSubexpressionEliminationOption) -> KernelIR:
        """Recheck, copy, redirect readers, and remove the redundant expression."""
        match = self._resolve(ir, option)
        if match is None:
            raise TransformLegalityError(f"illegal CommonSubexpressionElimination option: {option}")
        new_ir = copy.deepcopy(ir)
        copied_match = self._resolve(new_ir, option)
        if copied_match is None:
            raise AssertionError(f"CommonSubexpressionElimination option disappeared after deepcopy: {option}")
        self._rewrite(new_ir, copied_match)
        return new_ir

    def _resolve(self, ir: KernelIR, option: CommonSubexpressionEliminationOption) -> _CommonSubexpressionMatch | None:
        """Resolve identical pointwise calls with compatible execution and storage."""
        result: _CommonSubexpressionMatch | None = None
        canonical_nid = option.canonical_block_nid
        redundant_nid = option.redundant_block_nid
        if canonical_nid not in ir.tree.graph or redundant_nid not in ir.tree.graph:
            return result
        if not isinstance(ir.tree.data(canonical_nid), BlockNode) or not isinstance(
            ir.tree.data(redundant_nid), BlockNode
        ):
            return result
        parent = ir.tree.parent(canonical_nid)
        siblings = ir.tree.children(parent) if parent is not None else []
        if (
            parent is None
            or ir.tree.parent(redundant_nid) != parent
            or canonical_nid not in siblings
            or redundant_nid not in siblings
            or siblings.index(canonical_nid) >= siblings.index(redundant_nid)
        ):
            return result
        canonical_leaf_nid = single_leaf(ir.tree, canonical_nid)
        redundant_leaf_nid = single_leaf(ir.tree, redundant_nid)
        if canonical_leaf_nid is None or redundant_leaf_nid is None:
            return result
        canonical_leaf = ir.tree.isa(canonical_leaf_nid)
        redundant_leaf = ir.tree.isa(redundant_leaf_nid)
        canonical_contract = canonical_leaf.op_cls.algebraic_contract(canonical_leaf.kwargs)
        redundant_contract = redundant_leaf.op_cls.algebraic_contract(redundant_leaf.kwargs)
        if (
            not isinstance(canonical_contract, PointwiseContract)
            or canonical_contract != redundant_contract
            or canonical_leaf.op_cls is not redundant_leaf.op_cls
            or canonical_leaf.kwargs != redundant_leaf.kwargs
            or canonical_leaf.access_patterns
            or redundant_leaf.access_patterns
            or not self._same_execution(ir, canonical_nid, redundant_nid)
        ):
            return result
        inputs = contract_input_operands(canonical_contract)
        if any(
            canonical_leaf.operand_bindings.get(slot) != redundant_leaf.operand_bindings.get(slot) for slot in inputs
        ):
            return result
        canonical_output = canonical_leaf.operand_bindings.get(canonical_contract.output_operand)
        redundant_output = redundant_leaf.operand_bindings.get(canonical_contract.output_operand)
        if canonical_output is None or redundant_output is None or canonical_output.tensor == redundant_output.tensor:
            return result
        canonical_buffer = ir.buffer(canonical_output.tensor)
        redundant_buffer = ir.buffer(redundant_output.tensor)
        if (
            canonical_output.ranges != redundant_output.ranges
            or canonical_buffer.shape != redundant_buffer.shape
            or canonical_buffer.location != redundant_buffer.location
            or canonical_buffer.physical_dtype() != redundant_buffer.physical_dtype()
            or canonical_buffer.versions != redundant_buffer.versions
            or canonical_buffer.list_len != redundant_buffer.list_len
            or redundant_output.tensor in ir.param_buffers
            or redundant_output.tensor in ir.return_names
            or redundant_buffer.location == "shared_hbm"
        ):
            return result
        if not self._definitions_and_uses_are_compatible(
            ir, canonical_output.tensor, redundant_output.tensor, canonical_leaf_nid, redundant_leaf_nid
        ):
            return result
        input_tensors = {
            region.tensor for slot in inputs if (region := canonical_leaf.operand_bindings.get(slot)) is not None
        }
        if self._intervening_write(ir, siblings, canonical_nid, redundant_nid, input_tensors):
            return result
        result = _CommonSubexpressionMatch(
            option=option,
            canonical_leaf_nid=canonical_leaf_nid,
            redundant_leaf_nid=redundant_leaf_nid,
            canonical_output=canonical_output,
            redundant_output=redundant_output,
        )
        return result

    def _same_execution(self, ir: KernelIR, canonical_nid: int, redundant_nid: int) -> bool:
        """Return whether two blocks bind identical axes and local loops."""
        canonical = ir.tree.block(canonical_nid)
        redundant = ir.tree.block(redundant_nid)
        canonical_loops = tuple(
            (node.loop_var, node.extent)
            for nid in ir.tree.preorder(canonical_nid)
            if isinstance((node := ir.tree.data(nid)), ForNode)
        )
        redundant_loops = tuple(
            (node.loop_var, node.extent)
            for nid in ir.tree.preorder(redundant_nid)
            if isinstance((node := ir.tree.data(nid)), ForNode)
        )
        return (
            canonical.iter_vars == redundant.iter_vars
            and canonical.iter_values == redundant.iter_values
            and canonical.axis_map == redundant.axis_map
            and canonical_loops == redundant_loops
        )

    def _definitions_and_uses_are_compatible(
        self,
        ir: KernelIR,
        canonical_tensor: str,
        redundant_tensor: str,
        canonical_leaf_nid: int,
        redundant_leaf_nid: int,
    ) -> bool:
        """Require unique definitions and readers ordered after each definition."""
        preorder = list(ir.tree.preorder())
        writers: dict[str, set[int]] = {canonical_tensor: set(), redundant_tensor: set()}
        readers: dict[str, set[int]] = {canonical_tensor: set(), redundant_tensor: set()}
        for nid in preorder:
            node = ir.tree.data(nid)
            if not isinstance(node, ISANode):
                continue
            rmw_operands = node.op_cls.rmw_operands(node.kwargs)
            for slot, region in node.operand_bindings.items():
                if region.tensor not in writers:
                    continue
                if slot in node.op_cls.INPUT_OPERANDS or slot in rmw_operands:
                    readers[region.tensor].add(nid)
                if slot not in node.op_cls.INPUT_OPERANDS:
                    writers[region.tensor].add(nid)
        return (
            writers[canonical_tensor] == {canonical_leaf_nid}
            and writers[redundant_tensor] == {redundant_leaf_nid}
            and bool(readers[redundant_tensor])
            and all(preorder.index(reader) > preorder.index(canonical_leaf_nid) for reader in readers[canonical_tensor])
            and all(preorder.index(reader) > preorder.index(redundant_leaf_nid) for reader in readers[redundant_tensor])
        )

    def _intervening_write(
        self, ir: KernelIR, siblings: list[int], canonical_nid: int, redundant_nid: int, input_tensors: set[str]
    ) -> bool:
        """Return whether an intervening sibling mutates any shared input."""
        start = siblings.index(canonical_nid) + 1
        stop = siblings.index(redundant_nid)
        for sibling in siblings[start:stop]:
            for nid in (sibling, *ir.tree.descendants(sibling)):
                node = ir.tree.data(nid)
                if not isinstance(node, ISANode):
                    continue
                for slot, region in node.operand_bindings.items():
                    if slot not in node.op_cls.INPUT_OPERANDS and region.tensor in input_tensors:
                        return True
        return False

    def _rewrite(self, ir: KernelIR, match: _CommonSubexpressionMatch) -> None:
        """Redirect all redundant readers and delete its producer and buffer."""
        redundant_descendants = {
            match.option.redundant_block_nid,
            *ir.tree.descendants(match.option.redundant_block_nid),
        }
        for nid in ir.tree.preorder():
            if nid in redundant_descendants:
                continue
            node = ir.tree.data(nid)
            if isinstance(node, ISANode):
                bindings = {
                    slot: (
                        replace(region, tensor=match.canonical_output.tensor)
                        if region.tensor == match.redundant_output.tensor
                        else region
                    )
                    for slot, region in node.operand_bindings.items()
                }
                if bindings != node.operand_bindings:
                    ir.tree.graph.nodes[nid]["data"] = replace(node, operand_bindings=bindings)
            elif isinstance(node, BlockNode):
                reads = tuple(
                    (
                        replace(region, tensor=match.canonical_output.tensor)
                        if region.tensor == match.redundant_output.tensor
                        else region
                    )
                    for region in node.reads
                )
                if reads != node.reads:
                    ir.tree.graph.nodes[nid]["data"] = replace(node, reads=reads)

        parent = ir.tree.parent(match.option.redundant_block_nid)
        if parent is None:
            raise AssertionError(f"redundant block {match.option.redundant_block_nid} has no parent")
        _replace_in_parent_children(ir.tree, parent, [match.option.redundant_block_nid], [])
        ir.tree.graph.remove_nodes_from(redundant_descendants)
        remove_buffers(ir, {match.redundant_output.tensor})
        finalize_rewrite(ir)


__all__ = ["CommonSubexpressionElimination", "CommonSubexpressionEliminationOption"]
