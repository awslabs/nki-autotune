"""Eliminate repeated pure pointwise expressions in one execution scope."""

from __future__ import annotations

from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Expr, Var, substitute
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode
from nkigym.ops.base import PointwiseContract
from nkigym.search.state_facts import operation_facts
from nkigym.transforms.base import (
    Transform,
    TransformLegalityError,
    TransformOption,
    copy_for_rewrite,
    intersects_software_pipeline,
    software_pipeline_overlap_nodes,
)
from nkigym.transforms.helper.canonical_rewrite import block_chain, finalize_rewrite, single_leaf
from nkigym.transforms.helper.value_graph import contract_input_operands


@dataclass(frozen=True)
class CommonSubexpressionEliminationOption(TransformOption):
    """Redirect one reader of a redundant pointwise block."""

    canonical_block_nid: int
    redundant_block_nid: int
    consumer_nid: int
    consumer_operand: str


@dataclass(frozen=True)
class _CommonSubexpressionMatch:
    """Resolved duplicate outputs and one concrete reader."""

    option: CommonSubexpressionEliminationOption
    canonical_leaf_nid: int
    redundant_leaf_nid: int
    consumer_nid: int
    canonical_output: BufferRegion
    redundant_output: BufferRegion


class CommonSubexpressionElimination(Transform[CommonSubexpressionEliminationOption]):
    """Share identical contract-declared pointwise expressions."""

    def analyze(self, ir: KernelIR) -> list[CommonSubexpressionEliminationOption]:
        """Return repeated pure pointwise blocks within each direct child list."""
        if not operation_facts(ir).pointwise_operators:
            return []
        options: list[CommonSubexpressionEliminationOption] = []
        overlap_nodes = software_pipeline_overlap_nodes(ir)
        for parent_nid in ir.tree.preorder():
            blocks = [nid for nid in ir.tree.children(parent_nid) if isinstance(ir.tree.data(nid), BlockNode)]
            groups: dict[str, list[int]] = {}
            for block_nid in blocks:
                key = self._candidate_key(ir, block_nid)
                if key is not None:
                    groups.setdefault(key, []).append(block_nid)
            for pointwise_blocks in groups.values():
                for index, canonical_nid in enumerate(pointwise_blocks):
                    for redundant_nid in pointwise_blocks[index + 1 :]:
                        for consumer_nid, consumer_operand in self._reader_operands(ir, redundant_nid):
                            option = CommonSubexpressionEliminationOption(
                                canonical_block_nid=canonical_nid,
                                redundant_block_nid=redundant_nid,
                                consumer_nid=consumer_nid,
                                consumer_operand=consumer_operand,
                            )
                            if self._resolve(ir, option, overlap_nodes) is not None:
                                options.append(option)
        return options

    def _candidate_key(self, ir: KernelIR, block_nid: int) -> str | None:
        """Return a cheap exact-match key for one pure pointwise block."""
        leaf_nid = single_leaf(ir.tree, block_nid)
        key: str | None = None
        if leaf_nid is not None:
            leaf = ir.tree.isa(leaf_nid)
            contract = leaf.op_cls.algebraic_contract(leaf.kwargs)
            if isinstance(contract, PointwiseContract):
                block = ir.tree.block(block_nid)
                loop_extents = tuple(
                    node.extent
                    for nid in ir.tree.preorder(block_nid)
                    if isinstance((node := ir.tree.data(nid)), ForNode)
                )
                key = repr(
                    (
                        leaf.op_cls,
                        tuple(sorted((name, repr(value)) for name, value in leaf.kwargs.items())),
                        contract,
                        block.iter_vars,
                        tuple(sorted(block.axis_map.items())),
                        loop_extents,
                    )
                )
        return key

    def apply(self, ir: KernelIR, option: CommonSubexpressionEliminationOption) -> KernelIR:
        """Recheck, copy, and redirect one reader to the canonical expression."""
        match = self._resolve(ir, option)
        if match is None:
            raise TransformLegalityError(f"illegal CommonSubexpressionElimination option: {option}")
        new_ir = copy_for_rewrite(ir)
        copied_match = self._resolve(new_ir, option)
        if copied_match is None:
            raise AssertionError(f"CommonSubexpressionElimination option disappeared after deepcopy: {option}")
        self._rewrite(new_ir, copied_match)
        return new_ir

    def _resolve(
        self, ir: KernelIR, option: CommonSubexpressionEliminationOption, overlap_nodes: frozenset[int] | None = None
    ) -> _CommonSubexpressionMatch | None:
        """Resolve identical pointwise calls with compatible execution and storage."""
        result: _CommonSubexpressionMatch | None = None
        canonical_nid = option.canonical_block_nid
        redundant_nid = option.redundant_block_nid
        if any(nid not in ir.tree.graph for nid in (canonical_nid, redundant_nid, option.consumer_nid)):
            return result
        if not isinstance(ir.tree.data(canonical_nid), BlockNode) or not isinstance(
            ir.tree.data(redundant_nid), BlockNode
        ):
            return result
        if intersects_software_pipeline(ir, (canonical_nid, redundant_nid, option.consumer_nid), overlap_nodes):
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
        consumer = ir.tree.data(option.consumer_nid)
        if canonical_leaf_nid is None or redundant_leaf_nid is None or not isinstance(consumer, ISANode):
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
            not self._same_operand(ir, canonical_nid, redundant_nid, canonical_leaf, redundant_leaf, slot)
            for slot in inputs
        ):
            return result
        canonical_output = canonical_leaf.operand_bindings.get(canonical_contract.output_operand)
        redundant_output = redundant_leaf.operand_bindings.get(canonical_contract.output_operand)
        if canonical_output is None or redundant_output is None or canonical_output.tensor == redundant_output.tensor:
            return result
        consumed = consumer.operand_bindings.get(option.consumer_operand)
        if option.consumer_operand not in consumer.op_cls.INPUT_OPERANDS or consumed is None:
            return result
        if consumed.tensor != redundant_output.tensor:
            return result
        canonical_buffer = ir.buffer(canonical_output.tensor)
        redundant_buffer = ir.buffer(redundant_output.tensor)
        declarations = [
            nid
            for nid in ir.tree.blocks()
            if any(buffer.name == canonical_output.tensor for buffer in ir.tree.block(nid).alloc_buffers)
        ]
        if (
            not self._same_ranges(ir, canonical_nid, redundant_nid, canonical_output, redundant_output)
            or len(declarations) != 1
            or declarations[0] not in ir.tree.ancestors(option.consumer_nid)
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
            consumer_nid=option.consumer_nid,
            canonical_output=canonical_output,
            redundant_output=redundant_output,
        )
        return result

    def _reader_operands(self, ir: KernelIR, redundant_nid: int) -> tuple[tuple[int, str], ...]:
        """Return ISA reader operands of one candidate redundant output."""
        leaf_nid = single_leaf(ir.tree, redundant_nid)
        if leaf_nid is None:
            return ()
        leaf = ir.tree.isa(leaf_nid)
        contract = leaf.op_cls.algebraic_contract(leaf.kwargs)
        if not isinstance(contract, PointwiseContract):
            return ()
        output = leaf.operand_bindings.get(contract.output_operand)
        if output is None:
            return ()
        return tuple(
            (nid, slot)
            for nid in ir.dependency.touches_by_tensor.get(output.tensor, ())
            if nid != leaf_nid
            for slot, region in ir.tree.isa(nid).operand_bindings.items()
            if slot in ir.tree.isa(nid).op_cls.INPUT_OPERANDS and region.tensor == output.tensor
        )

    def _same_execution(self, ir: KernelIR, canonical_nid: int, redundant_nid: int) -> bool:
        """Return whether two blocks bind identical axes and local loops."""
        canonical = ir.tree.block(canonical_nid)
        redundant = ir.tree.block(redundant_nid)
        canonical_loops, canonical_substitutions = self._local_loop_form(ir, canonical_nid)
        redundant_loops, redundant_substitutions = self._local_loop_form(ir, redundant_nid)
        return (
            canonical.iter_vars == redundant.iter_vars
            and canonical.axis_map == redundant.axis_map
            and canonical_loops == redundant_loops
            and tuple(substitute(value, canonical_substitutions) for value in canonical.iter_values)
            == tuple(substitute(value, redundant_substitutions) for value in redundant.iter_values)
        )

    def _local_loop_form(self, ir: KernelIR, block_nid: int) -> tuple[tuple[int, ...], dict[str, Expr]]:
        """Return local loop extents and an alpha-normalizing substitution."""
        chain = block_chain(ir.tree, block_nid)
        loops = () if chain is None else tuple(node for node in chain if isinstance(node, ForNode))
        substitutions: dict[str, Expr] = {
            loop.loop_var: Var(name=f"_cse_loop_{index}") for index, loop in enumerate(loops)
        }
        return tuple(loop.extent for loop in loops), substitutions

    def _same_operand(
        self,
        ir: KernelIR,
        canonical_nid: int,
        redundant_nid: int,
        canonical_leaf: ISANode,
        redundant_leaf: ISANode,
        slot: str,
    ) -> bool:
        """Return whether one input slot is alpha-equivalent in both blocks."""
        canonical = canonical_leaf.operand_bindings.get(slot)
        redundant = redundant_leaf.operand_bindings.get(slot)
        return (
            canonical is not None
            and redundant is not None
            and canonical.tensor == redundant.tensor
            and self._same_ranges(ir, canonical_nid, redundant_nid, canonical, redundant)
        )

    def _same_ranges(
        self, ir: KernelIR, canonical_nid: int, redundant_nid: int, canonical: BufferRegion, redundant: BufferRegion
    ) -> bool:
        """Return whether two region ranges are equal after local alpha-renaming."""
        _canonical_loops, canonical_substitutions = self._local_loop_form(ir, canonical_nid)
        _redundant_loops, redundant_substitutions = self._local_loop_form(ir, redundant_nid)
        canonical_ranges = tuple(
            (substitute(lower, canonical_substitutions), substitute(width, canonical_substitutions))
            for lower, width in canonical.ranges
        )
        redundant_ranges = tuple(
            (substitute(lower, redundant_substitutions), substitute(width, redundant_substitutions))
            for lower, width in redundant.ranges
        )
        return canonical_ranges == redundant_ranges

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
        """Redirect one redundant reader while retaining the producer."""
        consumer = ir.tree.isa(match.consumer_nid)
        operand = match.option.consumer_operand
        old_region = consumer.operand_bindings[operand]
        bindings = {
            slot: (replace(region, tensor=match.canonical_output.tensor) if slot == operand else region)
            for slot, region in consumer.operand_bindings.items()
        }
        ir.tree.graph.nodes[match.consumer_nid]["data"] = replace(consumer, operand_bindings=bindings)
        block_nid = next(
            nid for nid in reversed(ir.tree.ancestors(match.consumer_nid)) if isinstance(ir.tree.data(nid), BlockNode)
        )
        block = ir.tree.block(block_nid)
        replaced = False
        reads: list[BufferRegion] = []
        for region in block.reads:
            if not replaced and region == old_region:
                reads.append(replace(region, tensor=match.canonical_output.tensor))
                replaced = True
            else:
                reads.append(region)
        if not replaced:
            raise AssertionError(f"CSE consumer block {block_nid} does not read operand {operand!r}")
        ir.tree.graph.nodes[block_nid]["data"] = replace(block, reads=tuple(reads))
        finalize_rewrite(ir)


__all__ = ["CommonSubexpressionElimination", "CommonSubexpressionEliminationOption"]
