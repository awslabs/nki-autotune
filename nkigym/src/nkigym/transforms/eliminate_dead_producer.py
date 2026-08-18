"""Remove one pure producer whose result has no readers."""

from __future__ import annotations

from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.tree import BlockNode, ISANode
from nkigym.ops.base import CopyContract, PointwiseContract
from nkigym.search.program_sharding import PROGRAM_SHARDS_ANNOTATION, configured_program_shards
from nkigym.search.state_facts import operation_facts
from nkigym.transforms.base import (
    Transform,
    TransformLegalityError,
    TransformOption,
    copy_for_rewrite,
    intersects_software_pipeline,
    software_pipeline_overlap_nodes,
)
from nkigym.transforms.helper.canonical_rewrite import finalize_rewrite, single_leaf
from nkigym.transforms.helper.tree_ops import _replace_in_parent_children


@dataclass(frozen=True)
class EliminateDeadProducerOption(TransformOption):
    """Identify one pure producer block whose output is unused."""

    producer_block_nid: int


@dataclass(frozen=True)
class _DeadProducerMatch:
    """Resolved dead producer leaf and output tensor."""

    block_nid: int
    leaf_nid: int
    output_tensor: str


class EliminateDeadProducer(Transform[EliminateDeadProducerOption]):
    """Delete one unused pure producer."""

    def analyze(self, ir: KernelIR) -> list[EliminateDeadProducerOption]:
        """Return every isolated pure block with an unread private output."""
        facts = operation_facts(ir)
        if not facts.has_copy and not facts.pointwise_operators:
            return []
        block_nids = tuple(ir.tree.blocks())
        options = [EliminateDeadProducerOption(producer_block_nid=block_nid) for block_nid in block_nids]
        overlap_nodes = software_pipeline_overlap_nodes(ir)
        owners = self._buffer_owners(ir, block_nids)
        return [option for option in options if self._resolve(ir, option, overlap_nodes, owners) is not None]

    def apply(self, ir: KernelIR, option: EliminateDeadProducerOption) -> KernelIR:
        """Recheck, copy, and remove one dead producer block."""
        match = self._resolve(ir, option)
        if match is None:
            raise TransformLegalityError(f"illegal EliminateDeadProducer option: {option}")
        result = copy_for_rewrite(ir)
        copied = self._resolve(result, option)
        if copied is None:
            raise AssertionError(f"EliminateDeadProducer option disappeared after deepcopy: {option}")
        parent = result.tree.parent(copied.block_nid)
        if parent is None:
            raise AssertionError(f"producer block {copied.block_nid} has no parent")
        removed = {copied.block_nid, *result.tree.descendants(copied.block_nid)}
        shards = configured_program_shards(result)
        _replace_in_parent_children(result.tree, parent, [copied.block_nid], [])
        result.tree.graph.remove_nodes_from(removed)
        if removed.intersection(shards):
            root = result.tree.block(result.tree.root)
            annotations = dict(root.annotations)
            annotations[PROGRAM_SHARDS_ANNOTATION] = {
                loop_nid: programs for loop_nid, programs in shards.items() if loop_nid not in removed
            }
            result.tree.graph.nodes[result.tree.root]["data"] = replace(root, annotations=annotations)
        finalize_rewrite(result)
        return result

    def _resolve(
        self,
        ir: KernelIR,
        option: EliminateDeadProducerOption,
        overlap_nodes: frozenset[int] | None = None,
        owners: dict[str, int | None] | None = None,
    ) -> _DeadProducerMatch | None:
        """Resolve an isolated pure producer with one private unread output."""
        result: _DeadProducerMatch | None = None
        block_nid = option.producer_block_nid
        if block_nid not in ir.tree.graph or not isinstance(ir.tree.data(block_nid), BlockNode):
            return result
        if intersects_software_pipeline(ir, (block_nid,), overlap_nodes):
            return result
        leaf_nid = single_leaf(ir.tree, block_nid)
        leaves = [nid for nid in ir.tree.descendants(block_nid) if isinstance(ir.tree.data(nid), ISANode)]
        if leaf_nid is None or leaves != [leaf_nid]:
            return result
        leaf = ir.tree.isa(leaf_nid)
        contract = leaf.op_cls.algebraic_contract(leaf.kwargs)
        if not isinstance(contract, (CopyContract, PointwiseContract)):
            return result
        output = leaf.operand_bindings.get(contract.output_operand)
        if output is None or output.tensor in ir.param_buffers or output.tensor in ir.return_names:
            return result
        buffer_owners = self._buffer_owners(ir, tuple(ir.tree.blocks())) if owners is None else owners
        if buffer_owners.get(output.tensor) is None:
            return result
        touches = ir.dependency.touches_by_tensor.get(output.tensor, ())
        readers = {nid for nid in touches if output.tensor in ir.dependency.info(nid).reads}
        writers = {nid for nid in touches if output.tensor in ir.dependency.info(nid).writes}
        if not readers and writers == {leaf_nid}:
            result = _DeadProducerMatch(block_nid, leaf_nid, output.tensor)
        return result

    @staticmethod
    def _buffer_owners(ir: KernelIR, block_nids: tuple[int, ...]) -> dict[str, int | None]:
        """Return each uniquely declaring block, or ``None`` for duplicates."""
        owners: dict[str, int | None] = {}
        for block_nid in block_nids:
            for buffer in ir.tree.block(block_nid).alloc_buffers:
                owners[buffer.name] = block_nid if buffer.name not in owners else None
        return owners


__all__ = ["EliminateDeadProducer", "EliminateDeadProducerOption"]
