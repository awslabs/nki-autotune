"""Propagate an on-chip copy source into a co-located consumer."""

from __future__ import annotations

from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ISANode
from nkigym.ops.base import CopyContract
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


@dataclass(frozen=True)
class CopyPropagationOption(TransformOption):
    """Identify one adjacent copy and consumer input."""

    copy_block_nid: int
    consumer_block_nid: int
    consumer_operand: str


@dataclass(frozen=True)
class _CopyPropagationMatch:
    """Resolved copy propagation with exact source and destination regions."""

    option: CopyPropagationOption
    copy_leaf_nid: int
    consumer_leaf_nid: int
    source: BufferRegion
    copied: BufferRegion


class CopyPropagation(Transform[CopyPropagationOption]):
    """Substitute one value-preserving copy source into its consumer."""

    def analyze(self, ir: KernelIR) -> list[CopyPropagationOption]:
        """Return every adjacent copy-consumer pair accepted by storage contracts."""
        if not operation_facts(ir).has_copy:
            return []
        options: list[CopyPropagationOption] = []
        buffers = ir.all_buffers()
        overlap_nodes = software_pipeline_overlap_nodes(ir)
        positions = {nid: index for index, nid in enumerate(ir.tree.preorder())}
        for parent_nid in ir.tree.preorder():
            children = ir.tree.children(parent_nid)
            for copy_nid, consumer_nid in zip(children, children[1:]):
                if not isinstance(ir.tree.data(copy_nid), BlockNode) or not isinstance(
                    ir.tree.data(consumer_nid), BlockNode
                ):
                    continue
                consumer_leaf_nid = single_leaf(ir.tree, consumer_nid)
                if consumer_leaf_nid is None:
                    continue
                consumer = ir.tree.isa(consumer_leaf_nid)
                for operand in consumer.op_cls.INPUT_OPERANDS:
                    option = CopyPropagationOption(
                        copy_block_nid=copy_nid, consumer_block_nid=consumer_nid, consumer_operand=operand
                    )
                    if (
                        self._resolve(ir, option, buffers, overlap_nodes, adjacent=True, positions=positions)
                        is not None
                    ):
                        options.append(option)
        return options

    def apply(self, ir: KernelIR, option: CopyPropagationOption) -> KernelIR:
        """Recheck, copy, and propagate the source region into one consumer."""
        match = self._resolve(ir, option, ir.all_buffers())
        if match is None:
            raise TransformLegalityError(f"illegal CopyPropagation option: {option}")
        new_ir = copy_for_rewrite(ir)
        copied_match = self._resolve(new_ir, option, new_ir.all_buffers())
        if copied_match is None:
            raise AssertionError(f"CopyPropagation option disappeared after deepcopy: {option}")
        self._rewrite(new_ir, copied_match)
        return new_ir

    def _resolve(
        self,
        ir: KernelIR,
        option: CopyPropagationOption,
        buffers: dict[str, Buffer],
        overlap_nodes: frozenset[int] | None = None,
        adjacent: bool | None = None,
        positions: dict[int, int] | None = None,
    ) -> _CopyPropagationMatch | None:
        """Resolve an option when copy semantics, storage, and use-def all agree."""
        result: _CopyPropagationMatch | None = None
        copy_nid = option.copy_block_nid
        consumer_nid = option.consumer_block_nid
        if copy_nid not in ir.tree.graph or consumer_nid not in ir.tree.graph:
            return result
        if not isinstance(ir.tree.data(copy_nid), BlockNode) or not isinstance(ir.tree.data(consumer_nid), BlockNode):
            return result
        if intersects_software_pipeline(ir, (copy_nid, consumer_nid), overlap_nodes):
            return result
        if adjacent is None:
            parent = ir.tree.parent(copy_nid)
            siblings = ir.tree.children(parent) if parent is not None else []
            adjacent = (
                parent is not None
                and ir.tree.parent(consumer_nid) == parent
                and copy_nid in siblings
                and siblings.index(consumer_nid) == siblings.index(copy_nid) + 1
            )
        if not adjacent:
            return result
        copy_leaf_nid = single_leaf(ir.tree, copy_nid)
        consumer_leaf_nid = single_leaf(ir.tree, consumer_nid)
        if copy_leaf_nid is None or consumer_leaf_nid is None:
            return result
        copy_leaf = ir.tree.isa(copy_leaf_nid)
        consumer_leaf = ir.tree.isa(consumer_leaf_nid)
        if copy_leaf.access_patterns or consumer_leaf.access_patterns:
            return result
        contract = copy_leaf.op_cls.algebraic_contract(copy_leaf.kwargs)
        if not isinstance(contract, CopyContract):
            return result
        source = copy_leaf.operand_bindings.get(contract.input_operand)
        copied = copy_leaf.operand_bindings.get(contract.output_operand)
        consumed = consumer_leaf.operand_bindings.get(option.consumer_operand)
        if source is None or copied is None or consumed != copied:
            return result
        if copied.tensor in ir.param_buffers or copied.tensor in ir.return_names:
            return result
        copied_buffer = buffers[copied.tensor]
        source_buffer = buffers[source.tensor]
        locations = {
            operand: buffers[region.tensor].location
            for operand, region in consumer_leaf.operand_bindings.items()
            if operand in consumer_leaf.op_cls.INPUT_OPERANDS
        }
        locations[option.consumer_operand] = source_buffer.location
        dtypes = {
            operand: buffers[region.tensor].physical_dtype()
            for operand, region in consumer_leaf.operand_bindings.items()
            if operand in consumer_leaf.op_cls.INPUT_OPERANDS
        }
        dtypes[option.consumer_operand] = source_buffer.physical_dtype()
        required_dtype = consumer_leaf.op_cls.REQUIRED_INPUT_STORAGE_DTYPES.get(option.consumer_operand)
        if (
            copied_buffer.location == "shared_hbm"
            or source_buffer.location not in {"sbuf", "psum"}
            or (source_buffer.dtype, source_buffer.physical_dtype())
            != (copied_buffer.dtype, copied_buffer.physical_dtype())
            or not consumer_leaf.op_cls.accepts_input_locations(locations)
            or not consumer_leaf.op_cls.accepts_input_storage_dtypes(dtypes)
            or (required_dtype is not None and source_buffer.physical_dtype() != required_dtype)
            or source.tensor == copied.tensor
            or len(source.ranges) != len(copied.ranges)
        ):
            return result
        if tuple(width for _lower, width in source.ranges) != tuple(width for _lower, width in copied.ranges):
            return result
        copy_block = ir.tree.block(copy_nid)
        if any(buffer.name != copied.tensor for buffer in copy_block.alloc_buffers):
            return result
        if not self._has_single_use_and_definition(ir, copied.tensor, copy_leaf_nid, consumer_leaf_nid):
            return result
        if not self._source_remains_stable(
            ir,
            source.tensor,
            copy_leaf_nid,
            consumer_leaf_nid,
            positions if positions is not None else {nid: index for index, nid in enumerate(ir.tree.preorder())},
        ):
            return result
        result = _CopyPropagationMatch(
            option=option,
            copy_leaf_nid=copy_leaf_nid,
            consumer_leaf_nid=consumer_leaf_nid,
            source=source,
            copied=copied,
        )
        return result

    def _has_single_use_and_definition(
        self, ir: KernelIR, tensor: str, copy_leaf_nid: int, consumer_leaf_nid: int
    ) -> bool:
        """Return whether the copied tensor has exactly one writer and one reader."""
        readers: list[int] = []
        writers: list[int] = []
        for nid in ir.dependency.touches_by_tensor.get(tensor, ()):
            node = ir.tree.data(nid)
            assert isinstance(node, ISANode)
            rmw_operands = node.op_cls.rmw_operands(node.kwargs)
            for slot, region in node.operand_bindings.items():
                if region.tensor != tensor:
                    continue
                if slot in node.op_cls.INPUT_OPERANDS or slot in rmw_operands:
                    readers.append(nid)
                if slot not in node.op_cls.INPUT_OPERANDS:
                    writers.append(nid)
        return readers == [consumer_leaf_nid] and writers == [copy_leaf_nid]

    def _source_remains_stable(
        self, ir: KernelIR, tensor: str, copy_leaf_nid: int, consumer_leaf_nid: int, positions: dict[int, int]
    ) -> bool:
        """Return whether no intervening instruction overwrites ``tensor``."""
        copy_position = positions[copy_leaf_nid]
        consumer_position = positions[consumer_leaf_nid]
        if copy_position >= consumer_position:
            return False
        for nid in ir.dependency.touches_by_tensor.get(tensor, ()):
            position = positions[nid]
            if copy_position < position <= consumer_position:
                node = ir.tree.isa(nid)
                if any(
                    region.tensor == tensor and operand not in node.op_cls.INPUT_OPERANDS
                    for operand, region in node.operand_bindings.items()
                ):
                    return False
        return True

    def _rewrite(self, ir: KernelIR, match: _CopyPropagationMatch) -> None:
        """Retarget the consumer and rebuild derived metadata."""
        consumer = ir.tree.isa(match.consumer_leaf_nid)
        bindings = dict(consumer.operand_bindings)
        bindings[match.option.consumer_operand] = match.source
        ir.tree.graph.nodes[match.consumer_leaf_nid]["data"] = replace(consumer, operand_bindings=bindings)

        consumer_block = ir.tree.block(match.option.consumer_block_nid)
        replaced = False
        reads: list[BufferRegion] = []
        for region in consumer_block.reads:
            if region == match.copied and not replaced:
                reads.append(match.source)
                replaced = True
            else:
                reads.append(region)
        if not replaced:
            raise AssertionError(
                f"consumer block {match.option.consumer_block_nid} does not read {match.copied.tensor!r}"
            )
        ir.tree.graph.nodes[match.option.consumer_block_nid]["data"] = replace(consumer_block, reads=tuple(reads))
        finalize_rewrite(ir)


__all__ = ["CopyPropagation", "CopyPropagationOption"]
