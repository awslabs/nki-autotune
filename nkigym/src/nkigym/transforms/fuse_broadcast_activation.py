"""Fuse a broadcast pointwise operation into an activation bias input."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.tree import BlockNode, BufferRegion, ISANode
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.base import PointwiseContract, ReductionContract
from nkigym.transforms._canonical_rewrite import finalize_rewrite, remove_buffers, single_leaf
from nkigym.transforms._tree_ops import _replace_in_parent_children
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class FuseBroadcastActivationOption(TransformOption):
    """Identify adjacent broadcast-pointwise and activation-reduction blocks."""

    pointwise_block_nid: int
    activation_block_nid: int


@dataclass(frozen=True)
class _BroadcastActivationMatch:
    """Resolved operands for one native activation-bias fusion."""

    option: FuseBroadcastActivationOption
    pointwise_leaf_nid: int
    activation_leaf_nid: int
    data: BufferRegion
    broadcast: BufferRegion
    intermediate: BufferRegion
    activation_input: BufferRegion


class FuseBroadcastActivation(Transform[FuseBroadcastActivationOption]):
    """Move an elementwise broadcast addition into activation bias."""

    def analyze(self, ir: KernelIR) -> list[FuseBroadcastActivationOption]:
        """Return adjacent pointwise-activation pairs with native bias support."""
        options: list[FuseBroadcastActivationOption] = []
        for parent_nid in ir.tree.preorder():
            children = ir.tree.children(parent_nid)
            for pointwise_nid, activation_nid in zip(children, children[1:]):
                option = FuseBroadcastActivationOption(
                    pointwise_block_nid=pointwise_nid, activation_block_nid=activation_nid
                )
                if self._resolve(ir, option) is not None:
                    options.append(option)
        return options

    def apply(self, ir: KernelIR, option: FuseBroadcastActivationOption) -> KernelIR:
        """Recheck, copy, rewrite the activation input, and remove the matrix temporary."""
        match = self._resolve(ir, option)
        if match is None:
            raise TransformLegalityError(f"illegal FuseBroadcastActivation option: {option}")
        new_ir = copy.deepcopy(ir)
        copied_match = self._resolve(new_ir, option)
        if copied_match is None:
            raise AssertionError(f"FuseBroadcastActivation option disappeared after deepcopy: {option}")
        self._rewrite(new_ir, copied_match)
        return new_ir

    def _resolve(self, ir: KernelIR, option: FuseBroadcastActivationOption) -> _BroadcastActivationMatch | None:
        """Resolve a contract-compatible pair with a unique intermediate use."""
        result: _BroadcastActivationMatch | None = None
        pointwise_nid = option.pointwise_block_nid
        activation_nid = option.activation_block_nid
        if pointwise_nid not in ir.tree.graph or activation_nid not in ir.tree.graph:
            return result
        if not isinstance(ir.tree.data(pointwise_nid), BlockNode) or not isinstance(
            ir.tree.data(activation_nid), BlockNode
        ):
            return result
        parent = ir.tree.parent(pointwise_nid)
        siblings = ir.tree.children(parent) if parent is not None else []
        if (
            parent is None
            or ir.tree.parent(activation_nid) != parent
            or pointwise_nid not in siblings
            or siblings.index(activation_nid) != siblings.index(pointwise_nid) + 1
        ):
            return result
        pointwise_leaf_nid = single_leaf(ir.tree, pointwise_nid)
        activation_leaf_nid = single_leaf(ir.tree, activation_nid)
        if pointwise_leaf_nid is None or activation_leaf_nid is None:
            return result
        pointwise_leaf = ir.tree.isa(pointwise_leaf_nid)
        activation_leaf = ir.tree.isa(activation_leaf_nid)
        if pointwise_leaf.access_patterns or activation_leaf.access_patterns:
            return result
        pointwise = pointwise_leaf.op_cls.algebraic_contract(pointwise_leaf.kwargs)
        activation = activation_leaf.op_cls.algebraic_contract(activation_leaf.kwargs)
        if not isinstance(pointwise, PointwiseContract) or not isinstance(activation, ReductionContract):
            return result
        if activation_leaf.op_cls is not NKIActivationReduce or activation.bias_operand != "bias":
            return result
        if (
            pointwise.operator != "add"
            or pointwise.reverse
            or len(pointwise.input_operands) != 2
            or len(pointwise.broadcast_operands) != 1
            or pointwise.scale != 1.0
            or pointwise.bias != 0.0
            or "bias" in activation_leaf.operand_bindings
            or activation.scale != 1.0
            or activation.bias != 0.0
        ):
            return result
        broadcast_slot = next(iter(pointwise.broadcast_operands))
        data_slots = [slot for slot in pointwise.input_operands if slot != broadcast_slot]
        if len(data_slots) != 1:
            return result
        data = pointwise_leaf.operand_bindings.get(data_slots[0])
        broadcast = pointwise_leaf.operand_bindings.get(broadcast_slot)
        intermediate = pointwise_leaf.operand_bindings.get(pointwise.output_operand)
        activation_input = activation_leaf.operand_bindings.get(activation.input_operand)
        if (
            data is None
            or broadcast is None
            or intermediate is None
            or activation_input is None
            or activation_input.tensor != intermediate.tensor
        ):
            return result
        if len(ir.buffer(data.tensor).shape) != 2 or len(ir.buffer(broadcast.tensor).shape) != 1:
            return result
        if ir.buffer(broadcast.tensor).location not in NKIActivationReduce.INPUT_LOCATIONS["bias"]:
            return result
        if ir.buffer(broadcast.tensor).versions != 1 or ir.buffer(intermediate.tensor).location == "shared_hbm":
            return result
        if not self._has_unique_consumer(ir, intermediate.tensor, pointwise_leaf_nid, activation_leaf_nid):
            return result
        removed_nodes = {pointwise_nid, *ir.tree.descendants(pointwise_nid)}
        removed_allocations = {
            buffer.name
            for nid in removed_nodes
            if isinstance(ir.tree.data(nid), BlockNode)
            for buffer in ir.tree.block(nid).alloc_buffers
        }
        if removed_allocations - {intermediate.tensor}:
            return result
        result = _BroadcastActivationMatch(
            option=option,
            pointwise_leaf_nid=pointwise_leaf_nid,
            activation_leaf_nid=activation_leaf_nid,
            data=data,
            broadcast=broadcast,
            intermediate=intermediate,
            activation_input=activation_input,
        )
        return result

    def _has_unique_consumer(self, ir: KernelIR, tensor: str, producer_leaf_nid: int, consumer_leaf_nid: int) -> bool:
        """Return whether the intermediate has one writer and one static consumer."""
        readers: set[int] = set()
        writers: set[int] = set()
        for nid in ir.tree.preorder():
            node = ir.tree.data(nid)
            if not isinstance(node, ISANode):
                continue
            rmw_operands = node.op_cls.rmw_operands(node.kwargs)
            for slot, region in node.operand_bindings.items():
                if region.tensor != tensor:
                    continue
                if slot in node.op_cls.INPUT_OPERANDS or slot in rmw_operands:
                    readers.add(nid)
                if slot not in node.op_cls.INPUT_OPERANDS:
                    writers.add(nid)
        return readers == {consumer_leaf_nid} and writers == {producer_leaf_nid}

    def _rewrite(self, ir: KernelIR, match: _BroadcastActivationMatch) -> None:
        """Bind the activation bias and remove the absorbed addition."""
        self._remove_pointwise_block(ir, match.option.pointwise_block_nid)

        activation = ir.tree.isa(match.activation_leaf_nid)
        bindings = dict(activation.operand_bindings)
        bindings["data"] = replace(match.activation_input, tensor=match.data.tensor)
        bindings["bias"] = match.broadcast
        kwargs = dict(activation.kwargs)
        kwargs.pop("bias", None)
        ir.tree.graph.nodes[match.activation_leaf_nid]["data"] = replace(
            activation, operand_bindings=bindings, kwargs=kwargs
        )
        activation_block = ir.tree.block(match.option.activation_block_nid)
        writes = activation_block.writes
        ir.tree.graph.nodes[match.option.activation_block_nid]["data"] = replace(
            activation_block, reads=(bindings["data"], bindings["bias"]), writes=writes
        )
        remove_buffers(ir, {match.intermediate.tensor})
        finalize_rewrite(ir)

    def _remove_pointwise_block(self, ir: KernelIR, block_nid: int) -> None:
        """Delete a fully absorbed pointwise block."""
        parent = ir.tree.parent(block_nid)
        if parent is None:
            raise AssertionError(f"pointwise block {block_nid} has no parent")
        _replace_in_parent_children(ir.tree, parent, [block_nid], [])
        ir.tree.graph.remove_nodes_from({block_nid, *ir.tree.descendants(block_nid)})


__all__ = ["FuseBroadcastActivation", "FuseBroadcastActivationOption"]
