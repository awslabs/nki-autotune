"""Decompose one matrix-minus-row broadcast into negation followed by addition."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.tree import BlockNode, BufferRegion
from nkigym.ops.activation import NKIActivation
from nkigym.ops.base import PointwiseContract
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.transforms._canonical_rewrite import (
    append_block,
    append_root_buffers,
    finalize_rewrite,
    fresh_name,
    required_spec,
    single_leaf,
)
from nkigym.transforms._tree_ops import _replace_in_parent_children
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class DecomposeBroadcastSubtractOption(TransformOption):
    """Identify one canonical matrix-minus-row pointwise block."""

    pointwise_block_nid: int


@dataclass(frozen=True)
class _Match:
    """Resolved operands for one broadcast-subtraction decomposition."""

    block_nid: int
    leaf_nid: int
    broadcast_operand: str
    broadcast: BufferRegion
    partition_axis: str


class DecomposeBroadcastSubtract(Transform[DecomposeBroadcastSubtractOption]):
    """Rewrite ``data - row`` as ``data + (-row)``."""

    def analyze(self, ir: KernelIR) -> list[DecomposeBroadcastSubtractOption]:
        """Return supported broadcast subtractions."""
        options: list[DecomposeBroadcastSubtractOption] = []
        for block_nid in ir.tree.blocks():
            if block_nid == ir.tree.root:
                continue
            option = DecomposeBroadcastSubtractOption(pointwise_block_nid=block_nid)
            if _resolve(ir, option) is not None:
                options.append(option)
        return options

    def apply(self, ir: KernelIR, option: DecomposeBroadcastSubtractOption) -> KernelIR:
        """Recheck, copy, and introduce one row-vector negation."""
        match = _resolve(ir, option)
        if match is None:
            raise TransformLegalityError(f"illegal DecomposeBroadcastSubtract option: {option}")
        new_ir = copy.deepcopy(ir)
        copied_match = _resolve(new_ir, option)
        if copied_match is None:
            raise AssertionError(f"DecomposeBroadcastSubtract option disappeared after deepcopy: {option}")
        _rewrite(new_ir, copied_match)
        return new_ir


def _resolve(ir: KernelIR, option: DecomposeBroadcastSubtractOption) -> _Match | None:
    """Resolve one canonical tensor-scalar subtraction."""
    result: _Match | None = None
    block_nid = option.pointwise_block_nid
    if (
        block_nid in ir.tree.graph
        and ir.tree.parent(block_nid) is not None
        and isinstance(ir.tree.data(block_nid), BlockNode)
    ):
        leaf_nid = single_leaf(ir.tree, block_nid)
        if leaf_nid is not None:
            leaf = ir.tree.isa(leaf_nid)
            contract = leaf.op_cls.algebraic_contract(leaf.kwargs)
            if (
                leaf.op_cls is NKITensorScalar
                and isinstance(contract, PointwiseContract)
                and contract.operator == "subtract"
                and not contract.reverse
                and contract.scale == 1.0
                and contract.bias == 0.0
                and len(contract.broadcast_operands) == 1
            ):
                broadcast_operand = next(iter(contract.broadcast_operands))
                broadcast = leaf.operand_bindings.get(broadcast_operand)
                block = ir.tree.block(block_nid)
                partition_axis = block.axis_map.get("P")
                partition_extent = next(
                    (
                        iter_var.dom[1] - iter_var.dom[0]
                        for iter_var in block.iter_vars
                        if iter_var.axis == partition_axis
                    ),
                    None,
                )
                broadcast_buffer = ir.buffer(broadcast.tensor) if broadcast is not None else None
                if (
                    broadcast is not None
                    and isinstance(partition_axis, str)
                    and isinstance(partition_extent, int)
                    and broadcast_buffer is not None
                    and broadcast_buffer.location == "sbuf"
                    and broadcast_buffer.shape == (partition_extent,)
                    and broadcast_buffer.versions == 1
                ):
                    result = _Match(
                        block_nid=block_nid,
                        leaf_nid=leaf_nid,
                        broadcast_operand=broadcast_operand,
                        broadcast=broadcast,
                        partition_axis=partition_axis,
                    )
    return result


def _rewrite(ir: KernelIR, match: _Match) -> None:
    """Insert the negation and retarget the subtraction as addition."""
    source_buffer = ir.buffer(match.broadcast.tensor)
    negative_name = fresh_name(ir, f"{match.broadcast.tensor}_negative")
    append_root_buffers(ir, (replace(source_buffer, name=negative_name),))
    spec = required_spec(
        ir,
        NKIActivation,
        {"data": match.broadcast.tensor, "dst": negative_name},
        {"P": match.partition_axis},
        {"op": "copy", "scale": -1.0},
    )
    negation_block = append_block(ir.tree, spec)
    parent = ir.tree.parent(match.block_nid)
    if parent is None:
        raise AssertionError(f"pointwise block {match.block_nid} has no parent")
    _replace_in_parent_children(ir.tree, parent, [match.block_nid], [negation_block, match.block_nid])

    leaf = ir.tree.isa(match.leaf_nid)
    negative_region = replace(match.broadcast, tensor=negative_name)
    bindings = dict(leaf.operand_bindings)
    bindings[match.broadcast_operand] = negative_region
    kwargs = dict(leaf.kwargs)
    kwargs["op0"] = "add"
    ir.tree.graph.nodes[match.leaf_nid]["data"] = replace(leaf, operand_bindings=bindings, kwargs=kwargs)

    block = ir.tree.block(match.block_nid)
    reads = tuple(negative_region if region == match.broadcast else region for region in block.reads)
    ir.tree.graph.nodes[match.block_nid]["data"] = replace(block, reads=reads)
    finalize_rewrite(ir)


__all__ = ["DecomposeBroadcastSubtract", "DecomposeBroadcastSubtractOption"]
