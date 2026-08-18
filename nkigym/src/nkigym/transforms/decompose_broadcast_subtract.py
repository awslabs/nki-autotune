"""Apply one behavior-preserving pointwise algebraic reassociation."""

from __future__ import annotations

from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import to_affine
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode
from nkigym.ops.activation import NKIActivation
from nkigym.ops.base import PointwiseContract
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.search.state_facts import operation_facts
from nkigym.transforms.base import (
    Transform,
    TransformLegalityError,
    TransformOption,
    copy_for_rewrite,
    intersects_software_pipeline,
    software_pipeline_overlap_nodes,
)
from nkigym.transforms.helper.canonical_rewrite import append_root_buffers, finalize_rewrite, fresh_name, single_leaf
from nkigym.transforms.helper.tree_ops import _replace_in_parent_children


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
    """Replace one matrix-minus-row operation with negation followed by addition."""

    def analyze(self, ir: KernelIR) -> list[DecomposeBroadcastSubtractOption]:
        """Return supported broadcast-subtraction decompositions."""
        options: list[DecomposeBroadcastSubtractOption] = []
        if "subtract" in operation_facts(ir).pointwise_operators:
            overlap_nodes = software_pipeline_overlap_nodes(ir)
            for block_nid in ir.tree.blocks():
                if block_nid == ir.tree.root:
                    continue
                option = DecomposeBroadcastSubtractOption(pointwise_block_nid=block_nid)
                if _resolve(ir, option, overlap_nodes) is not None:
                    options.append(option)
        return options

    def apply(self, ir: KernelIR, option: DecomposeBroadcastSubtractOption) -> KernelIR:
        """Recheck and decompose one selected broadcast subtraction."""
        match = _resolve(ir, option)
        if match is None:
            raise TransformLegalityError(f"illegal broadcast-subtraction decomposition: {option}")
        new_ir = copy_for_rewrite(ir)
        copied_match = _resolve(new_ir, option)
        if copied_match is None:
            raise AssertionError(f"broadcast-subtraction option disappeared after deepcopy: {option}")
        _rewrite(new_ir, copied_match)
        return new_ir


def _resolve(
    ir: KernelIR, option: DecomposeBroadcastSubtractOption, overlap_nodes: frozenset[int] | None = None
) -> _Match | None:
    """Resolve one canonical tensor-scalar subtraction."""
    result: _Match | None = None
    block_nid = option.pointwise_block_nid
    if (
        block_nid in ir.tree.graph
        and ir.tree.parent(block_nid) is not None
        and isinstance(ir.tree.data(block_nid), BlockNode)
        and not intersects_software_pipeline(ir, (block_nid,), overlap_nodes)
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
                    and not _has_internal_producer(ir, block_nid, broadcast.tensor)
                ):
                    result = _Match(
                        block_nid=block_nid,
                        leaf_nid=leaf_nid,
                        broadcast_operand=broadcast_operand,
                        broadcast=broadcast,
                        partition_axis=partition_axis,
                    )
    return result


def _has_internal_producer(ir: KernelIR, block_nid: int, tensor: str) -> bool:
    """Return whether ``tensor`` is produced after entry to ``block_nid``."""
    result = False
    for nid in ir.tree.descendants(block_nid):
        if isinstance(ir.tree.data(nid), ISANode) and any(
            region.tensor == tensor for region in ir.dependency.info(nid).write_regions
        ):
            result = True
    return result


def _rewrite(ir: KernelIR, match: _Match) -> None:
    """Insert the negation and retarget the subtraction as addition."""
    source_buffer = ir.buffer(match.broadcast.tensor)
    negative_name = fresh_name(ir, f"{match.broadcast.tensor}_negative")
    append_root_buffers(ir, (replace(source_buffer, name=negative_name),))
    negative_region = replace(match.broadcast, tensor=negative_name)
    negation_block = _append_negation_block(ir, match, negative_region)
    parent = ir.tree.parent(match.block_nid)
    if parent is None:
        raise AssertionError(f"pointwise block {match.block_nid} has no parent")
    _replace_in_parent_children(ir.tree, parent, [match.block_nid], [negation_block, match.block_nid])

    leaf = ir.tree.isa(match.leaf_nid)
    bindings = dict(leaf.operand_bindings)
    bindings[match.broadcast_operand] = negative_region
    kwargs = dict(leaf.kwargs)
    kwargs["op0"] = "add"
    ir.tree.graph.nodes[match.leaf_nid]["data"] = replace(leaf, operand_bindings=bindings, kwargs=kwargs)

    block = ir.tree.block(match.block_nid)
    reads = tuple(negative_region if region == match.broadcast else region for region in block.reads)
    ir.tree.graph.nodes[match.block_nid]["data"] = replace(block, reads=reads)
    finalize_rewrite(ir)


def _append_negation_block(ir: KernelIR, match: _Match, negative_region: BufferRegion) -> int:
    """Append a negation with the pointwise block's exact partition scope."""
    pointwise = ir.tree.block(match.block_nid)
    partition_bindings = [
        (iter_var, iter_value)
        for iter_var, iter_value in zip(pointwise.iter_vars, pointwise.iter_values)
        if iter_var.axis == match.partition_axis
    ]
    if len(partition_bindings) != 1:
        raise AssertionError(
            f"pointwise block {match.block_nid} must bind partition axis {match.partition_axis!r} exactly once"
        )
    partition_iter_var, partition_iter_value = partition_bindings[0]
    bound_names = {name for name in to_affine(partition_iter_value) if name is not None}
    ancestors = ir.tree.ancestors(match.leaf_nid)
    block_index = ancestors.index(match.block_nid)
    local_loops = tuple(
        node
        for nid in ancestors[block_index + 1 :]
        if isinstance((node := ir.tree.data(nid)), ForNode) and node.loop_var in bound_names
    )
    block = BlockNode(
        iter_vars=(partition_iter_var,),
        iter_values=(partition_iter_value,),
        reads=(match.broadcast,),
        writes=(negative_region,),
        alloc_buffers=(),
        axis_map={"P": match.partition_axis},
    )
    block_nid = ir.tree.add_node(block)
    parent_nid = block_nid
    for loop in local_loops:
        parent_nid = ir.tree.add_node(loop, parent=parent_nid)
    ir.tree.add_node(
        ISANode(
            op_cls=NKIActivation,
            operand_bindings={"data": match.broadcast, "dst": negative_region},
            kwargs={"op": "copy", "scale": -1.0},
        ),
        parent=parent_nid,
    )
    return block_nid


__all__ = ["DecomposeBroadcastSubtract", "DecomposeBroadcastSubtractOption"]
