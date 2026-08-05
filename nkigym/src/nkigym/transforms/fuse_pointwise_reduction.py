"""Fuse a pointwise producer and its adjacent free-axis reduction."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any

from nkigym.ir import KernelIR
from nkigym.ir.tree import BlockNode, BufferRegion, ISANode
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.base import AxisRole, PointwiseContract, ReductionContract
from nkigym.ops.tensor_scalar_reduce import NKITensorScalarReduce
from nkigym.transforms._canonical_rewrite import block_chain, finalize_rewrite, single_leaf
from nkigym.transforms._tree_ops import _replace_in_parent_children
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption

_ACTIVATION_REDUCERS = frozenset({"add"})
_TENSOR_SCALAR_REDUCERS = frozenset({"add", "maximum", "multiply"})


@dataclass(frozen=True)
class FusePointwiseReductionOption(TransformOption):
    """Identify adjacent pointwise and reduction blocks."""

    pointwise_block_nid: int
    reduction_block_nid: int


@dataclass(frozen=True)
class _Fusion:
    """Resolved fused instruction and rewritten block payloads."""

    option: FusePointwiseReductionOption
    pointwise_leaf_nid: int
    reduction_leaf_nid: int
    op_cls: type[NKIActivationReduce] | type[NKITensorScalarReduce]
    bindings: dict[str, BufferRegion]
    kwargs: dict[str, Any]
    reduction_axis: str


class FusePointwiseReduction(Transform[FusePointwiseReductionOption]):
    """Fuse adjacent contract-compatible pointwise and reduction instructions."""

    def analyze(self, ir: KernelIR) -> list[FusePointwiseReductionOption]:
        """Enumerate adjacent pointwise-reduction pairs with a native fused ISA."""
        options: list[FusePointwiseReductionOption] = []
        for parent_nid in ir.tree.preorder():
            children = ir.tree.children(parent_nid)
            for pointwise_nid, reduction_nid in zip(children, children[1:]):
                option = FusePointwiseReductionOption(
                    pointwise_block_nid=pointwise_nid, reduction_block_nid=reduction_nid
                )
                if self._resolve(ir, option) is not None:
                    options.append(option)
        return options

    def apply(self, ir: KernelIR, option: FusePointwiseReductionOption) -> KernelIR:
        """Re-check ``option``, replace the pair, and rebuild derived metadata."""
        fusion = self._resolve(ir, option)
        if fusion is None:
            raise TransformLegalityError(f"illegal FusePointwiseReduction option: {option}")
        new_ir = copy.deepcopy(ir)
        copied_fusion = self._resolve(new_ir, option)
        if copied_fusion is None:
            raise AssertionError(f"FusePointwiseReduction option disappeared after deepcopy: {option}")
        self._rewrite(new_ir, copied_fusion)
        return new_ir

    def _resolve(self, ir: KernelIR, option: FusePointwiseReductionOption) -> _Fusion | None:
        """Resolve one option to a fused native instruction, if legal."""
        result: _Fusion | None = None
        pointwise_nid = option.pointwise_block_nid
        reduction_nid = option.reduction_block_nid
        if pointwise_nid not in ir.tree.graph or reduction_nid not in ir.tree.graph:
            return result
        if not isinstance(ir.tree.data(pointwise_nid), BlockNode) or not isinstance(
            ir.tree.data(reduction_nid), BlockNode
        ):
            return result
        parent = ir.tree.parent(pointwise_nid)
        siblings = ir.tree.children(parent) if parent is not None else []
        if (
            parent is None
            or ir.tree.parent(reduction_nid) != parent
            or pointwise_nid not in siblings
            or siblings.index(reduction_nid) != siblings.index(pointwise_nid) + 1
        ):
            return result

        pointwise_leaf_nid = single_leaf(ir.tree, pointwise_nid)
        reduction_leaf_nid = single_leaf(ir.tree, reduction_nid)
        pointwise_chain = block_chain(ir.tree, pointwise_nid)
        reduction_chain = block_chain(ir.tree, reduction_nid)
        if (
            pointwise_leaf_nid is None
            or reduction_leaf_nid is None
            or pointwise_chain is None
            or reduction_chain is None
            or pointwise_chain[1:-1] != reduction_chain[1:-1]
        ):
            return result

        pointwise_leaf = ir.tree.isa(pointwise_leaf_nid)
        reduction_leaf = ir.tree.isa(reduction_leaf_nid)
        pointwise_contract = pointwise_leaf.op_cls.algebraic_contract(pointwise_leaf.kwargs)
        reduction_contract = reduction_leaf.op_cls.algebraic_contract(reduction_leaf.kwargs)
        pointwise_block = ir.tree.block(pointwise_nid)
        reduction_block = ir.tree.block(reduction_nid)
        if not isinstance(pointwise_contract, PointwiseContract) or not isinstance(
            reduction_contract, ReductionContract
        ):
            return result
        if not self._blocks_align(pointwise_block, reduction_block, reduction_contract):
            return result
        if (
            reduction_contract.map_operator != "copy"
            or reduction_contract.scale != 1.0
            or reduction_contract.bias != 0.0
        ):
            return result

        mapped = pointwise_leaf.operand_bindings.get(pointwise_contract.output_operand)
        reduced_input = reduction_leaf.operand_bindings.get(reduction_contract.input_operand)
        reduced_output = reduction_leaf.operand_bindings.get(reduction_contract.output_operand)
        if mapped is None or mapped != reduced_input or reduced_output is None:
            return result
        native = self._native_fusion(pointwise_leaf, pointwise_contract, reduction_contract, mapped, reduced_output)
        if native is not None:
            op_cls, bindings, kwargs = native
            result = _Fusion(
                option=option,
                pointwise_leaf_nid=pointwise_leaf_nid,
                reduction_leaf_nid=reduction_leaf_nid,
                op_cls=op_cls,
                bindings=bindings,
                kwargs=kwargs,
                reduction_axis=pointwise_block.axis_map[reduction_contract.reduction_axis],
            )
        return result

    def _blocks_align(self, pointwise: BlockNode, reduction: BlockNode, contract: ReductionContract) -> bool:
        """Return whether both blocks describe the same iteration domain."""
        pointwise_axes = tuple((iter_var.axis, iter_var.dom) for iter_var in pointwise.iter_vars)
        reduction_axes = tuple((iter_var.axis, iter_var.dom) for iter_var in reduction.iter_vars)
        return (
            pointwise.axis_map == reduction.axis_map
            and contract.reduction_axis in pointwise.axis_map
            and pointwise_axes == reduction_axes
            and pointwise.iter_values == reduction.iter_values
        )

    def _native_fusion(
        self,
        leaf: ISANode,
        pointwise: PointwiseContract,
        reduction: ReductionContract,
        mapped: BufferRegion,
        reduced: BufferRegion,
    ) -> tuple[type[NKIActivationReduce] | type[NKITensorScalarReduce], dict[str, BufferRegion], dict[str, Any]] | None:
        """Select and bind the native fused instruction for two contracts."""
        result = self._activation_fusion(leaf, pointwise, reduction, mapped, reduced)
        if result is None:
            result = self._tensor_scalar_fusion(leaf, pointwise, reduction, mapped, reduced)
        return result

    def _activation_fusion(
        self,
        leaf: ISANode,
        pointwise: PointwiseContract,
        reduction: ReductionContract,
        mapped: BufferRegion,
        reduced: BufferRegion,
    ) -> tuple[type[NKIActivationReduce], dict[str, BufferRegion], dict[str, Any]] | None:
        """Build an activation-reduce fusion for one unary pointwise contract."""
        result: tuple[type[NKIActivationReduce], dict[str, BufferRegion], dict[str, Any]] | None = None
        if (
            len(pointwise.input_operands) == 1
            and not pointwise.broadcast_operands
            and not pointwise.reverse
            and reduction.combinator.combiner in _ACTIVATION_REDUCERS
        ):
            source = leaf.operand_bindings.get(pointwise.input_operands[0])
            if source is not None:
                kwargs: dict[str, Any] = {"op": pointwise.operator, "reduce_op": reduction.combinator.combiner}
                bindings = {"data": source, "dst": mapped, "reduce_res": reduced}
                if pointwise.scale != 1.0:
                    kwargs["scale"] = pointwise.scale
                if pointwise.bias != 0.0:
                    kwargs["bias"] = pointwise.bias
                if pointwise.bias_operand is not None and pointwise.bias_operand in leaf.operand_bindings:
                    bindings["bias"] = leaf.operand_bindings[pointwise.bias_operand]
                result = (NKIActivationReduce, bindings, kwargs)
        return result

    def _tensor_scalar_fusion(
        self,
        leaf: ISANode,
        pointwise: PointwiseContract,
        reduction: ReductionContract,
        mapped: BufferRegion,
        reduced: BufferRegion,
    ) -> tuple[type[NKITensorScalarReduce], dict[str, BufferRegion], dict[str, Any]] | None:
        """Build a tensor-scalar-reduce fusion for one broadcast binary contract."""
        result: tuple[type[NKITensorScalarReduce], dict[str, BufferRegion], dict[str, Any]] | None = None
        if (
            len(pointwise.input_operands) == 2
            and len(pointwise.broadcast_operands) == 1
            and not pointwise.reverse
            and pointwise.scale == 1.0
            and pointwise.bias == 0.0
            and reduction.combinator.combiner in _TENSOR_SCALAR_REDUCERS
        ):
            scalar_slot = next(iter(pointwise.broadcast_operands))
            data_slots = [slot for slot in pointwise.input_operands if slot != scalar_slot]
            source = leaf.operand_bindings.get(data_slots[0]) if len(data_slots) == 1 else None
            scalar_region = leaf.operand_bindings.get(scalar_slot)
            scalar_literal = leaf.kwargs.get(scalar_slot)
            if source is not None and (scalar_region is not None or scalar_literal is not None):
                bindings = {"data": source, "dst": mapped, "reduce_res": reduced}
                kwargs: dict[str, Any] = {"op0": pointwise.operator, "reduce_op": reduction.combinator.combiner}
                if scalar_region is not None:
                    bindings["operand0"] = scalar_region
                else:
                    kwargs["operand0"] = scalar_literal
                result = NKITensorScalarReduce, bindings, kwargs
        return result

    def _rewrite(self, ir: KernelIR, fusion: _Fusion) -> None:
        """Replace the pointwise block and delete the redundant reduction block."""
        pointwise_nid = fusion.option.pointwise_block_nid
        reduction_nid = fusion.option.reduction_block_nid
        pointwise_block = ir.tree.block(pointwise_nid)
        reduction_block = ir.tree.block(reduction_nid)
        iter_vars = tuple(
            replace(iter_var, role=AxisRole.ACCUMULATION) if iter_var.axis == fusion.reduction_axis else iter_var
            for iter_var in pointwise_block.iter_vars
        )
        writes = tuple(dict.fromkeys((*pointwise_block.writes, *reduction_block.writes)))
        allocations = tuple(
            {
                buffer.name: buffer for buffer in (*pointwise_block.alloc_buffers, *reduction_block.alloc_buffers)
            }.values()
        )
        ir.tree.graph.nodes[pointwise_nid]["data"] = replace(
            pointwise_block, iter_vars=iter_vars, writes=writes, alloc_buffers=allocations
        )
        ir.tree.graph.nodes[fusion.pointwise_leaf_nid]["data"] = ISANode(
            op_cls=fusion.op_cls, operand_bindings=fusion.bindings, kwargs=fusion.kwargs
        )

        parent = ir.tree.parent(reduction_nid)
        if parent is None:
            raise AssertionError(f"reduction block {reduction_nid} has no parent")
        _replace_in_parent_children(ir.tree, parent, [reduction_nid], [])
        ir.tree.graph.remove_nodes_from({reduction_nid, *ir.tree.descendants(reduction_nid)})
        finalize_rewrite(ir)


__all__ = ["FusePointwiseReduction", "FusePointwiseReductionOption"]
