"""Fuse affine binary pointwise work into a tensor-biased activation."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.tree import BlockNode, BufferRegion, ISANode
from nkigym.ops.activation import NKIActivation
from nkigym.ops.base import PointwiseContract
from nkigym.transforms._canonical_rewrite import finalize_rewrite, remove_buffers, single_leaf
from nkigym.transforms._tree_ops import _replace_in_parent_children
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class FusePointwiseActivationOption(TransformOption):
    """Identify adjacent binary-pointwise and activation blocks."""

    pointwise_block_nid: int
    activation_block_nid: int


@dataclass(frozen=True)
class _PointwiseActivationMatch:
    """Resolved native activation data, bias, and scale."""

    option: FusePointwiseActivationOption
    activation_leaf_nid: int
    data: BufferRegion
    bias: BufferRegion
    intermediate: BufferRegion
    scale: float


class FusePointwiseActivation(Transform[FusePointwiseActivationOption]):
    """Use ``activation(data * scale + tensor_bias)`` for an affine pair."""

    def analyze(self, ir: KernelIR) -> list[FusePointwiseActivationOption]:
        """Return adjacent pairs representable by the activation ISA."""
        options: list[FusePointwiseActivationOption] = []
        for parent_nid in ir.tree.preorder():
            children = ir.tree.children(parent_nid)
            for pointwise_nid, activation_nid in zip(children, children[1:]):
                option = FusePointwiseActivationOption(
                    pointwise_block_nid=pointwise_nid, activation_block_nid=activation_nid
                )
                if self._resolve(ir, option) is not None:
                    options.append(option)
        return options

    def apply(self, ir: KernelIR, option: FusePointwiseActivationOption) -> KernelIR:
        """Recheck, copy, bind the native tensor bias, and remove the temporary."""
        match = self._resolve(ir, option)
        if match is None:
            raise TransformLegalityError(f"illegal FusePointwiseActivation option: {option}")
        new_ir = copy.deepcopy(ir)
        copied_match = self._resolve(new_ir, option)
        if copied_match is None:
            raise AssertionError(f"FusePointwiseActivation option disappeared after deepcopy: {option}")
        self._rewrite(new_ir, copied_match)
        return new_ir

    def _resolve(self, ir: KernelIR, option: FusePointwiseActivationOption) -> _PointwiseActivationMatch | None:
        """Resolve one unique-use affine chain accepted by activation storage."""
        result: _PointwiseActivationMatch | None = None
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
        if not isinstance(pointwise, PointwiseContract) or not isinstance(activation, PointwiseContract):
            return result
        if (
            activation_leaf.op_cls is not NKIActivation
            or activation.bias_operand != "bias"
            or len(activation.input_operands) != 1
            or "bias" in activation_leaf.operand_bindings
            or activation.bias != 0.0
            or pointwise.operator not in {"add", "subtract"}
            or len(pointwise.input_operands) != 2
            or pointwise.broadcast_operands
            or pointwise.scale != 1.0
            or pointwise.bias != 0.0
            or pointwise.bias_operand is not None
        ):
            return result
        left = pointwise_leaf.operand_bindings.get(pointwise.input_operands[0])
        right = pointwise_leaf.operand_bindings.get(pointwise.input_operands[1])
        intermediate = pointwise_leaf.operand_bindings.get(pointwise.output_operand)
        activation_input = activation_leaf.operand_bindings.get(activation.input_operands[0])
        if left is None or right is None or intermediate is None or activation_input != intermediate:
            return result
        if pointwise.reverse:
            left, right = right, left
        native = self._native_operands(pointwise.operator, activation.scale, left, right)
        if native is None:
            return result
        data, bias, scale = native
        if (
            len(ir.buffer(data.tensor).shape) != 1
            or len(ir.buffer(bias.tensor).shape) != 1
            or data.ranges != bias.ranges
            or ir.buffer(data.tensor).location not in NKIActivation.INPUT_LOCATIONS["data"]
            or ir.buffer(bias.tensor).location not in NKIActivation.INPUT_LOCATIONS["bias"]
            or ir.buffer(intermediate.tensor).location == "shared_hbm"
            or intermediate.tensor in ir.param_buffers
            or intermediate.tensor == ir.return_name
        ):
            return result
        if not self._has_unique_consumer(ir, intermediate.tensor, pointwise_leaf_nid, activation_leaf_nid):
            return result
        result = _PointwiseActivationMatch(
            option=option,
            activation_leaf_nid=activation_leaf_nid,
            data=data,
            bias=bias,
            intermediate=intermediate,
            scale=scale,
        )
        return result

    def _native_operands(
        self, operator: str, activation_scale: float, left: BufferRegion, right: BufferRegion
    ) -> tuple[BufferRegion, BufferRegion, float] | None:
        """Map a supported affine expression to activation data and bias."""
        result: tuple[BufferRegion, BufferRegion, float] | None = None
        if operator == "add" and activation_scale == 1.0:
            result = (left, right, 1.0)
        elif operator == "subtract" and activation_scale == -1.0:
            result = (left, right, -1.0)
        elif operator == "subtract" and activation_scale == 1.0:
            result = (right, left, -1.0)
        return result

    def _has_unique_consumer(self, ir: KernelIR, tensor: str, producer_leaf_nid: int, consumer_leaf_nid: int) -> bool:
        """Return whether the intermediate has one writer and one reader."""
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

    def _rewrite(self, ir: KernelIR, match: _PointwiseActivationMatch) -> None:
        """Retarget the activation and delete the absorbed pointwise block."""
        activation = ir.tree.isa(match.activation_leaf_nid)
        bindings = dict(activation.operand_bindings)
        bindings["data"] = match.data
        bindings["bias"] = match.bias
        kwargs = dict(activation.kwargs)
        kwargs.pop("bias", None)
        if match.scale == 1.0:
            kwargs.pop("scale", None)
        else:
            kwargs["scale"] = match.scale
        ir.tree.graph.nodes[match.activation_leaf_nid]["data"] = replace(
            activation, operand_bindings=bindings, kwargs=kwargs
        )
        activation_block = ir.tree.block(match.option.activation_block_nid)
        active_abstract_axes = {
            abstract
            for slot, region in bindings.items()
            for abstract in activation.op_cls.OPERAND_AXES[slot][: len(ir.buffer(region.tensor).shape)]
        }
        axis_map = {
            abstract: concrete
            for abstract, concrete in activation_block.axis_map.items()
            if abstract in active_abstract_axes
        }
        active_concrete_axes = set(axis_map.values())
        retained = tuple(
            (iter_var, iter_value)
            for iter_var, iter_value in zip(activation_block.iter_vars, activation_block.iter_values)
            if iter_var.axis in active_concrete_axes
        )
        ir.tree.graph.nodes[match.option.activation_block_nid]["data"] = replace(
            activation_block,
            iter_vars=tuple(item[0] for item in retained),
            iter_values=tuple(item[1] for item in retained),
            reads=(match.data, match.bias),
            axis_map=axis_map,
        )

        parent = ir.tree.parent(match.option.pointwise_block_nid)
        if parent is None:
            raise AssertionError(f"pointwise block {match.option.pointwise_block_nid} has no parent")
        _replace_in_parent_children(ir.tree, parent, [match.option.pointwise_block_nid], [])
        ir.tree.graph.remove_nodes_from(
            {match.option.pointwise_block_nid, *ir.tree.descendants(match.option.pointwise_block_nid)}
        )
        remove_buffers(ir, {match.intermediate.tensor})
        finalize_rewrite(ir)


__all__ = ["FusePointwiseActivation", "FusePointwiseActivationOption"]
