"""Fuse a pointwise producer into one adjacent contract-compatible consumer."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, cast

from nkigym.ir import KernelIR
from nkigym.ir.tree import BlockNode, BufferRegion, ISANode
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.base import AxisRole, CopyContract, PointwiseContract, ReductionContract
from nkigym.ops.tensor_scalar_reduce import NKITensorScalarReduce
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption
from nkigym.transforms.helper.canonical_rewrite import block_chain, finalize_rewrite, remove_buffers, single_leaf
from nkigym.transforms.helper.tree_ops import _replace_in_parent_children

_ACTIVATION_REDUCERS = frozenset({"add"})
_TENSOR_SCALAR_REDUCERS = frozenset({"add", "maximum", "multiply"})


@dataclass(frozen=True)
class FusePointwiseOption(TransformOption):
    """Identify adjacent pointwise producer and consumer blocks."""

    pointwise_block_nid: int
    consumer_block_nid: int


@dataclass(frozen=True)
class _BroadcastActivationMatch:
    """Resolved operands for one activation-reduction bias fusion."""

    option: FusePointwiseOption
    activation_leaf_nid: int
    data: BufferRegion
    broadcast: BufferRegion
    intermediate: BufferRegion
    activation_input: BufferRegion


@dataclass(frozen=True)
class _PointwiseActivationMatch:
    """Resolved operands for one pointwise activation fusion."""

    option: FusePointwiseOption
    activation_leaf_nid: int
    data: BufferRegion
    bias: BufferRegion
    intermediate: BufferRegion
    scale: float


@dataclass(frozen=True)
class _PointwiseCopyMatch:
    """Resolved operands for writing a pointwise result into a copy destination."""

    option: FusePointwiseOption
    pointwise_leaf_nid: int
    copy_leaf_nid: int
    output_operand: str
    intermediate: BufferRegion
    destination: BufferRegion


@dataclass(frozen=True)
class _ReductionFusion:
    """Resolved pointwise-reduction instruction and block payloads."""

    option: FusePointwiseOption
    pointwise_leaf_nid: int
    reduction_leaf_nid: int
    op_cls: type[NKIActivationReduce] | type[NKITensorScalarReduce]
    bindings: dict[str, BufferRegion]
    kwargs: dict[str, Any]
    reduction_axis: str


class FusePointwise(Transform[FusePointwiseOption]):
    """Fuse one adjacent pointwise producer into its native consumer ISA."""

    def analyze(self, ir: KernelIR) -> list[FusePointwiseOption]:
        """Enumerate adjacent pointwise-consumer pairs with a native fused ISA."""
        options: list[FusePointwiseOption] = []
        for parent_nid in ir.tree.preorder():
            children = ir.tree.children(parent_nid)
            for pointwise_nid, consumer_nid in zip(children, children[1:]):
                option = FusePointwiseOption(pointwise_block_nid=pointwise_nid, consumer_block_nid=consumer_nid)
                if self._resolve(ir, option) is not None:
                    options.append(option)
        return options

    def apply(self, ir: KernelIR, option: FusePointwiseOption) -> KernelIR:
        """Re-check ``option``, replace the pair, and rebuild derived metadata."""
        fusion = self._resolve(ir, option)
        if fusion is None:
            raise TransformLegalityError(f"illegal FusePointwise option: {option}")
        new_ir = copy.deepcopy(ir)
        copied_fusion = self._resolve(new_ir, option)
        if copied_fusion is None:
            raise AssertionError(f"FusePointwise option disappeared after deepcopy: {option}")
        self._rewrite(new_ir, copied_fusion)
        return new_ir

    def _resolve(
        self, ir: KernelIR, option: FusePointwiseOption
    ) -> _BroadcastActivationMatch | _PointwiseActivationMatch | _PointwiseCopyMatch | _ReductionFusion | None:
        """Resolve one adjacent pair to exactly one native fusion."""
        fusion = self._resolve_copy(ir, option)
        if fusion is None:
            fusion = self._resolve_activation(ir, option)
        if fusion is None:
            fusion = self._resolve_broadcast_activation(ir, option)
        if fusion is None:
            fusion = self._resolve_reduction(ir, option)
        return fusion

    def _rewrite(
        self,
        ir: KernelIR,
        fusion: _BroadcastActivationMatch | _PointwiseActivationMatch | _PointwiseCopyMatch | _ReductionFusion,
    ) -> None:
        """Apply one resolved native fusion."""
        if isinstance(fusion, _PointwiseCopyMatch):
            self._rewrite_copy(ir, fusion)
        elif isinstance(fusion, _PointwiseActivationMatch):
            self._rewrite_activation(ir, fusion)
        elif isinstance(fusion, _BroadcastActivationMatch):
            self._rewrite_broadcast_activation(ir, fusion)
        else:
            self._rewrite_reduction(ir, fusion)

    def _resolve_copy(self, ir: KernelIR, option: FusePointwiseOption) -> _PointwiseCopyMatch | None:
        """Resolve a pointwise result followed by one value-preserving copy."""
        result: _PointwiseCopyMatch | None = None
        pointwise_nid = option.pointwise_block_nid
        copy_nid = option.consumer_block_nid
        if (
            pointwise_nid in ir.tree.graph
            and copy_nid in ir.tree.graph
            and isinstance(ir.tree.data(pointwise_nid), BlockNode)
            and isinstance(ir.tree.data(copy_nid), BlockNode)
            and self._are_adjacent(ir, pointwise_nid, copy_nid)
        ):
            pointwise_leaf_nid = single_leaf(ir.tree, pointwise_nid)
            copy_leaf_nid = single_leaf(ir.tree, copy_nid)
            pointwise_chain = block_chain(ir.tree, pointwise_nid)
            copy_chain = block_chain(ir.tree, copy_nid)
            if (
                pointwise_leaf_nid is not None
                and copy_leaf_nid is not None
                and pointwise_chain is not None
                and copy_chain is not None
                and pointwise_chain[1:-1] == copy_chain[1:-1]
            ):
                pointwise_leaf = ir.tree.isa(pointwise_leaf_nid)
                copy_leaf = ir.tree.isa(copy_leaf_nid)
                pointwise = pointwise_leaf.op_cls.algebraic_contract(pointwise_leaf.kwargs)
                copy_contract = copy_leaf.op_cls.algebraic_contract(copy_leaf.kwargs)
                pointwise_block = ir.tree.block(pointwise_nid)
                copy_block = ir.tree.block(copy_nid)
                aligned = (
                    pointwise_block.iter_vars == copy_block.iter_vars
                    and pointwise_block.iter_values == copy_block.iter_values
                    and pointwise_block.axis_map == copy_block.axis_map
                )
                if (
                    aligned
                    and isinstance(pointwise, PointwiseContract)
                    and isinstance(copy_contract, CopyContract)
                    and not pointwise_leaf.access_patterns
                    and not copy_leaf.access_patterns
                ):
                    intermediate = pointwise_leaf.operand_bindings.get(pointwise.output_operand)
                    copied = copy_leaf.operand_bindings.get(copy_contract.input_operand)
                    destination = copy_leaf.operand_bindings.get(copy_contract.output_operand)
                    if intermediate is not None and intermediate == copied and destination is not None:
                        source_buffer = ir.buffer(intermediate.tensor)
                        destination_buffer = ir.buffer(destination.tensor)
                        required_dtype = pointwise_leaf.op_cls.OUTPUT_STORAGE_DTYPE
                        legal = (
                            intermediate.tensor not in ir.param_buffers
                            and intermediate.tensor not in ir.return_names
                            and destination.tensor not in ir.param_buffers
                            and destination.tensor not in ir.return_names
                            and source_buffer.dtype == destination_buffer.dtype
                            and destination_buffer.location == pointwise_leaf.op_cls.OUTPUT_LOCATION
                            and (required_dtype is None or destination_buffer.physical_dtype() == required_dtype)
                            and tuple(width for _lower, width in intermediate.ranges)
                            == tuple(width for _lower, width in destination.ranges)
                            and all(buffer.name == destination.tensor for buffer in copy_block.alloc_buffers)
                            and self._has_unique_consumer(ir, intermediate.tensor, pointwise_leaf_nid, copy_leaf_nid)
                            and self._has_unique_writer(ir, destination.tensor, copy_leaf_nid)
                        )
                        if legal:
                            result = _PointwiseCopyMatch(
                                option=option,
                                pointwise_leaf_nid=pointwise_leaf_nid,
                                copy_leaf_nid=copy_leaf_nid,
                                output_operand=pointwise.output_operand,
                                intermediate=intermediate,
                                destination=destination,
                            )
        return result

    def _rewrite_copy(self, ir: KernelIR, match: _PointwiseCopyMatch) -> None:
        """Write the pointwise result to the copy destination and remove the copy."""
        pointwise = ir.tree.isa(match.pointwise_leaf_nid)
        bindings = dict(pointwise.operand_bindings)
        bindings[match.output_operand] = match.destination
        ir.tree.graph.nodes[match.pointwise_leaf_nid]["data"] = replace(pointwise, operand_bindings=bindings)

        pointwise_block = ir.tree.block(match.option.pointwise_block_nid)
        copy_block = ir.tree.block(match.option.consumer_block_nid)
        writes = tuple(
            match.destination if region == match.intermediate else region for region in pointwise_block.writes
        )
        allocations = tuple(
            {
                buffer.name: buffer
                for buffer in (*pointwise_block.alloc_buffers, *copy_block.alloc_buffers)
                if buffer.name != match.intermediate.tensor
            }.values()
        )
        ir.tree.graph.nodes[match.option.pointwise_block_nid]["data"] = replace(
            pointwise_block, writes=writes, alloc_buffers=allocations
        )

        parent = ir.tree.parent(match.option.consumer_block_nid)
        if parent is None:
            raise AssertionError(f"copy block {match.option.consumer_block_nid} has no parent")
        remove_buffers(ir, {match.intermediate.tensor})
        _replace_in_parent_children(ir.tree, parent, [match.option.consumer_block_nid], [])
        ir.tree.graph.remove_nodes_from(
            {match.option.consumer_block_nid, *ir.tree.descendants(match.option.consumer_block_nid)}
        )
        finalize_rewrite(ir)

    def _resolve_activation(self, ir: KernelIR, option: FusePointwiseOption) -> _PointwiseActivationMatch | None:
        """Resolve a pointwise expression accepted by the activation ISA."""
        result: _PointwiseActivationMatch | None = None
        pointwise_nid = option.pointwise_block_nid
        activation_nid = option.consumer_block_nid
        if (
            pointwise_nid in ir.tree.graph
            and activation_nid in ir.tree.graph
            and isinstance(ir.tree.data(pointwise_nid), BlockNode)
            and isinstance(ir.tree.data(activation_nid), BlockNode)
            and self._are_adjacent(ir, pointwise_nid, activation_nid)
        ):
            pointwise_leaf_nid = single_leaf(ir.tree, pointwise_nid)
            activation_leaf_nid = single_leaf(ir.tree, activation_nid)
            if pointwise_leaf_nid is not None and activation_leaf_nid is not None:
                pointwise_leaf = ir.tree.isa(pointwise_leaf_nid)
                activation_leaf = ir.tree.isa(activation_leaf_nid)
                pointwise = pointwise_leaf.op_cls.algebraic_contract(pointwise_leaf.kwargs)
                activation = activation_leaf.op_cls.algebraic_contract(activation_leaf.kwargs)
                contracts = isinstance(pointwise, PointwiseContract) and isinstance(activation, PointwiseContract)
                pointwise_contract = cast(PointwiseContract, pointwise)
                activation_contract = cast(PointwiseContract, activation)
                supported = contracts and (
                    activation_leaf.op_cls is NKIActivation
                    and activation_contract.bias_operand == "bias"
                    and len(activation_contract.input_operands) == 1
                    and "bias" not in activation_leaf.operand_bindings
                    and activation_contract.bias == 0.0
                    and pointwise_contract.operator in {"add", "subtract"}
                    and len(pointwise_contract.input_operands) == 2
                    and not pointwise_contract.broadcast_operands
                    and pointwise_contract.scale == 1.0
                    and pointwise_contract.bias == 0.0
                    and pointwise_contract.bias_operand is None
                    and not pointwise_leaf.access_patterns
                    and not activation_leaf.access_patterns
                )
                if supported:
                    left = pointwise_leaf.operand_bindings.get(pointwise_contract.input_operands[0])
                    right = pointwise_leaf.operand_bindings.get(pointwise_contract.input_operands[1])
                    intermediate = pointwise_leaf.operand_bindings.get(pointwise_contract.output_operand)
                    activation_input = activation_leaf.operand_bindings.get(activation_contract.input_operands[0])
                    if left is not None and right is not None and intermediate is not None:
                        if pointwise_contract.reverse:
                            left, right = right, left
                        native = self._native_activation_operands(
                            pointwise_contract.operator, activation_contract.scale, left, right
                        )
                        if native is not None and activation_input == intermediate:
                            data, bias, scale = native
                            buffers = ir.all_buffers()
                            legal = (
                                len(buffers[data.tensor].shape) == 1
                                and len(buffers[bias.tensor].shape) == 1
                                and data.ranges == bias.ranges
                                and buffers[data.tensor].location in NKIActivation.INPUT_LOCATIONS["data"]
                                and buffers[bias.tensor].location in NKIActivation.INPUT_LOCATIONS["bias"]
                                and buffers[intermediate.tensor].location != "shared_hbm"
                                and intermediate.tensor not in ir.param_buffers
                                and intermediate.tensor not in ir.return_names
                                and self._has_unique_consumer(
                                    ir, intermediate.tensor, pointwise_leaf_nid, activation_leaf_nid
                                )
                            )
                            if legal:
                                result = _PointwiseActivationMatch(
                                    option=option,
                                    activation_leaf_nid=activation_leaf_nid,
                                    data=data,
                                    bias=bias,
                                    intermediate=intermediate,
                                    scale=scale,
                                )
        return result

    def _resolve_broadcast_activation(
        self, ir: KernelIR, option: FusePointwiseOption
    ) -> _BroadcastActivationMatch | None:
        """Resolve a broadcast addition accepted as activation-reduction bias."""
        result: _BroadcastActivationMatch | None = None
        pointwise_nid = option.pointwise_block_nid
        activation_nid = option.consumer_block_nid
        if (
            pointwise_nid in ir.tree.graph
            and activation_nid in ir.tree.graph
            and isinstance(ir.tree.data(pointwise_nid), BlockNode)
            and isinstance(ir.tree.data(activation_nid), BlockNode)
            and self._are_adjacent(ir, pointwise_nid, activation_nid)
        ):
            pointwise_leaf_nid = single_leaf(ir.tree, pointwise_nid)
            activation_leaf_nid = single_leaf(ir.tree, activation_nid)
            if pointwise_leaf_nid is not None and activation_leaf_nid is not None:
                pointwise_leaf = ir.tree.isa(pointwise_leaf_nid)
                activation_leaf = ir.tree.isa(activation_leaf_nid)
                pointwise = pointwise_leaf.op_cls.algebraic_contract(pointwise_leaf.kwargs)
                activation = activation_leaf.op_cls.algebraic_contract(activation_leaf.kwargs)
                contracts = isinstance(pointwise, PointwiseContract) and isinstance(activation, ReductionContract)
                pointwise_contract = cast(PointwiseContract, pointwise)
                activation_contract = cast(ReductionContract, activation)
                supported = contracts and (
                    activation_leaf.op_cls is NKIActivationReduce
                    and activation_contract.bias_operand == "bias"
                    and pointwise_contract.operator == "add"
                    and not pointwise_contract.reverse
                    and len(pointwise_contract.input_operands) == 2
                    and len(pointwise_contract.broadcast_operands) == 1
                    and pointwise_contract.scale == 1.0
                    and pointwise_contract.bias == 0.0
                    and "bias" not in activation_leaf.operand_bindings
                    and activation_contract.scale == 1.0
                    and activation_contract.bias == 0.0
                    and not pointwise_leaf.access_patterns
                    and not activation_leaf.access_patterns
                )
                if supported:
                    broadcast_slot = next(iter(pointwise_contract.broadcast_operands))
                    data_slots = [slot for slot in pointwise_contract.input_operands if slot != broadcast_slot]
                    data = pointwise_leaf.operand_bindings.get(data_slots[0]) if len(data_slots) == 1 else None
                    broadcast = pointwise_leaf.operand_bindings.get(broadcast_slot)
                    intermediate = pointwise_leaf.operand_bindings.get(pointwise_contract.output_operand)
                    activation_input = activation_leaf.operand_bindings.get(activation_contract.input_operand)
                    if (
                        data is not None
                        and broadcast is not None
                        and intermediate is not None
                        and activation_input is not None
                        and activation_input.tensor == intermediate.tensor
                    ):
                        buffers = ir.all_buffers()
                        removed_nodes = {pointwise_nid, *ir.tree.descendants(pointwise_nid)}
                        removed_allocations = {
                            buffer.name
                            for nid in removed_nodes
                            if isinstance(ir.tree.data(nid), BlockNode)
                            for buffer in ir.tree.block(nid).alloc_buffers
                        }
                        legal = (
                            len(buffers[data.tensor].shape) == 2
                            and len(buffers[broadcast.tensor].shape) == 1
                            and buffers[broadcast.tensor].location in NKIActivationReduce.INPUT_LOCATIONS["bias"]
                            and buffers[broadcast.tensor].versions == 1
                            and buffers[intermediate.tensor].location != "shared_hbm"
                            and self._has_unique_consumer(
                                ir, intermediate.tensor, pointwise_leaf_nid, activation_leaf_nid
                            )
                            and not removed_allocations - {intermediate.tensor}
                        )
                        if legal:
                            result = _BroadcastActivationMatch(
                                option=option,
                                activation_leaf_nid=activation_leaf_nid,
                                data=data,
                                broadcast=broadcast,
                                intermediate=intermediate,
                                activation_input=activation_input,
                            )
        return result

    def _are_adjacent(self, ir: KernelIR, producer_nid: int, consumer_nid: int) -> bool:
        """Return whether two blocks are adjacent siblings in that order."""
        parent = ir.tree.parent(producer_nid)
        siblings = ir.tree.children(parent) if parent is not None else []
        return (
            parent is not None
            and ir.tree.parent(consumer_nid) == parent
            and producer_nid in siblings
            and siblings.index(consumer_nid) == siblings.index(producer_nid) + 1
        )

    def _native_activation_operands(
        self, operator: str, activation_scale: float, left: BufferRegion, right: BufferRegion
    ) -> tuple[BufferRegion, BufferRegion, float] | None:
        """Map an affine expression to activation data, bias, and scale."""
        result: tuple[BufferRegion, BufferRegion, float] | None = None
        if operator == "add" and activation_scale == 1.0:
            result = (left, right, 1.0)
        elif operator == "subtract" and activation_scale == -1.0:
            result = (left, right, -1.0)
        elif operator == "subtract" and activation_scale == 1.0:
            result = (right, left, -1.0)
        return result

    def _has_unique_consumer(self, ir: KernelIR, tensor: str, producer_leaf_nid: int, consumer_leaf_nid: int) -> bool:
        """Return whether a temporary has one writer and one reader."""
        readers: set[int] = set()
        writers: set[int] = set()
        for nid in ir.tree.preorder():
            node = ir.tree.data(nid)
            if isinstance(node, ISANode):
                rmw_operands = node.op_cls.rmw_operands(node.kwargs)
                for slot, region in node.operand_bindings.items():
                    if region.tensor == tensor:
                        if slot in node.op_cls.INPUT_OPERANDS or slot in rmw_operands:
                            readers.add(nid)
                        if slot not in node.op_cls.INPUT_OPERANDS:
                            writers.add(nid)
        return readers == {consumer_leaf_nid} and writers == {producer_leaf_nid}

    def _has_unique_writer(self, ir: KernelIR, tensor: str, writer_leaf_nid: int) -> bool:
        """Return whether one leaf is the tensor's only writer."""
        writers: set[int] = set()
        for nid in ir.tree.preorder():
            node = ir.tree.data(nid)
            if not isinstance(node, ISANode):
                continue
            for slot, region in node.operand_bindings.items():
                if region.tensor == tensor and slot not in node.op_cls.INPUT_OPERANDS:
                    writers.add(nid)
        return writers == {writer_leaf_nid}

    def _rewrite_activation(self, ir: KernelIR, match: _PointwiseActivationMatch) -> None:
        """Retarget an activation and remove its pointwise producer."""
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
        activation_block = ir.tree.block(match.option.consumer_block_nid)
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
        ir.tree.graph.nodes[match.option.consumer_block_nid]["data"] = replace(
            activation_block,
            iter_vars=tuple(item[0] for item in retained),
            iter_values=tuple(item[1] for item in retained),
            reads=(match.data, match.bias),
            axis_map=axis_map,
        )
        self._remove_pointwise_block(ir, match.option.pointwise_block_nid)
        remove_buffers(ir, {match.intermediate.tensor})
        finalize_rewrite(ir)

    def _rewrite_broadcast_activation(self, ir: KernelIR, match: _BroadcastActivationMatch) -> None:
        """Bind activation-reduction bias and remove its pointwise producer."""
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
        activation_block = ir.tree.block(match.option.consumer_block_nid)
        ir.tree.graph.nodes[match.option.consumer_block_nid]["data"] = replace(
            activation_block, reads=(bindings["data"], bindings["bias"])
        )
        remove_buffers(ir, {match.intermediate.tensor})
        finalize_rewrite(ir)

    def _remove_pointwise_block(self, ir: KernelIR, block_nid: int) -> None:
        """Delete one fully absorbed pointwise block."""
        parent = ir.tree.parent(block_nid)
        if parent is None:
            raise AssertionError(f"pointwise block {block_nid} has no parent")
        _replace_in_parent_children(ir.tree, parent, [block_nid], [])
        ir.tree.graph.remove_nodes_from({block_nid, *ir.tree.descendants(block_nid)})

    def _resolve_reduction(self, ir: KernelIR, option: FusePointwiseOption) -> _ReductionFusion | None:
        """Resolve one option to a fused native instruction, if legal."""
        result: _ReductionFusion | None = None
        pointwise_nid = option.pointwise_block_nid
        reduction_nid = option.consumer_block_nid
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
            result = _ReductionFusion(
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

    def _rewrite_reduction(self, ir: KernelIR, fusion: _ReductionFusion) -> None:
        """Replace the pointwise block and delete the redundant reduction block."""
        pointwise_nid = fusion.option.pointwise_block_nid
        reduction_nid = fusion.option.consumer_block_nid
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


__all__ = ["FusePointwise", "FusePointwiseOption"]
