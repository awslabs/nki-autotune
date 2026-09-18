"""Commute one broadcast factor across one algebraic boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace

from nkigym.ir import LE, AccessPattern, Const, Expr, KernelIR, Var
from nkigym.ir.arith.analyzer import Analyzer
from nkigym.ir.arith.expr import expr_variables, substitute
from nkigym.ir.program_sharding import PROGRAM_SHARDS_ANNOTATION, configured_program_shards
from nkigym.ir.tree import BufferRegion, ForNode, ISANode
from nkigym.ops.base import BilinearReductionContract, CopyContract, InitializerContract, PermutationContract
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.tensor_tensor import NKITensorTensor
from nkigym.ops.transpose_broadcast import NKITransposeBroadcast
from nkigym.transforms.base import (
    Transform,
    TransformLegalityError,
    TransformOption,
    copy_for_rewrite,
    intersects_software_pipeline,
)
from nkigym.transforms.copy_propagation import source_remains_stable
from nkigym.transforms.helper.canonical_rewrite import (
    CanonicalSpec,
    append_block,
    append_root_buffers,
    block_chain,
    canonical_spec,
    finalize_rewrite,
    fresh_name,
    is_canonical_block,
    owning_block,
    replace_input_binding,
    rewrite_block,
    single_leaf,
)
from nkigym.transforms.helper.tree_ops import _replace_in_parent_children


@dataclass(frozen=True)
class CommuteBroadcastFactorOption(TransformOption):
    """Identify one transpose or bilinear boundary crossed by a factor."""

    target_nid: int


@dataclass(frozen=True)
class _PermutationMove:
    """One factor immediately before a two-axis permutation."""

    bypass_leaf: int
    permutation_leaf: int
    passthrough: str
    factor: str
    factor_region: BufferRegion


@dataclass(frozen=True)
class _BilinearMove:
    """One transposed factor immediately before an additive matmul."""

    pointwise_leaf: int
    reducer_leaf: int
    passthrough: str
    factor: str
    factor_region: BufferRegion
    drain_leaf: int


class CommuteBroadcastFactor(Transform[CommuteBroadcastFactorOption]):
    """Move one broadcast multiplication across one linear boundary."""

    def analyze(self, ir: KernelIR) -> list[CommuteBroadcastFactorOption]:
        """Return every independently useful one-boundary commute."""
        targets = _permutation_moves(ir).keys() | _bilinear_moves(ir).keys()
        return [CommuteBroadcastFactorOption(target_nid=nid) for nid in sorted(targets)]

    def apply(self, ir: KernelIR, option: CommuteBroadcastFactorOption) -> KernelIR:
        """Recheck and move one factor across the selected boundary."""
        permutation = _permutation_moves(ir).get(option.target_nid)
        bilinear = _bilinear_moves(ir).get(option.target_nid)
        if (permutation is None) == (bilinear is None):
            raise TransformLegalityError(f"illegal broadcast-factor commute option: {option}")
        result = copy_for_rewrite(ir)
        if permutation is not None:
            copied = _permutation_moves(result).get(option.target_nid)
            if copied is None:
                raise AssertionError(f"permutation commute disappeared after deepcopy: {option}")
            _commute_permutation(result, copied)
        else:
            copied = _bilinear_moves(result).get(option.target_nid)
            if copied is None:
                raise AssertionError(f"bilinear commute disappeared after deepcopy: {option}")
            _commute_bilinear(result, copied)
        return result


def _ranges_equal(lhs: tuple[tuple[Expr, Expr], ...], rhs: tuple[tuple[Expr, Expr], ...]) -> bool:
    """Return whether two regions address the same logical coordinates."""
    analyzer = Analyzer()
    return len(lhs) == len(rhs) and all(
        analyzer.can_prove_equal(lhs_lower, rhs_lower) and analyzer.can_prove_equal(lhs_width, rhs_width)
        for (lhs_lower, lhs_width), (rhs_lower, rhs_width) in zip(lhs, rhs, strict=True)
    )


def _rebind_region(ir: KernelIR, source_leaf: int, target_leaf: int, region: BufferRegion) -> BufferRegion:
    """Express a region using the destination block's corresponding axis bindings."""
    source = ir.tree.block(owning_block(ir.tree, source_leaf))
    target = ir.tree.block(owning_block(ir.tree, target_leaf))
    target_values = {axis.axis: value for axis, value in zip(target.iter_vars, target.iter_values, strict=True)}
    substitutions = {
        value.name: target_values[axis.axis]
        for axis, value in zip(source.iter_vars, source.iter_values, strict=True)
        if isinstance(value, Var) and axis.axis in target_values
    }
    return replace(
        region,
        ranges=tuple(
            (substitute(lower, substitutions), substitute(width, substitutions)) for lower, width in region.ranges
        ),
    )


def _permutation_moves(ir: KernelIR) -> dict[int, _PermutationMove]:
    """Return contract-proven factor moves across one transpose."""
    result: dict[int, _PermutationMove] = {}
    positions = {nid: index for index, nid in enumerate(ir.tree.preorder())}
    for bypass_leaf in ir.tree.preorder():
        bypass = ir.tree.data(bypass_leaf)
        if (
            not isinstance(bypass, ISANode)
            or bypass.op_cls is not NKITensorScalar
            or bypass.kwargs.get("op0") != "multiply"
            or not {"data", "operand0", "dst"} <= bypass.operand_bindings.keys()
        ):
            continue
        successors = ir.dependency.direct_consumers(bypass_leaf)
        if len(successors) != 1:
            continue
        permutation_leaf = successors[0]
        permutation = ir.tree.isa(permutation_leaf)
        permutation_contract = permutation.op_cls.algebraic_contract(permutation.kwargs)
        passthrough = bypass.operand_bindings["data"].tensor
        passthrough_region = bypass.operand_bindings["data"]
        factor_region = bypass.operand_bindings["operand0"]
        factor = factor_region.tensor
        output = bypass.operand_bindings["dst"].tensor
        affected = (owning_block(ir.tree, bypass_leaf), owning_block(ir.tree, permutation_leaf))
        if (
            permutation.op_cls is not NKIDMATranspose
            or output in {passthrough, factor}
            or not isinstance(permutation_contract, PermutationContract)
            or permutation_contract.permutation != (1, 0)
            or permutation.operand_bindings.get("src") is None
            or permutation.operand_bindings["src"].tensor != output
            or len(factor_region.ranges) != 1
            or len(passthrough_region.ranges) != 2
            or len(bypass.operand_bindings["dst"].ranges) != 2
        ):
            continue
        permutation_source = permutation.operand_bindings["src"]
        factor_buffer = ir.buffer(factor)
        target_factor = _rebind_region(ir, bypass_leaf, permutation_leaf, factor_region)
        if (
            factor_buffer.location == "sbuf"
            and factor_buffer.physical_dtype() == "float32"
            and factor_buffer.list_len == factor_buffer.versions == 1
            and ir.buffer(passthrough) == replace(ir.buffer(output), name=passthrough)
            and is_canonical_block(ir, affected[1])
            and _ranges_equal(passthrough_region.ranges, bypass.operand_bindings["dst"].ranges)
            and _ranges_equal(
                _rebind_region(ir, bypass_leaf, permutation_leaf, passthrough_region).ranges, permutation_source.ranges
            )
            and _ranges_equal(factor_region.ranges, passthrough_region.ranges[:1])
            and _ranges_equal(target_factor.ranges, permutation_source.ranges[:1])
            and _transpose_broadcast_supports(factor_region)
            and "operand0" not in bypass.access_patterns
            and not intersects_software_pipeline(ir, affected)
            and source_remains_stable(ir, passthrough_region, bypass_leaf, permutation_leaf, positions)
            and source_remains_stable(ir, factor_region, bypass_leaf, permutation_leaf, positions)
        ):
            result[permutation_leaf] = _PermutationMove(
                bypass_leaf=bypass_leaf,
                permutation_leaf=permutation_leaf,
                passthrough=passthrough,
                factor=factor,
                factor_region=target_factor,
            )
    return result


def _transpose_broadcast_supports(factor_region: BufferRegion) -> bool:
    """Return whether one factor width is legal for a native transpose broadcast."""
    maximum = NKITransposeBroadcast.MAX_TILE_SIZE["F"]
    if not isinstance(maximum, int):
        raise AssertionError("transpose broadcast must declare a finite free-axis maximum")
    width = factor_region.ranges[0][1]
    return Analyzer().can_prove(LE(left=width, right=Const(value=maximum)))


def _producer_for_tensor(ir: KernelIR, consumer_leaf: int, tensor: str) -> int | None:
    """Return the sole direct producer writing ``tensor``."""
    producers = [
        nid for nid in ir.dependency.direct_producers(consumer_leaf) if tensor in ir.dependency.info(nid).writes
    ]
    return producers[0] if len(producers) == 1 else None


def _is_identity_writer(ir: KernelIR, leaf_nid: int, target_leaf: int, region: BufferRegion, identity: float) -> bool:
    """Return whether one producer initializes exactly ``region`` to ``identity``."""
    node = ir.tree.isa(leaf_nid)
    contract = node.op_cls.algebraic_contract(node.kwargs)
    return (
        isinstance(contract, InitializerContract)
        and contract.value == identity
        and (written := node.operand_bindings.get(contract.output_operand)) is not None
        and written.tensor == region.tensor
        and _ranges_equal(_rebind_region(ir, leaf_nid, target_leaf, written).ranges, region.ranges)
    )


def _bilinear_moves(ir: KernelIR) -> dict[int, _BilinearMove]:
    """Return factor moves from a transposed pointwise product through one matmul."""
    result: dict[int, _BilinearMove] = {}
    positions = {nid: index for index, nid in enumerate(ir.tree.preorder())}
    for pointwise_leaf in ir.tree.preorder():
        node = ir.tree.data(pointwise_leaf)
        if not isinstance(node, ISANode) or node.op_cls is not NKITensorTensor or node.kwargs.get("op") != "multiply":
            continue
        inputs = {slot: node.operand_bindings[slot].tensor for slot in ("data1", "data2")}
        producers = {slot: _producer_for_tensor(ir, pointwise_leaf, tensor) for slot, tensor in inputs.items()}
        broadcast_slots = [
            slot
            for slot, producer in producers.items()
            if producer is not None and ir.tree.isa(producer).op_cls is NKITransposeBroadcast
        ]
        permutation_slots = [
            slot
            for slot, producer in producers.items()
            if producer is not None
            and isinstance(
                ir.tree.isa(producer).op_cls.algebraic_contract(ir.tree.isa(producer).kwargs), PermutationContract
            )
        ]
        if len(broadcast_slots) != 1 or len(permutation_slots) != 1:
            continue
        broadcast_leaf = producers[broadcast_slots[0]]
        permutation_leaf = producers[permutation_slots[0]]
        assert broadcast_leaf is not None and permutation_leaf is not None
        consumers = [
            nid
            for nid in ir.dependency.direct_consumers(pointwise_leaf)
            if isinstance(
                ir.tree.isa(nid).op_cls.algebraic_contract(ir.tree.isa(nid).kwargs), BilinearReductionContract
            )
        ]
        if len(consumers) != 1:
            continue
        reducer_leaf = consumers[0]
        reducer_node = ir.tree.isa(reducer_leaf)
        reducer = reducer_node.op_cls.algebraic_contract(reducer_node.kwargs)
        assert isinstance(reducer, BilinearReductionContract)
        reducer_output = reducer_node.operand_bindings[reducer.output_operand]
        prior_writers = [
            nid
            for nid in ir.dependency.direct_producers(reducer_leaf)
            if reducer_output.tensor in ir.dependency.info(nid).writes
        ]
        pointwise_output = node.operand_bindings["dst"].tensor
        reducer_slots = [
            slot
            for slot in (reducer.left_operand, reducer.right_operand)
            if reducer_node.operand_bindings[slot].tensor == pointwise_output
        ]
        drains = ir.dependency.direct_consumers(reducer_leaf)
        broadcast_node = ir.tree.isa(broadcast_leaf)
        factor_region = broadcast_node.operand_bindings["data"]
        factor = factor_region.tensor
        passthrough = inputs[permutation_slots[0]]
        blocks = tuple(owning_block(ir.tree, nid) for nid in (pointwise_leaf, permutation_leaf, reducer_leaf, *drains))
        drain = ir.tree.isa(drains[0]) if len(drains) == 1 else None
        drain_source = None if drain is None else drain.operand_bindings.get("src")
        target_factor = (
            _rebind_region(ir, broadcast_leaf, drains[0], factor_region) if len(drains) == 1 else factor_region
        )
        if (
            len(reducer_slots) == 1
            and pointwise_output not in inputs.values()
            and reducer_slots[0] == reducer.left_operand
            and reducer.combinator.combiner == "add"
            and all(
                _is_identity_writer(ir, nid, reducer_leaf, reducer_output, reducer.combinator.identity)
                for nid in prior_writers
            )
            and len(drains) == 1
            and isinstance(
                ir.tree.isa(drains[0]).op_cls.algebraic_contract(ir.tree.isa(drains[0]).kwargs), CopyContract
            )
            and ir.tree.isa(drains[0]).op_cls is NKITensorCopy
            and drain_source is not None
            and len(factor_region.ranges) == 1
            and len(drain_source.ranges) == 2
            and _ranges_equal(target_factor.ranges, drain_source.ranges[:1])
            and ir.buffer(passthrough) == replace(ir.buffer(pointwise_output), name=passthrough)
            and all(
                _ranges_equal(node.operand_bindings[slot].ranges, node.operand_bindings["dst"].ranges)
                for slot in ("data1", "data2")
            )
            and _ranges_equal(
                _rebind_region(ir, broadcast_leaf, pointwise_leaf, broadcast_node.operand_bindings["dst"]).ranges,
                node.operand_bindings[broadcast_slots[0]].ranges,
            )
            and ir.buffer(factor).physical_dtype() == "float32"
            and not intersects_software_pipeline(ir, blocks)
            and source_remains_stable(
                ir, node.operand_bindings[permutation_slots[0]], pointwise_leaf, reducer_leaf, positions
            )
            and source_remains_stable(ir, factor_region, broadcast_leaf, drains[0], positions)
        ):
            result[reducer_leaf] = _BilinearMove(
                pointwise_leaf=pointwise_leaf,
                reducer_leaf=reducer_leaf,
                passthrough=passthrough,
                factor=factor,
                factor_region=target_factor,
                drain_leaf=drains[0],
            )
    return result


def _commute_permutation(ir: KernelIR, move: _PermutationMove) -> None:
    """Move one partition-vector factor across one transpose."""
    permutation_leaf = ir.tree.isa(move.permutation_leaf)
    permutation_block = owning_block(ir.tree, move.permutation_leaf)
    prior_shards = _block_shards(ir, permutation_block)
    block = ir.tree.block(permutation_block)
    output = permutation_leaf.operand_bindings["dst"]
    loop_names = {
        variable.axis: value.name
        for variable, value in zip(block.iter_vars, block.iter_values, strict=True)
        if isinstance(value, Var)
    }
    transposed_name = fresh_name(ir, f"{move.passthrough}_transposed")
    broadcast_name = fresh_name(ir, f"{move.factor}_transposed_broadcast")
    output_buffer = ir.buffer(output.tensor)
    factor_buffer = ir.buffer(move.factor)
    transposed_map = {"P": block.axis_map["F"], "F": block.axis_map["P"]}
    append_root_buffers(
        ir,
        (
            replace(output_buffer, name=transposed_name),
            replace(
                output_buffer,
                name=broadcast_name,
                dtype=factor_buffer.dtype,
                location="psum",
                storage_dtype=factor_buffer.physical_dtype(),
            ),
        ),
    )
    transpose_spec = canonical_spec(
        ir,
        NKIDMATranspose,
        {"src": move.passthrough, "dst": transposed_name},
        dict(block.axis_map),
        {},
        loop_names=loop_names,
    )
    broadcast_spec = canonical_spec(
        ir,
        NKITransposeBroadcast,
        {"data": move.factor, "dst": broadcast_name},
        transposed_map,
        {"partitions": output_buffer.partition_extent()},
        loop_names=loop_names,
    )
    if transpose_spec is None or broadcast_spec is None:
        raise AssertionError("could not reconstruct the transposed broadcast multiplication")
    inputs = (replace(output, tensor=broadcast_name), replace(output, tensor=transposed_name))
    pointwise_spec = CanonicalSpec(
        block=replace(transpose_spec.block, axis_map=transposed_map, reads=inputs, writes=(output,)),
        loops=transpose_spec.loops,
        leaf=ISANode(
            op_cls=NKITensorTensor,
            operand_bindings={"data1": inputs[0], "data2": inputs[1], "dst": output},
            kwargs={"op": "multiply"},
        ),
    )
    rewrite_block(ir.tree, permutation_block, transpose_spec)
    broadcast_block = append_block(ir.tree, broadcast_spec)
    pointwise_block = append_block(ir.tree, pointwise_spec)
    broadcast_leaf = single_leaf(ir.tree, broadcast_block)
    if broadcast_leaf is None:
        raise AssertionError(f"broadcast block {broadcast_block} has no sole ISA leaf")
    broadcast_node = ir.tree.isa(broadcast_leaf)
    if not _ranges_equal(broadcast_node.operand_bindings["data"].ranges, move.factor_region.ranges):
        raise AssertionError("commuted broadcast did not preserve the factor region")
    factor_stride = factor_buffer.logical_tile_count() * factor_buffer.per_tile_physical_shape()[2]
    ir.tree.graph.nodes[broadcast_leaf]["data"] = replace(
        broadcast_node,
        access_patterns={
            "data": AccessPattern(
                pattern=(
                    (Const(value=factor_stride), move.factor_region.ranges[0][1]),
                    (Const(value=0), Const(value=output_buffer.partition_extent())),
                ),
                offset=move.factor_region.ranges[0][0],
            )
        },
    )
    parent = ir.tree.parent(permutation_block)
    if parent is None:
        raise AssertionError(f"permutation block {permutation_block} has no parent")
    _replace_in_parent_children(
        ir.tree, parent, [permutation_block], [permutation_block, broadcast_block, pointwise_block]
    )
    finalize_rewrite(ir)
    _retarget_block_shards(ir, permutation_block, prior_shards)


def _block_shards(ir: KernelIR, block_nid: int) -> dict[int, tuple[str, int, int]]:
    """Return configured shard signatures inside one block."""
    descendants = ir.tree.descendants(block_nid)
    block = ir.tree.block(block_nid)
    axes = {
        name: variable.axis
        for variable, value in zip(block.iter_vars, block.iter_values)
        for name in expr_variables(value)
    }
    return {
        loop_nid: (axes[ir.tree.loop(loop_nid).loop_var], ir.tree.loop(loop_nid).extent, programs)
        for loop_nid, programs in configured_program_shards(ir).items()
        if loop_nid in descendants
    }


def _retarget_block_shards(ir: KernelIR, block_nid: int, prior_shards: dict[int, tuple[str, int, int]]) -> None:
    """Retarget removed shard loops to canonical replacements in the same block."""
    if not prior_shards:
        return
    root = ir.tree.block(ir.tree.root)
    annotations = dict(root.annotations)
    shards = dict(annotations.get(PROGRAM_SHARDS_ANNOTATION, {}))
    descendants = ir.tree.descendants(block_nid)
    block = ir.tree.block(block_nid)
    bindings = {variable.axis: expr_variables(value) for variable, value in zip(block.iter_vars, block.iter_values)}
    for old_nid, (axis, extent, programs) in prior_shards.items():
        if old_nid in ir.tree.graph:
            continue
        matches = [
            nid
            for nid in descendants
            if isinstance(ir.tree.data(nid), ForNode)
            and ir.tree.loop(nid).loop_var in bindings[axis]
            and ir.tree.loop(nid).extent == extent
        ]
        if len(matches) != 1:
            raise AssertionError(f"expected one replacement for sharded loop {old_nid}, found {matches}")
        shards.pop(old_nid, None)
        shards[matches[0]] = programs
    annotations[PROGRAM_SHARDS_ANNOTATION] = shards
    ir.tree.graph.nodes[ir.tree.root]["data"] = replace(root, annotations=annotations)


def _commute_bilinear(ir: KernelIR, move: _BilinearMove) -> None:
    """Move one output-axis factor across one additive matmul."""
    replace_input_binding(ir, move.reducer_leaf, "stationary", move.passthrough)
    drain = ir.tree.isa(move.drain_leaf)
    drain_block_nid = owning_block(ir.tree, move.drain_leaf)
    drain_block = ir.tree.block(drain_block_nid)
    chain = block_chain(ir.tree, drain_block_nid)
    if chain is None:
        raise AssertionError(f"drain block {drain_block_nid} is not a single-ISA chain")
    source = drain.operand_bindings["src"]
    output = drain.operand_bindings["dst"]
    temporary = fresh_name(ir, f"{output.tensor}_normalized")
    append_root_buffers(ir, (replace(ir.buffer(output.tensor), name=temporary),))
    temporary_region = replace(output, tensor=temporary)
    bindings = dict(drain.operand_bindings)
    bindings["src"] = temporary_region
    ir.tree.graph.nodes[move.drain_leaf]["data"] = replace(drain, operand_bindings=bindings)
    ir.tree.graph.nodes[drain_block_nid]["data"] = replace(drain_block, reads=(temporary_region,))
    multiply_block = ir.tree.add_node(
        replace(drain_block, reads=(source, move.factor_region), writes=(temporary_region,), alloc_buffers=())
    )
    parent = multiply_block
    for loop in chain[1:-1]:
        if isinstance(loop, ForNode):
            parent = ir.tree.add_node(loop, parent=parent)
    ir.tree.add_node(
        ISANode(
            op_cls=NKITensorScalar,
            operand_bindings={"data": source, "operand0": move.factor_region, "dst": temporary_region},
            kwargs={"op0": "multiply"},
        ),
        parent=parent,
    )
    owner = ir.tree.parent(drain_block_nid)
    if owner is None:
        raise AssertionError(f"drain block {drain_block_nid} has no parent")
    _replace_in_parent_children(ir.tree, owner, [drain_block_nid], [multiply_block, drain_block_nid])
    finalize_rewrite(ir)


__all__ = ["CommuteBroadcastFactor", "CommuteBroadcastFactorOption"]
