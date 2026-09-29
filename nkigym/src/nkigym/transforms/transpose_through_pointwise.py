"""Commute a transpose through one elementwise producer."""

from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Const, Var
from nkigym.ir.program_sharding import configured_program_shards
from nkigym.ir.tree import BlockNode, Buffer, ISANode
from nkigym.ops.base import AxisRole, CopyContract, PointwiseContract, PointwiseSequenceContract
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.transpose import NKITranspose
from nkigym.transforms.base import (
    Transform,
    TransformLegalityError,
    TransformOption,
    copy_for_rewrite,
    intersects_software_pipeline,
)
from nkigym.transforms.helper.canonical_rewrite import (
    append_root_buffers,
    block_chain,
    canonical_spec,
    finalize_rewrite,
    fresh_name,
    is_canonical_block,
    remove_buffers,
    replace_buffer,
    required_spec,
    rewrite_block,
    single_leaf,
)
from nkigym.transforms.helper.transpose_pattern import match_transpose_chain
from nkigym.transforms.helper.tree_ops import _replace_in_parent_children


@dataclass(frozen=True)
class TransposeThroughPointwiseOption(TransformOption):
    """Select a transpose immediately following an elementwise producer."""

    transpose_nid: int


@dataclass(frozen=True)
class _Transpose:
    """One direct DMA transpose or logical transpose with its required drain."""

    transpose_block: int
    transpose_leaf: int
    source: str
    output: str
    source_axes: tuple[str, str]
    drain_block: int | None
    drain_leaf: int | None
    psum: str | None


@dataclass(frozen=True)
class _Match:
    """One elementwise producer followed by a materialized transpose."""

    producer_block: int
    producer_leaf: int
    chain: _Transpose
    input_operand: str
    output_operand: str
    source: Buffer


class TransposeThroughPointwise(Transform[TransposeThroughPointwiseOption]):
    """Apply ``T(f(x)) = f(T(x))`` without changing the elementwise operation.

    Only bound rank-two input data is moved; other operands must be literals.
    Retain the producer's tile sizes and loop order exactly. Split must prepare
    tiles that fit and align with the transposed partition storage. The existing
    following copy stays after the producer. A logical transpose requires the
    producer to accept PSUM input; DMA transposes retain SBUF input.
    """

    def analyze(self, ir: KernelIR) -> list[TransposeThroughPointwiseOption]:
        """Return adjacent transposes whose pointwise tiles already fit."""
        children = tuple(ir.tree.children(ir.tree.root))
        return [
            option
            for nid in children
            if _match(ir, option := TransposeThroughPointwiseOption(nid), children) is not None
        ]

    def apply(self, ir: KernelIR, option: TransposeThroughPointwiseOption) -> KernelIR:
        """Recheck, copy, and transpose the input before the same computation."""
        match = _match(ir, option, tuple(ir.tree.children(ir.tree.root)))
        if match is None:
            raise TransformLegalityError(f"illegal TransposeThroughPointwise option: {option}")
        result = copy_for_rewrite(ir)
        _rewrite(result, match)
        finalize_rewrite(result)
        return result


def _match(ir: KernelIR, option: TransposeThroughPointwiseOption, children: tuple[int, ...]) -> _Match | None:
    """Resolve one private, canonical value edge and its transpose chain."""
    if option.transpose_nid not in children:
        return None
    index = children.index(option.transpose_nid)
    if index == 0:
        return None
    producer = children[index - 1]
    following = children[index + 1] if index + 1 < len(children) else None
    chain = _transpose(ir, option.transpose_nid, following)
    if chain is None or not _tiled_producer(ir, producer):
        return None
    affected = (producer, chain.transpose_block, *((chain.drain_block,) if chain.drain_block is not None else ()))
    if intersects_software_pipeline(ir, affected) or any(
        ir.tree.block(nid).annotations or ir.tree.block(nid).alloc_buffers for nid in affected
    ):
        return None
    if any(loop in ir.tree.descendants(block) for loop in configured_program_shards(ir) for block in affected):
        return None
    leaf_nid = single_leaf(ir.tree, producer)
    if leaf_nid is None:
        return None
    node = ir.tree.isa(leaf_nid)
    contract = node.op_cls.algebraic_contract(node.kwargs)
    if not isinstance(contract, (PointwiseContract, PointwiseSequenceContract, CopyContract)):
        return None
    inputs = [slot for slot in node.operand_bindings if slot in node.op_cls.INPUT_OPERANDS]
    if len(inputs) != 1 or set(node.operand_bindings) != {inputs[0], contract.output_operand}:
        return None
    input_operand = inputs[0]
    source = ir.buffer(node.operand_bindings[input_operand].tensor)
    produced = ir.buffer(node.operand_bindings[contract.output_operand].tensor)
    if (
        produced.name != chain.source
        or source.shape != produced.shape
        or len(source.shape) != 2
        or source.location != "sbuf"
        or chain.drain_block is not None
        and "psum" not in node.op_cls.INPUT_LOCATIONS.get(input_operand, ())
        or produced.name in ir.return_names
        or not _private_result(ir, produced.name, leaf_nid, chain.transpose_leaf)
        or node.access_patterns
        or node.op_cls.rmw_operands(node.kwargs)
        or any(variable.role is not AxisRole.PARALLEL for variable in ir.tree.block(producer).iter_vars)
        or not ir.tree.isa(chain.transpose_leaf).op_cls.accepts_input_storage_dtypes(
            {("src" if chain.drain_block is None else "data"): source.physical_dtype()}
        )
    ):
        return None
    axes = node.op_cls.OPERAND_AXES[input_operand]
    output_axes = node.op_cls.OPERAND_AXES[contract.output_operand]
    block = ir.tree.block(producer)
    if axes != output_axes or len(axes) != 2 or tuple(block.axis_map[axis] for axis in axes) != chain.source_axes:
        return None
    intermediate = replace(ir.buffer(chain.output), name=produced.name)
    if (
        intermediate.partition_size is not None
        and intermediate.shape[0] % intermediate.partition_size
        or intermediate.logical_tile_count() % intermediate.list_len
    ):
        return None
    mapping = _swapped_axes(block.axis_map, axes)
    template = _swapped_tiles(node)
    partition = template.operand_bindings[input_operand].ranges[0][1]
    storage = (ir.buffer(chain.output),) + ((ir.buffer(chain.psum),) if chain.psum is not None else ())
    if not isinstance(partition, Const) or any(
        partition.value > buffer.partition_extent() or source.shape[1] > partition.value != buffer.partition_extent()
        for buffer in storage
    ):
        return None
    target = canonical_spec(
        ir,
        node.op_cls,
        {input_operand: produced.name, contract.output_operand: chain.output},
        mapping,
        node.kwargs,
        loop_names=_loop_names(block),
        tile_template=template,
    )
    if target is None or any(
        tuple(width for _, width in region.ranges)
        != tuple(width for _, width in template.operand_bindings[slot].ranges)
        for slot, region in target.leaf.operand_bindings.items()
    ):
        return None
    return _Match(producer, leaf_nid, chain, input_operand, contract.output_operand, source)


def _swapped_axes(mapping: dict[str, str], axes: tuple[str, ...]) -> dict[str, str]:
    """Relabel the two data axes while retaining their concrete iteration order."""
    swapped = dict(zip(axes, reversed(axes), strict=True))
    return {swapped.get(axis, axis): concrete for axis, concrete in mapping.items()}


def _swapped_tiles(node: ISANode) -> ISANode:
    """Carry each existing pointwise tile through the coordinate permutation."""
    return replace(
        node,
        operand_bindings={
            slot: replace(region, ranges=region.ranges[::-1]) for slot, region in node.operand_bindings.items()
        },
    )


def _loop_names(block: BlockNode) -> dict[str, str]:
    """Retain the loop names belonging to each concrete producer axis."""
    return {
        variable.axis: value.name
        for variable, value in zip(block.iter_vars, block.iter_values, strict=True)
        if isinstance(value, Var)
    }


def _tiled_producer(ir: KernelIR, nid: int) -> bool:
    """Recognize a regular pointwise tile grid without selecting new tile sizes."""
    chain = block_chain(ir.tree, nid)
    leaf = single_leaf(ir.tree, nid)
    if chain is None or leaf is None:
        return False
    block, node = ir.tree.block(nid), ir.tree.isa(leaf)
    spec = canonical_spec(
        ir,
        node.op_cls,
        {slot: region.tensor for slot, region in node.operand_bindings.items()},
        block.axis_map,
        node.kwargs,
        loop_names=_loop_names(block),
        tile_template=node,
    )
    return spec is not None and (replace(block, alloc_buffers=()), *chain[1:]) == (spec.block, *spec.loops, spec.leaf)


def _private_result(ir: KernelIR, tensor: str, producer: int, transpose: int) -> bool:
    """Permit earlier allocation reuse while requiring the new value's sole reader."""
    positions = {nid: index for index, nid in enumerate(ir.tree.preorder())}
    return all(
        nid in {producer, transpose} or positions[nid] < positions[producer]
        for nid in ir.dependency.touches_by_tensor.get(tensor, ())
    )


def _transpose(ir: KernelIR, block: int, following: int | None) -> _Transpose | None:
    """Match an explicit transpose with unchanged logical and physical values."""
    logical = match_transpose_chain(ir, block, following, adjacent=True) if following is not None else None
    if logical is not None:
        return _Transpose(
            logical.transpose_block,
            logical.transpose_leaf,
            logical.source,
            logical.output,
            logical.source_axes,
            logical.drain_block,
            logical.drain_leaf,
            logical.psum,
        )
    leaf = single_leaf(ir.tree, block)
    if leaf is None or not is_canonical_block(ir, block) or ir.tree.isa(leaf).op_cls is not NKIDMATranspose:
        return None
    node = ir.tree.isa(leaf)
    source, output = (ir.buffer(node.operand_bindings[slot].tensor) for slot in ("src", "dst"))
    axes = ir.tree.block(block).axis_map
    if (
        source.location != "sbuf"
        or output.location != "sbuf"
        or len(source.shape) != 2
        or output.shape != source.shape[::-1]
        or source.dtype != output.dtype
        or source.physical_dtype() != output.physical_dtype()
    ):
        return None
    return _Transpose(block, leaf, source.name, output.name, (axes["P"], axes["F"]), None, None, None)


def _rewrite(ir: KernelIR, match: _Match) -> None:
    """Swap the producer and transpose while retaining the following copy's order."""
    chain = match.chain
    node = ir.tree.isa(match.producer_leaf)
    transpose = ir.tree.isa(chain.transpose_leaf)
    transpose_axes = dict(ir.tree.block(chain.transpose_block).axis_map)
    axes = node.op_cls.OPERAND_AXES[match.input_operand]
    producer_axes = _swapped_axes(ir.tree.block(match.producer_block).axis_map, axes)
    source = match.source
    representation = ir.buffer(chain.source) if chain.drain_block is not None else source
    suffix = "result" if chain.drain_block is not None else "input"
    intermediate = replace(
        ir.buffer(chain.output),
        name=fresh_name(ir, f"{chain.source}_transposed_{suffix}"),
        dtype=representation.dtype,
        storage_dtype=representation.physical_dtype(),
    )
    append_root_buffers(ir, (intermediate,))
    old_order = [match.producer_block, chain.transpose_block]
    if chain.drain_block is None:
        specs = [
            (
                chain.transpose_block,
                required_spec(
                    ir,
                    NKIDMATranspose,
                    {"src": source.name, "dst": intermediate.name},
                    transpose_axes,
                    transpose.kwargs,
                ),
            )
        ]
        pointwise_input, pointwise_output = intermediate.name, chain.output
    else:
        if chain.psum is None or chain.drain_leaf is None:
            raise AssertionError("logical transpose lost its materialization")
        psum = replace(ir.buffer(chain.psum), dtype=source.dtype, storage_dtype=source.physical_dtype())
        replace_buffer(ir, psum)
        drain = ir.tree.isa(chain.drain_leaf)
        drain_axes = dict(ir.tree.block(chain.drain_block).axis_map)
        specs = [
            (
                chain.transpose_block,
                required_spec(
                    ir, NKITranspose, {"data": source.name, "dst": psum.name}, transpose_axes, transpose.kwargs
                ),
            ),
            (
                chain.drain_block,
                required_spec(
                    ir, NKITensorCopy, {"src": intermediate.name, "dst": chain.output}, drain_axes, drain.kwargs
                ),
            ),
        ]
        old_order.append(chain.drain_block)
        pointwise_input, pointwise_output = psum.name, intermediate.name
    pointwise = canonical_spec(
        ir,
        node.op_cls,
        {match.input_operand: pointwise_input, match.output_operand: pointwise_output},
        producer_axes,
        node.kwargs,
        loop_names=_loop_names(ir.tree.block(match.producer_block)),
        tile_template=_swapped_tiles(node),
    )
    if pointwise is None:
        raise AssertionError("pointwise transpose cannot preserve its existing tile geometry")
    specs.insert(1, (match.producer_block, pointwise))
    for nid, spec in specs:
        rewrite_block(ir.tree, nid, spec)
    _replace_in_parent_children(ir.tree, ir.tree.root, old_order, [nid for nid, _spec in specs])
    if not any(
        isinstance(node := ir.tree.data(nid), ISANode)
        and any(region.tensor == chain.source for region in node.operand_bindings.values())
        for nid in ir.tree.preorder()
    ):
        remove_buffers(ir, {chain.source})


__all__ = ["TransposeThroughPointwise", "TransposeThroughPointwiseOption"]
