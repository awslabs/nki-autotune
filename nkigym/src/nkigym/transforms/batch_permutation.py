"""Batch independent permutation instructions through a preserved view axis."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from nkigym.ir import AccessPattern, Add, Const, Expr, KernelIR, Mul, substitute, to_affine
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import PARTITION_DIM, BlockNode, Buffer, BufferRegion, ForNode, ISANode
from nkigym.ops.base import AxisRole, BatchedPermutationContract, PermutationContract
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption
from nkigym.transforms.helper.tree_ops import _replace_in_parent_children, invalidate_stale_software_pipelines


@dataclass(frozen=True)
class BatchPermutationOption(TransformOption):
    """Absorb ``loop_nid`` as the batch axis of its permutation instruction."""

    loop_nid: int


@dataclass(frozen=True)
class _BatchMatch:
    """Resolved permutation loop and its affine batch geometry."""

    block_nid: int
    loop_nid: int
    leaf_nid: int
    contract: PermutationContract
    batching: BatchedPermutationContract
    source_axis: int
    output_axis: int


class BatchPermutation(Transform[BatchPermutationOption]):
    """Replace a loop of independent permutations with one batched permutation."""

    def analyze(self, ir: KernelIR) -> list[BatchPermutationOption]:
        """Return every directly tensorizable permutation loop."""
        options: list[BatchPermutationOption] = []
        for nid in ir.tree.preorder():
            if _match_loop(ir, nid) is not None:
                options.append(BatchPermutationOption(loop_nid=nid))
        return options

    def apply(self, ir: KernelIR, option: BatchPermutationOption) -> KernelIR:
        """Recheck, copy, and absorb one loop into access-pattern views."""
        match = _match_option(ir, option)
        if match is None:
            raise TransformLegalityError(
                f"BatchPermutation loop {option.loop_nid} is not an eligible permutation batch"
            )
        new_ir = copy.deepcopy(ir)
        copied_match = _match_option(new_ir, option)
        if copied_match is None:
            raise AssertionError("BatchPermutation match disappeared after deepcopy")
        _apply_match(new_ir, copied_match)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir


def _match_option(ir: KernelIR, option: BatchPermutationOption) -> _BatchMatch | None:
    """Resolve ``option`` without accepting an unknown node id."""
    result: _BatchMatch | None = None
    if option.loop_nid in ir.tree.graph:
        result = _match_loop(ir, option.loop_nid)
    return result


def _match_loop(ir: KernelIR, loop_nid: int) -> _BatchMatch | None:
    """Return the contract and geometry for one eligible loop."""
    result: _BatchMatch | None = None
    node = ir.tree.data(loop_nid)
    children = ir.tree.children(loop_nid) if isinstance(node, ForNode) else []
    if isinstance(node, ForNode) and node.extent > 1 and len(children) == 1:
        leaf_nid = children[0]
        leaf = ir.tree.data(leaf_nid)
        if isinstance(leaf, ISANode) and not leaf.access_patterns:
            contract = leaf.op_cls.algebraic_contract(leaf.kwargs)
            if isinstance(contract, PermutationContract) and contract.batching is not None:
                block_nid = _owning_block(ir, leaf_nid)
                axes = _match_geometry(ir, block_nid, node, leaf, contract)
                if axes is not None:
                    result = _BatchMatch(
                        block_nid=block_nid,
                        loop_nid=loop_nid,
                        leaf_nid=leaf_nid,
                        contract=contract,
                        batching=contract.batching,
                        source_axis=axes[0],
                        output_axis=axes[1],
                    )
    return result


def _match_geometry(
    ir: KernelIR, block_nid: int, loop: ForNode, leaf: ISANode, contract: PermutationContract
) -> tuple[int, int] | None:
    """Return varying input/output axes when regions form contiguous batches."""
    result: tuple[int, int] | None = None
    batching = contract.batching
    if batching is None or not _valid_batch_contract(contract, batching):
        return result
    if set(leaf.operand_bindings) != {contract.input_operand, contract.output_operand}:
        return result
    source = leaf.operand_bindings[contract.input_operand]
    output = leaf.operand_bindings[contract.output_operand]
    if source.tensor == output.tensor or len(source.ranges) != 2 or len(output.ranges) != 2:
        return result
    source_buffer = ir.buffer(source.tensor)
    output_buffer = ir.buffer(output.tensor)
    if not _supported_buffer(source_buffer) or not _supported_buffer(output_buffer):
        return result
    if output_buffer.location != "sbuf":
        return result
    source_widths = _constant_widths(source)
    output_widths = _constant_widths(output)
    if source_widths is None or output_widths is None:
        return result
    expected_output = tuple(source_widths[index] for index in contract.permutation)
    if output_widths != expected_output:
        return result
    source_axis = _contiguous_batch_axis(source, source_buffer, loop.loop_var, source_widths)
    output_axis = _contiguous_batch_axis(output, output_buffer, loop.loop_var, output_widths)
    expected_output_axis = contract.permutation.index(source_axis) if source_axis is not None else None
    if source_axis is None or output_axis is None or output_axis != expected_output_axis:
        return result
    if not _parallel_loop(ir.tree.block(block_nid), loop.loop_var):
        return result
    result = (source_axis, output_axis)
    return result


def _valid_batch_contract(contract: PermutationContract, batching: BatchedPermutationContract) -> bool:
    """Return whether the expanded permutation embeds every logical axis once."""
    logical_rank = len(contract.permutation)
    expanded_rank = len(batching.permutation)
    valid = logical_rank == 2 and sorted(contract.permutation) == list(range(logical_rank))
    valid = valid and sorted(batching.permutation) == list(range(expanded_rank))
    valid = valid and len(batching.input_axes) == logical_rank
    valid = valid and len(set(batching.input_axes)) == logical_rank
    valid = valid and all(0 <= axis < expanded_rank for axis in batching.input_axes)
    valid = valid and 0 <= batching.batch_axis < expanded_rank
    valid = valid and batching.batch_axis not in batching.input_axes
    return valid


def _supported_buffer(buffer: Buffer) -> bool:
    """Return whether one allocation has a stable contiguous physical layout."""
    valid = len(buffer.shape) == 2
    if buffer.location == "shared_hbm":
        valid = valid and buffer.versions == 1
    else:
        valid = valid and buffer.list_len == 1 and buffer.shape[0] % PARTITION_DIM == 0
    return valid


def _constant_widths(region: BufferRegion) -> tuple[int, int] | None:
    """Return two constant region widths, or ``None``."""
    result: tuple[int, int] | None = None
    if len(region.ranges) == 2:
        widths = tuple(width for _lower, width in region.ranges)
        if all(isinstance(width, Const) and width.value > 0 for width in widths):
            first, second = widths
            assert isinstance(first, Const) and isinstance(second, Const)
            result = (first.value, second.value)
    return result


def _contiguous_batch_axis(region: BufferRegion, buffer: Buffer, loop_var: str, widths: tuple[int, int]) -> int | None:
    """Return the sole region axis advanced by one adjacent tile."""
    coefficients = tuple(to_affine(lower).get(loop_var, 0) for lower, _width in region.ranges)
    varying = [axis for axis, coefficient in enumerate(coefficients) if coefficient != 0]
    result: int | None = None
    if len(varying) == 1:
        axis = varying[0]
        expected = widths[axis]
        if axis == 0 and buffer.location != "shared_hbm":
            if widths[axis] % PARTITION_DIM != 0:
                return result
            expected = widths[axis] // PARTITION_DIM
        if coefficients[axis] == expected:
            result = axis
    return result


def _parallel_loop(block: BlockNode, loop_var: str) -> bool:
    """Return whether ``loop_var`` binds exactly one parallel block axis."""
    roles = [
        iter_var.role for iter_var, value in zip(block.iter_vars, block.iter_values) if loop_var in to_affine(value)
    ]
    return roles == [AxisRole.PARALLEL]


def _owning_block(ir: KernelIR, leaf_nid: int) -> int:
    """Return the nearest block owning ``leaf_nid``."""
    blocks = [nid for nid in reversed(ir.tree.ancestors(leaf_nid)) if isinstance(ir.tree.data(nid), BlockNode)]
    if not blocks:
        raise ValueError(f"ISA leaf {leaf_nid} has no owning block")
    return blocks[0]


def _apply_match(ir: KernelIR, match: _BatchMatch) -> None:
    """Materialize widened footprints and four-dimensional operand views."""
    loop = ir.tree.loop(match.loop_nid)
    leaf = ir.tree.isa(match.leaf_nid)
    source = leaf.operand_bindings[match.contract.input_operand]
    output = leaf.operand_bindings[match.contract.output_operand]
    widened_source = _widen_region(source, loop, match.source_axis)
    widened_output = _widen_region(output, loop, match.output_axis)
    bindings = dict(leaf.operand_bindings)
    bindings[match.contract.input_operand] = widened_source
    bindings[match.contract.output_operand] = widened_output
    kwargs = dict(leaf.kwargs)
    kwargs["axes"] = match.batching.permutation
    block = ir.tree.block(match.block_nid)
    substitutions: dict[str, Expr] = {loop.loop_var: Const(value=0)}
    ir.tree.graph.nodes[match.block_nid]["data"] = replace(
        block,
        iter_values=tuple(substitute(value, substitutions) for value in block.iter_values),
        reads=(widened_source,),
        writes=(widened_output,),
    )
    parent = ir.tree.parent(match.loop_nid)
    if parent is None:
        raise AssertionError(f"batch loop {match.loop_nid} has no parent")
    _replace_in_parent_children(ir.tree, parent, [match.loop_nid], [match.leaf_nid])
    ir.tree.graph.remove_node(match.loop_nid)
    invalidate_stale_software_pipelines(ir)

    source_buffer = ir.buffer(source.tensor)
    output_buffer = ir.buffer(output.tensor)
    expanded_rank = len(match.batching.permutation)
    source_view = _make_access_pattern(
        source, source_buffer, loop, match.batching.input_axes, match.batching.batch_axis, expanded_rank
    )
    output_axes = _output_axis_positions(match.contract, match.batching)
    output_batch_axis = match.batching.permutation.index(match.batching.batch_axis)
    output_view = _make_access_pattern(output, output_buffer, loop, output_axes, output_batch_axis, expanded_rank)
    access_patterns = {match.contract.input_operand: source_view, match.contract.output_operand: output_view}
    ir.tree.graph.nodes[match.leaf_nid]["data"] = replace(
        leaf, operand_bindings=bindings, kwargs=kwargs, access_patterns=access_patterns
    )


def _output_axis_positions(contract: PermutationContract, batching: BatchedPermutationContract) -> tuple[int, ...]:
    """Return expanded-output positions for logical output axes."""
    return tuple(batching.permutation.index(batching.input_axes[input_axis]) for input_axis in contract.permutation)


def _make_access_pattern(
    region: BufferRegion,
    buffer: Buffer,
    loop: ForNode,
    logical_positions: tuple[int, ...],
    batch_position: int,
    expanded_rank: int,
) -> AccessPattern:
    """Build the physical view that exposes ``loop`` as one batch dimension."""
    widths = _constant_widths(region)
    if widths is None:
        raise AssertionError(f"batch region {region.tensor} lost constant widths")
    dimensions = [(Const(value=1), Const(value=1)) for _ in range(expanded_rank)]
    axis_strides = _logical_axis_strides(buffer)
    for axis, position in enumerate(logical_positions):
        dimensions[position] = (Const(value=axis_strides[axis]), Const(value=widths[axis]))
    dimensions[batch_position] = (Const(value=_batch_stride(region, buffer, loop.loop_var)), Const(value=loop.extent))
    offset = _linear_offset(region, buffer, {loop.loop_var: Const(value=0)})
    return AccessPattern(pattern=tuple(dimensions), offset=offset)


def _logical_axis_strides(buffer: Buffer) -> tuple[int, int]:
    """Return flattened strides within one logical two-dimensional tile."""
    if buffer.location == "shared_hbm":
        strides = (buffer.shape[1], 1)
    else:
        physical = buffer.per_tile_physical_shape()
        strides = (physical[1] * physical[2], 1)
    return strides


def _batch_stride(region: BufferRegion, buffer: Buffer, loop_var: str) -> int:
    """Return the flattened element stride between adjacent loop iterations."""
    first_coefficient = to_affine(region.ranges[0][0]).get(loop_var, 0)
    second_coefficient = to_affine(region.ranges[1][0]).get(loop_var, 0)
    first_base_stride = buffer.shape[1]
    stride = first_coefficient * first_base_stride + second_coefficient
    if stride <= 0:
        raise AssertionError(f"{region.tensor}: batch stride must be positive, got {stride}")
    return stride


def _linear_offset(region: BufferRegion, buffer: Buffer, substitutions: dict[str, Expr]) -> Expr:
    """Return the flattened base offset for one logical two-dimensional region."""
    first = substitute(region.ranges[0][0], substitutions)
    second = substitute(region.ranges[1][0], substitutions)
    return Add(left=Mul(left=first, right=Const(value=buffer.shape[1])), right=second)


def _widen_region(region: BufferRegion, loop: ForNode, varying_axis: int) -> BufferRegion:
    """Remove the loop variable and widen its contiguous logical footprint."""
    substitutions: dict[str, Expr] = {loop.loop_var: Const(value=0)}
    ranges: list[tuple[Expr, Expr]] = []
    for axis, (lower, width) in enumerate(region.ranges):
        if not isinstance(width, Const):
            raise AssertionError(f"{region.tensor}: non-constant batch width {width!r}")
        widened = width.value * loop.extent if axis == varying_axis else width.value
        ranges.append((substitute(lower, substitutions), Const(value=widened)))
    return BufferRegion(tensor=region.tensor, ranges=tuple(ranges))


__all__ = ["BatchPermutation", "BatchPermutationOption"]
