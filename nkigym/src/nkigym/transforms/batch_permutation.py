"""Batch independent permutation instructions through a preserved view axis."""

from __future__ import annotations

from dataclasses import dataclass, replace

from nkigym.ir import AccessPattern, Add, Const, Expr, KernelIR, Mod, Mul, Var, substitute
from nkigym.ir.arith.analyzer import Analyzer
from nkigym.ir.arith.expr import affine_coefficient, expr_variables
from nkigym.ir.dependency import Dependency
from nkigym.ir.program_sharding import PROGRAM_SHARDS_ANNOTATION, configured_program_shards
from nkigym.ir.tree import PARTITION_DIM, BlockNode, Buffer, BufferRegion, ForNode, ISANode
from nkigym.ops.base import AxisRole, BatchedPermutationContract, PermutationContract
from nkigym.transforms.base import (
    Transform,
    TransformLegalityError,
    TransformOption,
    copy_for_rewrite,
    intersects_software_pipeline,
    software_pipeline_overlap_nodes,
)
from nkigym.transforms.helper.tree_ops import _replace_in_parent_children


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
    expands_existing: bool = False


class BatchPermutation(Transform[BatchPermutationOption]):
    """Replace a loop of independent permutations with one batched permutation."""

    def analyze(self, ir: KernelIR) -> list[BatchPermutationOption]:
        """Return every directly tensorizable permutation loop."""
        options: list[BatchPermutationOption] = []
        buffers = ir.all_buffers()
        overlap_nodes = software_pipeline_overlap_nodes(ir)
        sharded_loops = configured_program_shards(ir)
        for nid in ir.tree.preorder():
            if (
                (nid not in sharded_loops or len(sharded_loops) > 1)
                and isinstance(ir.tree.data(nid), ForNode)
                and _match_loop(ir, nid, buffers, overlap_nodes) is not None
            ):
                options.append(BatchPermutationOption(loop_nid=nid))
        return options

    def apply(self, ir: KernelIR, option: BatchPermutationOption) -> KernelIR:
        """Recheck, copy, and absorb one loop into access-pattern views."""
        match = _match_option(ir, option)
        if match is None:
            raise TransformLegalityError(
                f"BatchPermutation loop {option.loop_nid} is not an eligible permutation batch"
            )
        new_ir = copy_for_rewrite(ir)
        copied_match = _match_option(new_ir, option)
        if copied_match is None:
            raise AssertionError("BatchPermutation match disappeared after deepcopy")
        _apply_match(new_ir, copied_match)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir


def _match_option(ir: KernelIR, option: BatchPermutationOption) -> _BatchMatch | None:
    """Resolve ``option`` without accepting an unknown node id."""
    result: _BatchMatch | None = None
    shards = configured_program_shards(ir)
    if option.loop_nid in ir.tree.graph and (option.loop_nid not in shards or len(shards) > 1):
        result = _match_loop(ir, option.loop_nid, ir.all_buffers())
    return result


def _match_loop(
    ir: KernelIR, loop_nid: int, buffers: dict[str, Buffer], overlap_nodes: frozenset[int] | None = None
) -> _BatchMatch | None:
    """Return the contract and geometry for one eligible loop."""
    result: _BatchMatch | None = None
    if intersects_software_pipeline(ir, (loop_nid,), overlap_nodes):
        return result
    node = ir.tree.data(loop_nid)
    children = ir.tree.children(loop_nid) if isinstance(node, ForNode) else []
    if isinstance(node, ForNode) and node.extent > 1 and len(children) == 1:
        leaf_nid = children[0]
        leaf = ir.tree.data(leaf_nid)
        if isinstance(leaf, ISANode):
            contract = leaf.op_cls.algebraic_contract(leaf.kwargs)
            if isinstance(contract, PermutationContract) and contract.batching is not None:
                block_nid = _owning_block(ir, leaf_nid)
                programs = configured_program_shards(ir).get(loop_nid, 1)
                axes = _match_geometry(ir, block_nid, node, leaf, contract, buffers, programs)
                expands_existing = bool(leaf.access_patterns)
                if axes is not None and (
                    not expands_existing or _valid_existing_batch(leaf, contract, axes, buffers, programs)
                ):
                    result = _BatchMatch(
                        block_nid=block_nid,
                        loop_nid=loop_nid,
                        leaf_nid=leaf_nid,
                        contract=contract,
                        batching=contract.batching,
                        source_axis=axes[0],
                        output_axis=axes[1],
                        expands_existing=expands_existing,
                    )
    return result


def _match_geometry(
    ir: KernelIR,
    block_nid: int,
    loop: ForNode,
    leaf: ISANode,
    contract: PermutationContract,
    buffers: dict[str, Buffer],
    programs: int,
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
    source_buffer = buffers[source.tensor]
    output_buffer = buffers[output.tensor]
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
    if loop.extent % programs:
        return result
    local_extent = loop.extent // programs
    source_axis = _contiguous_batch_axis(source, source_buffer, loop.loop_var, source_widths, local_extent)
    output_axis = _contiguous_batch_axis(output, output_buffer, loop.loop_var, output_widths, local_extent)
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


def _valid_existing_batch(
    leaf: ISANode, contract: PermutationContract, axes: tuple[int, int], buffers: dict[str, Buffer], programs: int
) -> bool:
    """Return whether a direct-HBM permutation view can absorb one sharded loop."""
    batching = contract.batching
    if batching is None or programs <= 1 or leaf.kwargs.get("axes") != batching.permutation:
        return False
    if set(leaf.access_patterns) != {contract.input_operand, contract.output_operand}:
        return False
    source = leaf.operand_bindings[contract.input_operand]
    output = leaf.operand_bindings[contract.output_operand]
    if buffers[source.tensor].location != "shared_hbm" or buffers[output.tensor].location != "sbuf":
        return False
    source_position = batching.input_axes[axes[0]]
    output_position = _output_axis_positions(contract, batching)[axes[1]]
    source_dimension = leaf.access_patterns[contract.input_operand].pattern[source_position]
    output_dimension = leaf.access_patterns[contract.output_operand].pattern[output_position]
    source_width = source.ranges[axes[0]][1]
    output_width = output.ranges[axes[1]][1]
    return source_dimension[1] == source_width and output_dimension[1] == output_width


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


def _contiguous_batch_axis(
    region: BufferRegion, buffer: Buffer, loop_var: str, widths: tuple[int, int], local_extent: int
) -> int | None:
    """Return the sole region axis advanced by one adjacent tile."""
    coefficients = tuple(
        affine_coefficient(_local_batch_expr(lower, loop_var, local_extent), loop_var)
        for lower, _width in region.ranges
    )
    if any(coefficient is None for coefficient in coefficients):
        return None
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
        iter_var.role
        for iter_var, value in zip(block.iter_vars, block.iter_values)
        if loop_var in expr_variables(value)
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
    if match.expands_existing:
        _expand_existing_batch(ir, match)
        return
    loop = ir.tree.loop(match.loop_nid)
    programs = configured_program_shards(ir).get(match.loop_nid, 1)
    local_extent = loop.extent // programs
    origin = _batch_origin(local_extent, programs)
    leaf = ir.tree.isa(match.leaf_nid)
    source = leaf.operand_bindings[match.contract.input_operand]
    output = leaf.operand_bindings[match.contract.output_operand]
    widened_source = _widen_region(source, loop, match.source_axis, local_extent, origin)
    widened_output = _widen_region(output, loop, match.output_axis, local_extent, origin)
    bindings = dict(leaf.operand_bindings)
    bindings[match.contract.input_operand] = widened_source
    bindings[match.contract.output_operand] = widened_output
    kwargs = dict(leaf.kwargs)
    kwargs["axes"] = match.batching.permutation
    block = ir.tree.block(match.block_nid)
    substitutions: dict[str, Expr] = {loop.loop_var: origin}
    ir.tree.graph.nodes[match.block_nid]["data"] = replace(
        block,
        iter_values=tuple(substitute(value, substitutions) for value in block.iter_values),
        reads=(widened_source,),
        writes=(widened_output,),
    )
    parent = ir.tree.parent(match.loop_nid)
    if parent is None:
        raise AssertionError(f"batch loop {match.loop_nid} has no parent")
    if programs > 1:
        _remove_program_shard(ir, match.loop_nid)
    _replace_in_parent_children(ir.tree, parent, [match.loop_nid], [match.leaf_nid])
    ir.tree.graph.remove_node(match.loop_nid)

    source_buffer = ir.buffer(source.tensor)
    output_buffer = ir.buffer(output.tensor)
    expanded_rank = len(match.batching.permutation)
    source_view = _make_access_pattern(
        source,
        source_buffer,
        loop,
        match.batching.input_axes,
        match.batching.batch_axis,
        expanded_rank,
        local_extent,
        origin,
    )
    output_axes = _output_axis_positions(match.contract, match.batching)
    output_batch_axis = match.batching.permutation.index(match.batching.batch_axis)
    output_view = _make_access_pattern(
        output, output_buffer, loop, output_axes, output_batch_axis, expanded_rank, local_extent, origin
    )
    access_patterns = {match.contract.input_operand: source_view, match.contract.output_operand: output_view}
    ir.tree.graph.nodes[match.leaf_nid]["data"] = replace(
        leaf, operand_bindings=bindings, kwargs=kwargs, access_patterns=access_patterns
    )


def _expand_existing_batch(ir: KernelIR, match: _BatchMatch) -> None:
    """Absorb one program-local outer loop into an existing permutation view."""
    loop = ir.tree.loop(match.loop_nid)
    programs = configured_program_shards(ir)[match.loop_nid]
    local_extent = loop.extent // programs
    global_origin = _batch_origin(local_extent, programs)
    leaf = ir.tree.isa(match.leaf_nid)
    source = leaf.operand_bindings[match.contract.input_operand]
    output = leaf.operand_bindings[match.contract.output_operand]
    widened_source = _widen_region(source, loop, match.source_axis, local_extent, global_origin)
    widened_output = _widen_region(output, loop, match.output_axis, local_extent, Const(value=0))
    batching = match.batching
    source_position = batching.input_axes[match.source_axis]
    output_position = _output_axis_positions(match.contract, batching)[match.output_axis]
    access_patterns = dict(leaf.access_patterns)
    access_patterns[match.contract.input_operand] = _widen_access_pattern(
        access_patterns[match.contract.input_operand], source_position, local_extent, loop.loop_var, global_origin
    )
    access_patterns[match.contract.output_operand] = _widen_access_pattern(
        access_patterns[match.contract.output_operand], output_position, local_extent, loop.loop_var, Const(value=0)
    )
    bindings = dict(leaf.operand_bindings)
    bindings[match.contract.input_operand] = widened_source
    bindings[match.contract.output_operand] = widened_output
    ir.tree.graph.nodes[match.leaf_nid]["data"] = replace(
        leaf, operand_bindings=bindings, access_patterns=access_patterns
    )
    block = ir.tree.block(match.block_nid)
    substitutions = {loop.loop_var: global_origin}
    ir.tree.graph.nodes[match.block_nid]["data"] = replace(
        block,
        iter_values=tuple(substitute(value, substitutions) for value in block.iter_values),
        reads=(widened_source,),
        writes=(widened_output,),
    )
    parent = ir.tree.parent(match.loop_nid)
    if parent is None:
        raise AssertionError(f"batch loop {match.loop_nid} has no parent")
    _remove_program_shard(ir, match.loop_nid)
    _replace_in_parent_children(ir.tree, parent, [match.loop_nid], [match.leaf_nid])
    ir.tree.graph.remove_node(match.loop_nid)


def _widen_access_pattern(
    pattern: AccessPattern, position: int, factor: int, loop_var: str, origin: Expr
) -> AccessPattern:
    """Widen one contiguous view dimension and replace its materialized loop origin."""
    dimensions = list(pattern.pattern)
    stride, width = dimensions[position]
    if not isinstance(width, Const):
        raise AssertionError(f"batch dimension {position} has non-constant width {width!r}")
    dimensions[position] = (stride, Const(value=width.value * factor))
    return AccessPattern(
        pattern=tuple(dimensions), offset=Analyzer().simplify(substitute(pattern.offset, {loop_var: origin}))
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
    local_extent: int,
    origin: Expr,
) -> AccessPattern:
    """Build the physical view that exposes ``loop`` as one batch dimension."""
    widths = _constant_widths(region)
    if widths is None:
        raise AssertionError(f"batch region {region.tensor} lost constant widths")
    dimensions = [(Const(value=1), Const(value=1)) for _ in range(expanded_rank)]
    axis_strides = _logical_axis_strides(buffer)
    for axis, position in enumerate(logical_positions):
        dimensions[position] = (Const(value=axis_strides[axis]), Const(value=widths[axis]))
    dimensions[batch_position] = (
        Const(value=_batch_stride(region, buffer, loop.loop_var, local_extent)),
        Const(value=local_extent),
    )
    offset = _linear_offset(region, buffer, {loop.loop_var: origin})
    return AccessPattern(pattern=tuple(dimensions), offset=offset)


def _logical_axis_strides(buffer: Buffer) -> tuple[int, int]:
    """Return flattened strides within one logical two-dimensional tile."""
    if buffer.location == "shared_hbm":
        strides = (buffer.shape[1], 1)
    else:
        physical = buffer.per_tile_physical_shape()
        strides = (physical[1] * physical[2], 1)
    return strides


def _batch_stride(region: BufferRegion, buffer: Buffer, loop_var: str, local_extent: int) -> int:
    """Return the flattened element stride between adjacent loop iterations."""
    first_coefficient = affine_coefficient(_local_batch_expr(region.ranges[0][0], loop_var, local_extent), loop_var)
    second_coefficient = affine_coefficient(_local_batch_expr(region.ranges[1][0], loop_var, local_extent), loop_var)
    if first_coefficient is None or second_coefficient is None:
        raise AssertionError(f"{region.tensor}: batch stride is not affine in {loop_var}")
    first_base_stride = buffer.shape[1]
    stride = first_coefficient * first_base_stride + second_coefficient
    if stride <= 0:
        raise AssertionError(f"{region.tensor}: batch stride must be positive, got {stride}")
    return stride


def _linear_offset(region: BufferRegion, buffer: Buffer, substitutions: dict[str, Expr]) -> Expr:
    """Return the flattened base offset for one logical two-dimensional region."""
    first = substitute(region.ranges[0][0], substitutions)
    second = substitute(region.ranges[1][0], substitutions)
    return Analyzer().simplify(Add(left=Mul(left=first, right=Const(value=buffer.shape[1])), right=second))


def _widen_region(
    region: BufferRegion, loop: ForNode, varying_axis: int, local_extent: int, origin: Expr
) -> BufferRegion:
    """Remove the loop variable and widen its contiguous logical footprint."""
    substitutions: dict[str, Expr] = {loop.loop_var: origin}
    ranges: list[tuple[Expr, Expr]] = []
    for axis, (lower, width) in enumerate(region.ranges):
        if not isinstance(width, Const):
            raise AssertionError(f"{region.tensor}: non-constant batch width {width!r}")
        widened = width.value * local_extent if axis == varying_axis else width.value
        ranges.append((Analyzer().simplify(substitute(lower, substitutions)), Const(value=widened)))
    return BufferRegion(tensor=region.tensor, ranges=tuple(ranges))


def _local_batch_expr(expr: Expr, loop_var: str, local_extent: int) -> Expr:
    """Replace one program-local modulo coordinate by its local loop value."""
    if expr == Mod(left=Var(name=loop_var), right=Const(value=local_extent)):
        return Var(name=loop_var)
    if isinstance(expr, (Const, Var)):
        return expr
    return replace(
        expr,
        left=_local_batch_expr(expr.left, loop_var, local_extent),
        right=_local_batch_expr(expr.right, loop_var, local_extent),
    )


def _batch_origin(local_extent: int, programs: int) -> Expr:
    """Return the first global iteration owned by the current program."""
    return Const(value=0) if programs == 1 else Mul(left=Var(name="nl.program_id(0)"), right=Const(value=local_extent))


def _remove_program_shard(ir: KernelIR, loop_nid: int) -> None:
    """Remove the materialized shard whose local iterations were batched."""
    root = ir.tree.block(ir.tree.root)
    annotations = dict(root.annotations)
    shards = dict(configured_program_shards(ir))
    del shards[loop_nid]
    annotations[PROGRAM_SHARDS_ANNOTATION] = shards
    ir.tree.graph.nodes[ir.tree.root]["data"] = replace(root, annotations=annotations)


__all__ = ["BatchPermutation", "BatchPermutationOption"]
