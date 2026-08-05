"""Insert and cancel behavior-preserving pairs of logical transposes."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from nkigym.ir import KernelIR
from nkigym.ir.tree import Buffer, ISANode
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.transpose import NKITranspose
from nkigym.transforms._canonical_rewrite import (
    append_block,
    append_root_buffers,
    axis_extents,
    finalize_rewrite,
    fresh_name,
    is_canonical_block,
    remove_buffers,
    replace_input_binding,
    required_spec,
    single_leaf,
)
from nkigym.transforms._transpose_pattern import TransposeChain, match_transpose_chain
from nkigym.transforms._tree_ops import _replace_in_parent_children
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class InsertTransposePairOption(TransformOption):
    """Insert a transpose pair before one bound consumer operand."""

    consumer_nid: int
    operand: str
    source: str


@dataclass(frozen=True)
class CancelTransposePairOption(TransformOption):
    """Cancel the adjacent pair beginning at ``first_transpose_nid``."""

    first_transpose_nid: int


@dataclass(frozen=True)
class _InsertMatch:
    """A canonical SBUF edge eligible for pair insertion."""

    consumer_block: int
    consumer_leaf: int
    operand: str
    source: str
    source_axes: tuple[str, str]


@dataclass(frozen=True)
class _CancelMatch:
    """Two adjacent logical transposes and the readers of their output."""

    first: TransposeChain
    second: TransposeChain
    consumers: tuple[tuple[int, str], ...]


class InsertTransposePair(Transform[InsertTransposePairOption]):
    """Insert ``T(T(x))`` on one canonical producer-consumer edge."""

    def analyze(self, ir: KernelIR) -> list[InsertTransposePairOption]:
        """Return every canonical SBUF input edge eligible for insertion."""
        options: list[InsertTransposePairOption] = []
        for block_nid in ir.tree.children(ir.tree.root):
            leaf_nid = single_leaf(ir.tree, block_nid)
            if leaf_nid is not None:
                leaf = ir.tree.isa(leaf_nid)
                for operand, region in leaf.operand_bindings.items():
                    option = InsertTransposePairOption(consumer_nid=leaf_nid, operand=operand, source=region.tensor)
                    if _match_insert(ir, option) is not None:
                        options.append(option)
        return options

    def apply(self, ir: KernelIR, option: InsertTransposePairOption) -> KernelIR:
        """Recheck ``option`` and insert four concrete transpose/drain blocks."""
        match = _match_insert(ir, option)
        if match is None:
            raise TransformLegalityError(
                f"InsertTransposePair target {option.consumer_nid}:{option.operand} is not an eligible canonical SBUF edge"
            )
        new_ir = copy.deepcopy(ir)
        copied_match = _match_insert(new_ir, option)
        if copied_match is None:
            raise AssertionError("InsertTransposePair match disappeared after deepcopy")
        _apply_insert(new_ir, copied_match)
        finalize_rewrite(new_ir)
        return new_ir


class CancelTransposePair(Transform[CancelTransposePairOption]):
    """Remove two adjacent logical transposes."""

    def analyze(self, ir: KernelIR) -> list[CancelTransposePairOption]:
        """Return every cancellable adjacent transpose pair."""
        options: list[CancelTransposePairOption] = []
        for block_nid in ir.tree.children(ir.tree.root):
            option = CancelTransposePairOption(first_transpose_nid=block_nid)
            if _match_cancel(ir, option) is not None:
                options.append(option)
        return options

    def apply(self, ir: KernelIR, option: CancelTransposePairOption) -> KernelIR:
        """Recheck ``option`` and replace the pair's output uses with its input."""
        match = _match_cancel(ir, option)
        if match is None:
            raise TransformLegalityError(
                f"CancelTransposePair target {option.first_transpose_nid} is not a cancellable adjacent pair"
            )
        new_ir = copy.deepcopy(ir)
        copied_match = _match_cancel(new_ir, option)
        if copied_match is None:
            raise AssertionError("CancelTransposePair match disappeared after deepcopy")
        _apply_cancel(new_ir, copied_match)
        finalize_rewrite(new_ir)
        return new_ir


def _match_insert(ir: KernelIR, option: InsertTransposePairOption) -> _InsertMatch | None:
    """Return one legal pair-insertion edge."""
    result: _InsertMatch | None = None
    if option.consumer_nid in ir.tree.graph and isinstance(ir.tree.data(option.consumer_nid), ISANode):
        consumer = ir.tree.isa(option.consumer_nid)
        consumer_block = _root_owner(ir, option.consumer_nid)
        input_bound = option.operand in consumer.op_cls.INPUT_OPERANDS and option.operand in consumer.operand_bindings
        if consumer_block is not None and input_bound and is_canonical_block(ir, consumer_block):
            source = consumer.operand_bindings[option.operand].tensor
            axes = consumer.op_cls.OPERAND_AXES[option.operand]
            buffers = ir.all_buffers()
            axis_map = ir.tree.block(consumer_block).axis_map
            mapped = len(axes) == 2 and all(axis in axis_map for axis in axes)
            if source == option.source and source in buffers and mapped:
                source_buffer = buffers[source]
                source_axes = (axis_map[axes[0]], axis_map[axes[1]])
                extents = axis_extents(ir)
                shape_matches = source_buffer.shape == tuple(extents[axis] for axis in source_axes)
                tileable = all(extent >= 128 and extent % 128 == 0 for extent in source_buffer.shape)
                physical_dtype = source_buffer.physical_dtype() == source_buffer.dtype
                if (
                    source_buffer.location == "sbuf"
                    and len(source_buffer.shape) == 2
                    and shape_matches
                    and tileable
                    and physical_dtype
                ):
                    result = _InsertMatch(
                        consumer_block=consumer_block,
                        consumer_leaf=option.consumer_nid,
                        operand=option.operand,
                        source=source,
                        source_axes=source_axes,
                    )
    return result


def _root_owner(ir: KernelIR, leaf_nid: int) -> int | None:
    """Return the top-level block owning ``leaf_nid``."""
    result: int | None = None
    root_children = set(ir.tree.children(ir.tree.root))
    for ancestor in reversed(ir.tree.ancestors(leaf_nid)):
        if ancestor in root_children:
            result = ancestor
            break
    return result


def _apply_insert(ir: KernelIR, match: _InsertMatch) -> None:
    """Insert two transpose chains and rebind the selected consumer."""
    source = ir.buffer(match.source)
    first_psum_name = fresh_name(ir, f"{match.source}_t_psum")
    first_sbuf_name = fresh_name(ir, f"{match.source}_t")
    second_psum_name = fresh_name(ir, f"{match.source}_tt_psum")
    second_sbuf_name = fresh_name(ir, f"{match.source}_tt")
    reversed_shape = source.shape[::-1]
    append_root_buffers(
        ir,
        (
            Buffer(
                name=first_psum_name,
                shape=reversed_shape,
                dtype=source.dtype,
                location="psum",
                storage_dtype=NKITranspose.OUTPUT_STORAGE_DTYPE,
            ),
            Buffer(name=first_sbuf_name, shape=reversed_shape, dtype=source.dtype, location="sbuf"),
            Buffer(
                name=second_psum_name,
                shape=source.shape,
                dtype=source.dtype,
                location="psum",
                storage_dtype=NKITranspose.OUTPUT_STORAGE_DTYPE,
            ),
            Buffer(name=second_sbuf_name, shape=source.shape, dtype=source.dtype, location="sbuf"),
        ),
    )

    first_axis, second_axis = match.source_axes
    first_transpose = required_spec(
        ir, NKITranspose, {"data": match.source, "dst": first_psum_name}, {"P": first_axis, "F": second_axis}, {}
    )
    first_drain = required_spec(
        ir, NKITensorCopy, {"src": first_psum_name, "dst": first_sbuf_name}, {"P": second_axis, "F": first_axis}, {}
    )
    second_transpose = required_spec(
        ir, NKITranspose, {"data": first_sbuf_name, "dst": second_psum_name}, {"P": second_axis, "F": first_axis}, {}
    )
    second_drain = required_spec(
        ir, NKITensorCopy, {"src": second_psum_name, "dst": second_sbuf_name}, {"P": first_axis, "F": second_axis}, {}
    )
    inserted = [
        append_block(ir.tree, first_transpose),
        append_block(ir.tree, first_drain),
        append_block(ir.tree, second_transpose),
        append_block(ir.tree, second_drain),
    ]
    replace_input_binding(ir, match.consumer_leaf, match.operand, second_sbuf_name)
    _replace_in_parent_children(ir.tree, ir.tree.root, [match.consumer_block], [*inserted, match.consumer_block])


def _match_cancel(ir: KernelIR, option: CancelTransposePairOption) -> _CancelMatch | None:
    """Return one legal adjacent pair."""
    result: _CancelMatch | None = None
    root_children = ir.tree.children(ir.tree.root)
    if option.first_transpose_nid in root_children:
        index = root_children.index(option.first_transpose_nid)
        if index + 3 < len(root_children):
            first = match_transpose_chain(ir, root_children[index], root_children[index + 1])
            second = match_transpose_chain(ir, root_children[index + 2], root_children[index + 3])
            if first is not None and second is not None and second.source == first.output:
                consumers = _input_uses(ir, second.output, excluded={second.drain_leaf})
                exact_middle = set(ir.dependency.touches_by_tensor.get(first.output, ())) == {
                    first.drain_leaf,
                    second.transpose_leaf,
                }
                exact_output = set(ir.dependency.touches_by_tensor.get(second.output, ())) == {
                    second.drain_leaf,
                    *(leaf for leaf, _operand in consumers),
                }
                restored_shape = ir.buffer(second.output).shape == ir.buffer(first.source).shape
                source_stable = _source_is_stable(ir, first.source, first.transpose_leaf)
                if consumers and exact_middle and exact_output and restored_shape and source_stable:
                    result = _CancelMatch(first=first, second=second, consumers=consumers)
    return result


def _input_uses(ir: KernelIR, tensor: str, excluded: set[int]) -> tuple[tuple[int, str], ...]:
    """Return every input operand bound to ``tensor`` outside ``excluded``."""
    uses: list[tuple[int, str]] = []
    for leaf_nid in ir.tree.preorder():
        node = ir.tree.data(leaf_nid)
        if not isinstance(node, ISANode) or leaf_nid in excluded:
            continue
        for operand in node.op_cls.INPUT_OPERANDS:
            region = node.operand_bindings.get(operand)
            if region is not None and region.tensor == tensor:
                uses.append((leaf_nid, operand))
    return tuple(uses)


def _source_is_stable(ir: KernelIR, tensor: str, first_transpose_leaf: int) -> bool:
    """Return whether ``tensor`` has no writer after the pair reads it."""
    order = {nid: index for index, nid in enumerate(ir.tree.preorder())}
    stable = True
    for leaf_nid in ir.tree.preorder():
        node = ir.tree.data(leaf_nid)
        if not isinstance(node, ISANode):
            continue
        writes_tensor = any(
            region.tensor == tensor and operand not in node.op_cls.INPUT_OPERANDS
            for operand, region in node.operand_bindings.items()
        )
        if writes_tensor and order[leaf_nid] > order[first_transpose_leaf]:
            stable = False
            break
    return stable


def _apply_cancel(ir: KernelIR, match: _CancelMatch) -> None:
    """Remove a matched pair and reconnect its readers."""
    for leaf_nid, operand in match.consumers:
        replace_input_binding(ir, leaf_nid, operand, match.first.source)
    blocks = [
        match.first.transpose_block,
        match.first.drain_block,
        match.second.transpose_block,
        match.second.drain_block,
    ]
    _replace_in_parent_children(ir.tree, ir.tree.root, blocks, [])
    for block_nid in blocks:
        ir.tree.graph.remove_nodes_from({block_nid, *ir.tree.descendants(block_nid)})
    remove_buffers(ir, {match.first.psum, match.first.output, match.second.psum, match.second.output})


__all__ = ["CancelTransposePair", "CancelTransposePairOption", "InsertTransposePair", "InsertTransposePairOption"]
