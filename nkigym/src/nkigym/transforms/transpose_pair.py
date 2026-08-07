"""Insert and cancel behavior-preserving pairs of logical transposes."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from nkigym.ir import KernelIR
from nkigym.ir.tree import Buffer, ISANode
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.transpose import NKITranspose
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption
from nkigym.transforms.helper.canonical_rewrite import (
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
from nkigym.transforms.helper.transpose_pattern import TransposeChain, match_transpose_chain
from nkigym.transforms.helper.tree_ops import _replace_in_parent_children


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


TransposePairOption = InsertTransposePairOption | CancelTransposePairOption


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
    """Two adjacent transpose executions and the readers of their output."""

    first: _TransposeExecution
    second: _TransposeExecution
    consumers: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class _TransposeExecution:
    """One logical or DMA transpose in a top-level dataflow chain."""

    blocks: tuple[int, ...]
    removable_buffers: tuple[str, ...]
    input_leaf: int
    output_leaf: int
    input_operand: str
    source: str
    output: str


class TransposePair(Transform[TransposePairOption]):
    """Insert or cancel one behavior-preserving logical transpose pair."""

    def analyze(self, ir: KernelIR) -> list[TransposePairOption]:
        """Return every legal pair insertion and cancellation."""
        options: list[TransposePairOption] = []
        pair_edges = _identity_pair_edges(ir)
        for block_nid in ir.tree.children(ir.tree.root):
            leaf_nid = single_leaf(ir.tree, block_nid)
            if leaf_nid is not None:
                leaf = ir.tree.isa(leaf_nid)
                for operand, region in leaf.operand_bindings.items():
                    option = InsertTransposePairOption(consumer_nid=leaf_nid, operand=operand, source=region.tensor)
                    if _match_insert(ir, option, pair_edges) is not None:
                        options.append(option)
        for block_nid in ir.tree.children(ir.tree.root):
            option = CancelTransposePairOption(first_transpose_nid=block_nid)
            if _match_cancel(ir, option) is not None:
                options.append(option)
        return options

    def apply(self, ir: KernelIR, option: TransposePairOption) -> KernelIR:
        """Recheck and apply one insertion or cancellation."""
        new_ir = copy.deepcopy(ir)
        if isinstance(option, InsertTransposePairOption):
            match = _match_insert(ir, option)
            copied_match = _match_insert(new_ir, option)
            if match is None or copied_match is None:
                raise TransformLegalityError(
                    f"illegal transpose-pair insertion at {option.consumer_nid}:{option.operand}"
                )
            _apply_insert(new_ir, copied_match)
        else:
            match = _match_cancel(ir, option)
            copied_match = _match_cancel(new_ir, option)
            if match is None or copied_match is None:
                raise TransformLegalityError(f"illegal transpose-pair cancellation at {option.first_transpose_nid}")
            _apply_cancel(new_ir, copied_match)
        finalize_rewrite(new_ir)
        return new_ir


def _match_insert(
    ir: KernelIR, option: InsertTransposePairOption, pair_edges: set[tuple[int, str, str]] | None = None
) -> _InsertMatch | None:
    """Return one legal pair-insertion edge."""
    result: _InsertMatch | None = None
    edge = (option.consumer_nid, option.operand, option.source)
    redundant = edge in (pair_edges if pair_edges is not None else _identity_pair_edges(ir))
    if (
        not redundant
        and option.consumer_nid in ir.tree.graph
        and isinstance(ir.tree.data(option.consumer_nid), ISANode)
    ):
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


def _identity_pair_edges(ir: KernelIR) -> set[tuple[int, str, str]]:
    """Return edges on which another transpose pair would be redundant."""
    pair_edges: set[tuple[int, str, str]] = set()
    for block_nid in ir.tree.children(ir.tree.root):
        match = _match_cancel(ir, CancelTransposePairOption(first_transpose_nid=block_nid))
        if match is not None:
            pair_edges.add((match.first.input_leaf, match.first.input_operand, match.first.source))
            pair_edges.add((match.second.input_leaf, match.second.input_operand, match.second.source))
            pair_edges.update((leaf, operand, match.second.output) for leaf, operand in match.consumers)
    return pair_edges


def _match_transpose_execution(ir: KernelIR, root_children: list[int], index: int) -> _TransposeExecution | None:
    """Return one logical or DMA transpose beginning at ``index``."""
    result: _TransposeExecution | None = None
    logical = (
        match_transpose_chain(ir, root_children[index], root_children[index + 1])
        if index + 1 < len(root_children)
        else None
    )
    if logical is not None:
        result = _TransposeExecution(
            blocks=(logical.transpose_block, logical.drain_block),
            removable_buffers=(logical.psum, logical.output),
            input_leaf=logical.transpose_leaf,
            output_leaf=logical.drain_leaf,
            input_operand="data",
            source=logical.source,
            output=logical.output,
        )
    else:
        result = _match_dma_transpose_execution(ir, root_children[index])
    return result


def _match_dma_transpose_execution(ir: KernelIR, block_nid: int) -> _TransposeExecution | None:
    """Return one canonical DMA transpose over complete SBUF/HBM buffers."""
    result: _TransposeExecution | None = None
    leaf_nid = single_leaf(ir.tree, block_nid)
    if leaf_nid is not None and is_canonical_block(ir, block_nid):
        leaf = ir.tree.isa(leaf_nid)
        if leaf.op_cls is NKIDMATranspose:
            source = leaf.operand_bindings["src"].tensor
            output = leaf.operand_bindings["dst"].tensor
            buffers = ir.all_buffers()
            if source in buffers and output in buffers:
                source_buffer = buffers[source]
                output_buffer = buffers[output]
                valid = (
                    source_buffer.location in {"shared_hbm", "sbuf"}
                    and output_buffer.location == "sbuf"
                    and len(source_buffer.shape) == 2
                    and output_buffer.shape == source_buffer.shape[::-1]
                    and source_buffer.dtype == output_buffer.dtype
                    and source_buffer.physical_dtype() == source_buffer.dtype
                    and output_buffer.physical_dtype() == source_buffer.dtype
                )
                if valid:
                    result = _TransposeExecution(
                        blocks=(block_nid,),
                        removable_buffers=(output,),
                        input_leaf=leaf_nid,
                        output_leaf=leaf_nid,
                        input_operand="src",
                        source=source,
                        output=output,
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
        first = _match_transpose_execution(ir, root_children, index)
        if first is not None:
            second_index = index + len(first.blocks)
            second = (
                _match_transpose_execution(ir, root_children, second_index)
                if second_index < len(root_children)
                else None
            )
            if first is not None and second is not None and second.source == first.output:
                consumers = _input_uses(ir, second.output, excluded={second.output_leaf})
                exact_middle = set(ir.dependency.touches_by_tensor.get(first.output, ())) == {
                    first.output_leaf,
                    second.input_leaf,
                }
                exact_output = set(ir.dependency.touches_by_tensor.get(second.output, ())) == {
                    second.output_leaf,
                    *(leaf for leaf, _operand in consumers),
                }
                restored_shape = ir.buffer(second.output).shape == ir.buffer(first.source).shape
                source_stable = _source_is_stable(ir, first.source, first.input_leaf)
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
    blocks = [*match.first.blocks, *match.second.blocks]
    removed_buffers = set(match.first.removable_buffers) | set(match.second.removable_buffers)
    preserved_buffers = tuple(
        buffer
        for block_nid in blocks
        for buffer in ir.tree.block(block_nid).alloc_buffers
        if buffer.name not in removed_buffers
    )
    remove_buffers(ir, removed_buffers)
    if preserved_buffers:
        append_root_buffers(ir, preserved_buffers)
    _replace_in_parent_children(ir.tree, ir.tree.root, blocks, [])
    for block_nid in blocks:
        ir.tree.graph.remove_nodes_from({block_nid, *ir.tree.descendants(block_nid)})


__all__ = ["CancelTransposePairOption", "InsertTransposePairOption", "TransposePair", "TransposePairOption"]
