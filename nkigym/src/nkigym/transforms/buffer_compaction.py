"""Per-buffer logical shape compaction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace

from nkigym.codegen.compact import _compact_one, compact_buffer_shapes
from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Expr, to_affine
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.search.buffer_placement import _offsets_consistently, layout_satisfies_output_alignment
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption, copy_for_rewrite
from nkigym.transforms.buffer_region_normalization import (
    _regions_requiring_normalization,
    access_patterns_fit_buffer,
    rebase_access_patterns,
)
from nkigym.transforms.helper.access_pattern import tensor_has_access_pattern

_LoopFact = tuple[int, str, int]
_AccessGeometry = tuple[tuple[tuple[Expr, Expr], ...], tuple[tuple[str, int], ...]]
_CompactionGeometry = tuple[tuple[int, ...], str, tuple[str, ...], tuple[_AccessGeometry, ...]]


@dataclass
class _CompactionFacts:
    """Tree-wide facts shared by buffer-shape compaction analysis."""

    leaf_loops: dict[int, tuple[_LoopFact, ...]]
    allocation_loops: dict[str, frozenset[int]]
    regions: dict[str, list[tuple[int, BufferRegion]]]
    buffers: dict[str, Buffer]


@dataclass(frozen=True)
class BufferCompactionOption(TransformOption):
    """Compact one logical allocation axis of ``tensor``.

    Attributes:
        tensor: Buffer name to compact.
        axis: Logical allocation axis to shrink.
    """

    tensor: str
    axis: int


class BufferCompaction(Transform[BufferCompactionOption]):
    """Shrink one buffer's logical allocation shape."""

    def analyze(self, ir: KernelIR) -> list[BufferCompactionOption]:
        """Offer one option per independently shrinkable allocation axis."""
        buffers = ir.all_buffers()
        tensors = tuple(name for name, buffer in buffers.items() if buffer.location in ("sbuf", "psum"))
        compacted = self._compacted_many(ir, tensors, buffers)
        return [
            BufferCompactionOption(tensor=tensor, axis=axis)
            for tensor in tensors
            if tensor in compacted
            for axis in range(len(buffers[tensor].shape))
            if (candidate := _axis_candidate(buffers[tensor], compacted[tensor], axis)) is not None
            and layout_satisfies_output_alignment(ir.tree, candidate)
            and (
                not tensor_has_access_pattern(ir.tree, tensor)
                or access_patterns_fit_buffer(ir.tree, tensor, candidate, prior=buffers[tensor])
            )
        ]

    def apply(self, ir: KernelIR, option: BufferCompactionOption) -> KernelIR:
        """Re-check legality and shrink one declaration axis."""
        compacted = self._check_legality(ir, option)
        new_ir = copy_for_rewrite(ir)
        current = new_ir.buffer(option.tensor)
        _replace_buffer(new_ir.tree, compacted)
        if tensor_has_access_pattern(new_ir.tree, option.tensor):
            rebase_access_patterns(new_ir.tree, option.tensor, current, compacted)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def _check_legality(self, ir: KernelIR, option: BufferCompactionOption) -> Buffer:
        """Reject unknown, HBM, explicit-pattern, expanding, incompatible, and no-op choices."""
        buffers = ir.all_buffers()
        if option.tensor not in buffers:
            raise TransformLegalityError(f"BufferCompaction: no buffer named {option.tensor!r}")
        current = buffers[option.tensor]
        if current.location == "shared_hbm":
            raise TransformLegalityError(f"BufferCompaction: {option.tensor} is shared_hbm (nothing to compact)")
        patterned = tensor_has_access_pattern(ir.tree, option.tensor)
        selected = frozenset((option.tensor,))
        if _regions_requiring_normalization(ir.tree, selected):
            raise TransformLegalityError(
                f"BufferCompaction: {option.tensor} regions must be normalized before compaction"
            )
        probe = copy_for_rewrite(ir).tree
        fully_compacted = compact_buffer_shapes(probe, selected)[option.tensor]
        compacted = _axis_candidate(current, fully_compacted, option.axis)
        if compacted is None:
            raise TransformLegalityError(
                f"BufferCompaction: axis {option.axis} of {option.tensor} cannot shrink from {current.shape}"
            )
        if not _list_layout_compatible(compacted):
            raise TransformLegalityError(
                f"BufferCompaction: compacted tile count T={compacted.logical_tile_count()} for "
                f"{option.tensor} is incompatible with existing list_len={compacted.list_len}"
            )
        if not layout_satisfies_output_alignment(ir.tree, compacted):
            raise TransformLegalityError(
                f"BufferCompaction: compacted {option.tensor} would violate producer output alignment"
            )
        if patterned and not access_patterns_fit_buffer(ir.tree, option.tensor, compacted, prior=current):
            raise TransformLegalityError(
                f"BufferCompaction: compacted {option.tensor} would exceed its explicit physical view"
            )
        return compacted

    def _compacted_many(self, ir: KernelIR, tensors: tuple[str, ...], buffers: dict[str, Buffer]) -> dict[str, Buffer]:
        """Return fully compacted declarations used to enumerate axis actions."""
        selected = frozenset(tensors)
        changed: dict[str, Buffer] = {}
        if selected:
            normalized = selected - _regions_requiring_normalization(ir.tree, selected)
            facts = _compaction_facts(ir.tree, normalized)
            compacted = _compacted_buffers(normalized, facts)
            changed = {
                tensor: compacted[tensor]
                for tensor in normalized
                if _shape_only_shrinks(buffers[tensor], compacted[tensor])
                and compacted[tensor].shape != buffers[tensor].shape
            }
        return changed


def _axis_candidate(current: Buffer, compacted: Buffer, axis: int) -> Buffer | None:
    """Return a declaration with exactly one independently compacted axis."""
    candidate = None
    if 0 <= axis < len(current.shape) and len(current.shape) == len(compacted.shape):
        target = compacted.shape[axis]
        if target < current.shape[axis]:
            shape = list(current.shape)
            shape[axis] = target
            proposed = replace(current, shape=tuple(shape))
            if _list_layout_compatible(proposed):
                candidate = proposed
    return candidate


def _replace_buffer(tree: KernelTree, compacted: Buffer) -> None:
    """Replace one buffer declaration without modifying access regions."""
    found = False
    for block_nid in tree.blocks():
        block = tree.block(block_nid)
        buffers = tuple(compacted if buffer.name == compacted.name else buffer for buffer in block.alloc_buffers)
        if buffers != block.alloc_buffers:
            if found:
                raise AssertionError(f"buffer {compacted.name!r} is declared by multiple blocks")
            tree.graph.nodes[block_nid]["data"] = replace(block, alloc_buffers=buffers)
            found = True
    if not found:
        raise KeyError(f"buffer {compacted.name!r} is declared by no block")


def _shape_only_shrinks(current: Buffer, compacted: Buffer) -> bool:
    """Return whether compaction preserves rank and never enlarges an axis."""
    return len(current.shape) == len(compacted.shape) and all(
        compacted_extent <= current_extent
        for current_extent, compacted_extent in zip(current.shape, compacted.shape, strict=True)
    )


def _list_layout_compatible(buffer: Buffer) -> bool:
    """Return whether the unchanged list length divides the compacted tile count."""
    logical_tiles = buffer.logical_tile_count()
    return buffer.list_len >= 1 and logical_tiles % buffer.list_len == 0


def _compaction_facts(tree: KernelTree, tensors: frozenset[str]) -> _CompactionFacts:
    """Index loop paths, selected regions, and buffers."""
    leaf_loops: dict[int, tuple[_LoopFact, ...]] = {}
    allocation_loops: dict[str, frozenset[int]] = {}
    regions: dict[str, list[tuple[int, BufferRegion]]] = defaultdict(list)
    buffers: dict[str, Buffer] = {}
    pending: list[tuple[int, tuple[_LoopFact, ...]]] = [(tree.root, ())]
    while pending:
        nid, enclosing_loops = pending.pop()
        data = tree.data(nid)
        child_loops = enclosing_loops
        if isinstance(data, BlockNode):
            for buffer in data.alloc_buffers:
                if buffer.name in buffers:
                    raise AssertionError(f"buffer {buffer.name!r} is declared by multiple blocks")
                buffers[buffer.name] = buffer
                allocation_loops[buffer.name] = frozenset(nid for nid, _loop_var, _extent in enclosing_loops)
        elif isinstance(data, ForNode):
            child_loops = (*enclosing_loops, (nid, data.loop_var, data.extent))
        elif isinstance(data, ISANode):
            leaf_loops[nid] = enclosing_loops
            for region in data.operand_bindings.values():
                if region.tensor in tensors:
                    regions[region.tensor].append((nid, region))
        pending.extend((child, child_loops) for child in reversed(tree.children(nid)))
    missing = tensors - buffers.keys()
    if missing:
        raise KeyError(f"buffers declared by no block: {sorted(missing)}")
    return _CompactionFacts(
        leaf_loops=leaf_loops, allocation_loops=allocation_loops, regions=dict(regions), buffers=buffers
    )


def _compacted_buffers(tensors: frozenset[str], facts: _CompactionFacts) -> dict[str, Buffer]:
    """Return read-only compacted buffer values using indexed loop paths."""
    compacted: dict[str, Buffer] = {}
    shapes: dict[_CompactionGeometry, tuple[int, ...]] = {}
    for tensor in tensors:
        pairs = facts.regions.get(tensor, [])
        anchors: list[str] = []
        if pairs:
            loop_sets = [{nid for nid, _name, _extent in facts.leaf_loops[leaf]} for leaf, _region in pairs]
            common = set.intersection(*loop_sets) & facts.allocation_loops[tensor]
            first_loops = facts.leaf_loops[pairs[0][0]]
            regions = [region for _leaf, region in pairs]
            for nid, loop_var, _extent in first_loops:
                if nid not in common:
                    continue
                if not _offsets_consistently(loop_var, regions):
                    break
                anchors.append(loop_var)
        leaf_extents = {
            leaf: {loop_var: extent for _nid, loop_var, extent in facts.leaf_loops[leaf]} for leaf, _region in pairs
        }
        buffer = facts.buffers[tensor]
        geometry: _CompactionGeometry = (
            buffer.shape,
            buffer.location,
            tuple(sorted(anchors)),
            tuple((region.ranges, tuple(sorted(leaf_extents[leaf].items()))) for leaf, region in pairs),
        )
        shape = shapes.get(geometry)
        if shape is None:
            shape = _compact_one(buffer, set(anchors), pairs, leaf_extents).shape
            shapes[geometry] = shape
        compacted[tensor] = replace(buffer, shape=shape)
    return compacted


__all__ = ["BufferCompaction", "BufferCompactionOption"]
