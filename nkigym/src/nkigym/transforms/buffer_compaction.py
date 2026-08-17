"""Per-buffer logical shape compaction and selected-region normalization."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace

from nkigym.codegen.compact import _compact_one, compact_buffer_shapes
from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Expr, to_affine
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.search.buffer_placement import _offsets_consistently
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption, copy_for_rewrite
from nkigym.transforms.helper.access_pattern import tensor_has_access_pattern
from nkigym.transforms.helper.normalize import (
    _dim_from_loopvar,
    _iter_value_loopvars,
    _recompute_region,
    normalize_selected_tensor_regions,
)

_RegionSnapshot = tuple[tuple[tuple[Expr, Expr], ...], ...]
_CompactionSnapshot = tuple[tuple[int, ...], _RegionSnapshot]
_LoopFact = tuple[int, str, int]
_AccessGeometry = tuple[tuple[tuple[Expr, Expr], ...], tuple[tuple[str, int], ...]]
_CompactionGeometry = tuple[tuple[int, ...], str, tuple[str, ...], tuple[_AccessGeometry, ...]]
_NormalizationKey = tuple[
    tuple[tuple[Expr, Expr], ...],
    tuple[str, ...] | None,
    tuple[tuple[str, str], ...],
    tuple[tuple[str, tuple[tuple[str, int], ...]], ...],
    tuple[int, ...],
    str,
]


@dataclass
class _BlockFacts:
    """Block-local payloads and enclosing loop facts from one tree walk."""

    block: BlockNode
    enclosing_loops: tuple[_LoopFact, ...]
    local_loops: list[tuple[str, int]]
    isas: list[ISANode]


@dataclass
class _CompactionFacts:
    """Tree-wide facts shared by shape and region compaction analysis."""

    blocks: dict[int, _BlockFacts]
    leaf_loops: dict[int, tuple[_LoopFact, ...]]
    regions: dict[str, list[tuple[int, BufferRegion]]]
    buffers: dict[str, Buffer]


@dataclass(frozen=True)
class BufferCompactionOption(TransformOption):
    """Compact the logical shape and local region frame of ``tensor``.

    Attributes:
        tensor: Buffer name to compact.
    """

    tensor: str


class BufferCompaction(Transform[BufferCompactionOption]):
    """Shrink one buffer's logical shape and normalize only its regions."""

    def analyze(self, ir: KernelIR) -> list[BufferCompactionOption]:
        """Offer on-chip buffers with legal shape or selected-region changes."""
        buffers = ir.all_buffers()
        tensors = tuple(
            name
            for name, buffer in buffers.items()
            if buffer.location in ("sbuf", "psum") and not tensor_has_access_pattern(ir.tree, name)
        )
        changed = self._would_change_many(ir, tensors, buffers)
        return [BufferCompactionOption(tensor=tensor) for tensor in tensors if tensor in changed]

    def apply(self, ir: KernelIR, option: BufferCompactionOption) -> KernelIR:
        """Re-check legality, compact a deep copy, normalize its regions, and rebuild dependencies."""
        self._check_legality(ir, option)
        new_ir = copy_for_rewrite(ir)
        selected = frozenset((option.tensor,))
        compact_buffer_shapes(new_ir.tree, selected)
        normalize_selected_tensor_regions(new_ir.tree, selected)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def _check_legality(self, ir: KernelIR, option: BufferCompactionOption) -> None:
        """Reject unknown, HBM, explicit-pattern, expanding, incompatible, and no-op choices."""
        buffers = ir.all_buffers()
        if option.tensor not in buffers:
            raise TransformLegalityError(f"BufferCompaction: no buffer named {option.tensor!r}")
        current = buffers[option.tensor]
        if current.location == "shared_hbm":
            raise TransformLegalityError(f"BufferCompaction: {option.tensor} is shared_hbm (nothing to compact)")
        if tensor_has_access_pattern(ir.tree, option.tensor):
            raise TransformLegalityError(
                f"BufferCompaction: {option.tensor} participates in an explicit access pattern"
            )
        probe = copy_for_rewrite(ir).tree
        selected = frozenset((option.tensor,))
        compacted = compact_buffer_shapes(probe, selected)[option.tensor]
        if not _shape_only_shrinks(current, compacted):
            raise TransformLegalityError(
                f"BufferCompaction: {option.tensor} would expand from {current.shape} to {compacted.shape}"
            )
        if not _list_layout_compatible(compacted):
            raise TransformLegalityError(
                f"BufferCompaction: compacted tile count T={compacted.logical_tile_count()} for "
                f"{option.tensor} is incompatible with existing list_len={compacted.list_len}"
            )
        normalize_selected_tensor_regions(probe, selected)
        before = _compaction_snapshots(ir.tree, selected)[option.tensor]
        after = _compaction_snapshots(probe, selected)[option.tensor]
        if after == before:
            raise TransformLegalityError(f"BufferCompaction: {option.tensor} is already compact (no-op)")

    def _would_change_many(self, ir: KernelIR, tensors: tuple[str, ...], buffers: dict[str, Buffer]) -> set[str]:
        """Return tensors with legal shape or selected-region changes using one index."""
        selected = frozenset(tensors)
        changed: set[str] = set()
        if selected:
            facts = _compaction_facts(ir.tree, selected)
            compacted = _compacted_buffers(selected, facts)
            eligible = {
                tensor
                for tensor in tensors
                if _shape_only_shrinks(buffers[tensor], compacted[tensor])
                and _list_layout_compatible(compacted[tensor])
            }
            changed = {tensor for tensor in eligible if compacted[tensor].shape != buffers[tensor].shape}
            changed.update(_regions_requiring_normalization(frozenset(eligible - changed), facts))
        return changed


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


def _compaction_snapshots(tree: KernelTree, tensors: frozenset[str]) -> dict[str, _CompactionSnapshot]:
    """Collect logical shape plus block and ISA regions for selected tensors."""
    shapes: dict[str, tuple[int, ...]] = {}
    regions: dict[str, list[tuple[tuple[Expr, Expr], ...]]] = {tensor: [] for tensor in tensors}
    for nid in tree.blocks():
        block = tree.data(nid)
        assert isinstance(block, BlockNode)
        for buffer in block.alloc_buffers:
            if buffer.name in tensors:
                shapes[buffer.name] = buffer.shape
        for region in (*block.reads, *block.writes):
            if region.tensor in tensors:
                regions[region.tensor].append(region.ranges)
    for nid in tree.preorder():
        data = tree.data(nid)
        if isinstance(data, ISANode):
            for region in data.operand_bindings.values():
                if region.tensor in tensors:
                    regions[region.tensor].append(region.ranges)
    missing = tensors - shapes.keys()
    if missing:
        raise KeyError(f"buffers declared by no block: {sorted(missing)}")
    return {tensor: (shapes[tensor], tuple(regions[tensor])) for tensor in tensors}


def _compaction_facts(tree: KernelTree, tensors: frozenset[str]) -> _CompactionFacts:
    """Index block-local payloads, loop paths, selected regions, and buffers."""
    blocks: dict[int, _BlockFacts] = {}
    leaf_loops: dict[int, tuple[_LoopFact, ...]] = {}
    regions: dict[str, list[tuple[int, BufferRegion]]] = defaultdict(list)
    buffers: dict[str, Buffer] = {}
    pending: list[tuple[int, tuple[_LoopFact, ...], int | None]] = [(tree.root, (), None)]
    while pending:
        nid, enclosing_loops, owner = pending.pop()
        data = tree.data(nid)
        child_loops = enclosing_loops
        child_owner = owner
        if isinstance(data, BlockNode):
            child_owner = nid
            blocks[nid] = _BlockFacts(block=data, enclosing_loops=enclosing_loops, local_loops=[], isas=[])
            for buffer in data.alloc_buffers:
                if buffer.name in buffers:
                    raise AssertionError(f"buffer {buffer.name!r} is declared by multiple blocks")
                buffers[buffer.name] = buffer
        elif isinstance(data, ForNode):
            child_loops = (*enclosing_loops, (nid, data.loop_var, data.extent))
            if owner is not None:
                blocks[owner].local_loops.append((data.loop_var, data.extent))
        elif isinstance(data, ISANode):
            if owner is None:
                raise AssertionError(f"ISA node {nid} has no owning block")
            blocks[owner].isas.append(data)
            leaf_loops[nid] = enclosing_loops
            for region in data.operand_bindings.values():
                if region.tensor in tensors:
                    regions[region.tensor].append((nid, region))
        pending.extend((child, child_loops, child_owner) for child in reversed(tree.children(nid)))
    missing = tensors - buffers.keys()
    if missing:
        raise KeyError(f"buffers declared by no block: {sorted(missing)}")
    return _CompactionFacts(blocks=blocks, leaf_loops=leaf_loops, regions=dict(regions), buffers=buffers)


def _compacted_buffers(tensors: frozenset[str], facts: _CompactionFacts) -> dict[str, Buffer]:
    """Return read-only compacted buffer values using indexed loop paths."""
    compacted: dict[str, Buffer] = {}
    shapes: dict[_CompactionGeometry, tuple[int, ...]] = {}
    for tensor in tensors:
        pairs = facts.regions.get(tensor, [])
        anchors: list[str] = []
        if pairs:
            loop_sets = [{nid for nid, _name, _extent in facts.leaf_loops[leaf]} for leaf, _region in pairs]
            common = set.intersection(*loop_sets)
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


def _block_dim_loops(facts: _BlockFacts) -> dict[str, list[tuple[str, int]]]:
    """Return dimension loop chains from indexed enclosing and local loops."""
    loop_dims: dict[str, str] = {}
    for iter_var, value in zip(facts.block.iter_vars, facts.block.iter_values):
        for loop_var in to_affine(value):
            if loop_var is not None:
                loop_dims[loop_var] = iter_var.axis
    for loop_var, _extent in facts.local_loops:
        loop_dims.setdefault(loop_var, _dim_from_loopvar(loop_var))
    bound = _iter_value_loopvars(facts.block)
    result: dict[str, list[tuple[str, int]]] = {}
    for _nid, loop_var, extent in facts.enclosing_loops:
        if loop_var in bound:
            dimension = loop_dims.get(loop_var, _dim_from_loopvar(loop_var))
            result.setdefault(dimension, []).append((loop_var, extent))
    for loop_var, extent in facts.local_loops:
        dimension = loop_dims.get(loop_var, _dim_from_loopvar(loop_var))
        result.setdefault(dimension, []).append((loop_var, extent))
    return result


def _regions_requiring_normalization(tensors: frozenset[str], facts: _CompactionFacts) -> set[str]:
    """Return selected tensors whose current regions differ from normalized regions."""
    changed: set[str] = set()
    normalized_ranges: dict[_NormalizationKey, tuple[tuple[Expr, Expr], ...]] = {}
    for block_facts in facts.blocks.values():
        block = block_facts.block
        dim_loops = _block_dim_loops(block_facts)
        axis_map_key = tuple(sorted(block.axis_map.items()))
        dim_loops_key = tuple(sorted((dimension, tuple(loops)) for dimension, loops in dim_loops.items()))
        tensor_axes: dict[str, tuple[str, ...]] = {}
        for isa in block_facts.isas:
            for slot, region in isa.operand_bindings.items():
                tensor_axes[region.tensor] = isa.op_cls.OPERAND_AXES[slot]

        def differs(region: BufferRegion, axes: tuple[str, ...] | None) -> bool:
            """Return whether one region differs from its cached normalized geometry."""
            buffer = facts.buffers[region.tensor]
            key: _NormalizationKey = (region.ranges, axes, axis_map_key, dim_loops_key, buffer.shape, buffer.location)
            expected = normalized_ranges.get(key)
            if expected is None:
                selected_axes = {} if axes is None else {region.tensor: axes}
                expected = _recompute_region(region, selected_axes, block.axis_map, dim_loops, facts.buffers).ranges
                normalized_ranges[key] = expected
            return expected != region.ranges

        for region in (*block.reads, *block.writes):
            if (
                region.tensor in tensors
                and region.tensor not in changed
                and differs(region, tensor_axes.get(region.tensor))
            ):
                changed.add(region.tensor)
        for isa in block_facts.isas:
            for slot, region in isa.operand_bindings.items():
                if (
                    region.tensor in tensors
                    and region.tensor not in changed
                    and differs(region, isa.op_cls.OPERAND_AXES[slot])
                ):
                    changed.add(region.tensor)
    return changed


__all__ = ["BufferCompaction", "BufferCompactionOption"]
