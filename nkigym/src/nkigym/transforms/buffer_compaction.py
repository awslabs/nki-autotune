"""Per-buffer logical shape compaction and selected-region normalization."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from nkigym.codegen.compact import compact_buffer_shape
from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Expr
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import BlockNode, Buffer, ISANode, KernelTree
from nkigym.transforms._access_pattern import tensor_has_access_pattern
from nkigym.transforms._normalize import normalize_selected_tensor_regions, normalize_tensor_regions
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption

_RegionSnapshot = tuple[tuple[tuple[Expr, Expr], ...], ...]
_CompactionSnapshot = tuple[tuple[int, ...], _RegionSnapshot]


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
        tensors = tuple(
            name
            for name, buffer in ir.all_buffers().items()
            if buffer.location in ("sbuf", "psum") and not tensor_has_access_pattern(ir.tree, name)
        )
        changed = self._would_change_many(ir, tensors)
        return [BufferCompactionOption(tensor=tensor) for tensor in tensors if tensor in changed]

    def apply(self, ir: KernelIR, option: BufferCompactionOption) -> KernelIR:
        """Re-check legality, compact a deep copy, normalize its regions, and rebuild dependencies."""
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        compact_buffer_shape(new_ir.tree, option.tensor)
        normalize_tensor_regions(new_ir.tree, option.tensor)
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
        probe = copy.deepcopy(ir.tree)
        compacted = compact_buffer_shape(probe, option.tensor)
        if not _shape_only_shrinks(current, compacted):
            raise TransformLegalityError(
                f"BufferCompaction: {option.tensor} would expand from {current.shape} to {compacted.shape}"
            )
        if not _list_layout_compatible(compacted):
            raise TransformLegalityError(
                f"BufferCompaction: compacted tile count T={compacted.logical_tile_count()} for "
                f"{option.tensor} is incompatible with existing list_len={compacted.list_len}"
            )
        normalize_tensor_regions(probe, option.tensor)
        before = _compaction_snapshots(ir.tree, frozenset((option.tensor,)))[option.tensor]
        after = _compaction_snapshots(probe, frozenset((option.tensor,)))[option.tensor]
        if after == before:
            raise TransformLegalityError(f"BufferCompaction: {option.tensor} is already compact (no-op)")

    def _would_change_many(self, ir: KernelIR, tensors: tuple[str, ...]) -> set[str]:
        """Return tensors with legal shape or selected-region changes using one probe."""
        selected = frozenset(tensors)
        changed: set[str] = set()
        if selected:
            before = _compaction_snapshots(ir.tree, selected)
            probe = copy.deepcopy(ir.tree)
            eligible: set[str] = set()
            for tensor in tensors:
                current = ir.buffer(tensor)
                compacted = compact_buffer_shape(probe, tensor)
                if _shape_only_shrinks(current, compacted) and _list_layout_compatible(compacted):
                    eligible.add(tensor)
            normalize_selected_tensor_regions(probe, frozenset(eligible))
            after = _compaction_snapshots(probe, selected)
            changed = {tensor for tensor in eligible if after[tensor] != before[tensor]}
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


__all__ = ["BufferCompaction", "BufferCompactionOption"]
