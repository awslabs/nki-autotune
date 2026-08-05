"""``BufferCompaction`` transform: per-buffer placement, shape shrink, and normalization.

Materializes, for ONE buffer, the compaction that used to run anonymously in the
``CodeMotion`` / ``RFactor`` tail (whole-tree ``place_buffers`` + ``compact_shapes``)
plus the render-time region rebase. Descends the buffer's declaration to its LCA
scope, shrinks its logical shape to the access bounding box, and re-normalizes its
access regions against that new extent. The extent-fit rule drops instance-selecting
outer loops and leaves intra-instance loops intact. Mirrors :class:`BufferLayout`'s
single-``tensor`` surface.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from nkigym.codegen.compact import compact_shapes, place_and_compact_buffer
from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Expr
from nkigym.ir.buffer_placement import place_buffers
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import BlockNode, ISANode, KernelTree
from nkigym.transforms._normalize import normalize_selected_tensor_regions, normalize_tensor_regions
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption

_RegionSnapshot = tuple[tuple[tuple[Expr, Expr], ...], ...]
_CompactionSnapshot = tuple[tuple[int, ...], int, _RegionSnapshot]


@dataclass(frozen=True)
class BufferCompactionOption(TransformOption):
    """Compact ``tensor`` (place at LCA, shrink shape, and normalize regions).

    Attributes:
        tensor: buffer name to compact.
    """

    tensor: str


class BufferCompaction(Transform[BufferCompactionOption]):
    """Materialize one buffer's placement, compact shape, and local region frame."""

    def analyze(self, ir: KernelIR) -> list[BufferCompactionOption]:
        """Offer every sbuf/psum buffer whose compacted form differs from its current one."""
        tensors = tuple(name for name, buf in ir.all_buffers().items() if buf.location in ("sbuf", "psum"))
        changed = self._would_change_many(ir, tensors)
        return [BufferCompactionOption(tensor=tensor) for tensor in tensors if tensor in changed]

    def apply(self, ir: KernelIR, option: BufferCompactionOption) -> KernelIR:
        """Re-check legality, compact a deep copy, normalize its regions, and rebuild deps."""
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        place_and_compact_buffer(new_ir.tree, option.tensor)
        normalize_tensor_regions(new_ir.tree, option.tensor)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def _check_legality(self, ir: KernelIR, option: BufferCompactionOption) -> None:
        """Loud rejects: unknown tensor, shared_hbm, or a no-op compaction."""
        buffers = ir.all_buffers()
        if option.tensor not in buffers:
            raise TransformLegalityError(f"BufferCompaction: no buffer named {option.tensor!r}")
        if buffers[option.tensor].location == "shared_hbm":
            raise TransformLegalityError(f"BufferCompaction: {option.tensor} is shared_hbm (nothing to compact)")
        if not self._would_change(ir, option.tensor):
            raise TransformLegalityError(f"BufferCompaction: {option.tensor} is already compact (no-op)")

    def _would_change(self, ir: KernelIR, tensor: str) -> bool:
        """Whether compaction would alter the tensor's declaration, shape, or regions."""
        probe = copy.deepcopy(ir.tree)
        before = _compaction_snapshots(ir.tree, frozenset((tensor,)))[tensor]
        place_and_compact_buffer(probe, tensor)
        normalize_tensor_regions(probe, tensor)
        after = _compaction_snapshots(probe, frozenset((tensor,)))[tensor]
        return after != before

    def _would_change_many(self, ir: KernelIR, tensors: tuple[str, ...]) -> set[str]:
        """Return changed tensors using one equivalent placement and normalization probe."""
        selected = frozenset(tensors)
        changed: set[str] = set()
        if selected:
            before = _compaction_snapshots(ir.tree, selected)
            probe = copy.deepcopy(ir.tree)
            place_buffers(probe)
            compact_shapes(probe)
            normalize_selected_tensor_regions(probe, selected)
            after = _compaction_snapshots(probe, selected)
            changed = {tensor for tensor in tensors if after[tensor] != before[tensor]}
        return changed


def _compaction_snapshots(tree: KernelTree, tensors: frozenset[str]) -> dict[str, _CompactionSnapshot]:
    """Collect declaration, logical shape, and ISA regions for selected tensors."""
    shapes: dict[str, tuple[int, ...]] = {}
    declarations: dict[str, int] = {}
    regions: dict[str, list[tuple[tuple[Expr, Expr], ...]]] = {tensor: [] for tensor in tensors}
    for nid in tree.blocks():
        block = tree.data(nid)
        assert isinstance(block, BlockNode)
        for buffer in block.alloc_buffers:
            if buffer.name in tensors:
                shapes[buffer.name] = buffer.shape
                declarations[buffer.name] = nid
    for nid in tree.preorder():
        data = tree.data(nid)
        if isinstance(data, ISANode):
            for region in data.operand_bindings.values():
                if region.tensor in tensors:
                    regions[region.tensor].append(region.ranges)
    missing = tensors - shapes.keys()
    if missing:
        raise KeyError(f"buffers declared by no block: {sorted(missing)}")
    return {tensor: (shapes[tensor], declarations[tensor], tuple(regions[tensor])) for tensor in tensors}


__all__ = ["BufferCompaction", "BufferCompactionOption"]
