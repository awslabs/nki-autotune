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

from nkigym.codegen.compact import place_and_compact_buffer
from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Expr
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import BlockNode, ISANode
from nkigym.transforms._normalize import normalize_tensor_regions
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


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
        options: list[BufferCompactionOption] = []
        for name, buf in ir.all_buffers().items():
            if buf.location in ("sbuf", "psum") and self._would_change(ir, name):
                options.append(BufferCompactionOption(tensor=name))
        return options

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
        probe = copy.deepcopy(ir)
        before_shape = probe.buffer(tensor).shape
        before_regions = _regions_snapshot(probe, tensor)
        before_decl = _decl_block_of(probe, tensor)
        place_and_compact_buffer(probe.tree, tensor)
        normalize_tensor_regions(probe.tree, tensor)
        after_decl = _decl_block_of(probe, tensor)
        changed = (
            probe.buffer(tensor).shape != before_shape
            or _regions_snapshot(probe, tensor) != before_regions
            or after_decl != before_decl
        )
        return changed


def _decl_block_of(ir: KernelIR, tensor: str) -> int:
    """Block nid that declares ``tensor`` in its alloc_buffers."""
    for nid in ir.tree.blocks():
        block = ir.tree.data(nid)
        assert isinstance(block, BlockNode)
        if any(buf.name == tensor for buf in block.alloc_buffers):
            return nid
    raise KeyError(f"{tensor} declared by no block")


def _regions_snapshot(ir: KernelIR, tensor: str) -> tuple[tuple[tuple[Expr, Expr], ...], ...]:
    """Immutable snapshot of every region naming ``tensor`` (for change detection)."""
    out: list[tuple[tuple[Expr, Expr], ...]] = []
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if isinstance(data, ISANode):
            for region in data.operand_bindings.values():
                if region.tensor == tensor:
                    out.append(region.ranges)
    return tuple(out)


__all__ = ["BufferCompaction", "BufferCompactionOption"]
