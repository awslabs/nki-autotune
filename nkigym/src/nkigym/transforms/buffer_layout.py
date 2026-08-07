"""``BufferLayout`` transform — re-factorize a buffer's logical tile axis into a
``list_len x tiles_per_list`` form whose product is the logical tile count.

A pure field-set on :attr:`Buffer.list_len`: it changes neither regions nor
tree structure, only allocation granularity. Pipeline versions remain a separate
multiplier within every list entry. Mirrors :class:`SoftwarePipeline`, which sets
the sibling :attr:`Buffer.versions`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import BlockNode
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption
from nkigym.transforms.helper.access_pattern import tensor_has_access_pattern


@dataclass(frozen=True)
class BufferLayoutOption(TransformOption):
    """Relayout ``tensor`` into ``list_len`` allocations of equal logical tile count.

    Attributes:
        tensor: buffer name to relayout.
        list_len: target list length b. 1 = list-of-one (packed), T = full split.
    """

    tensor: str
    list_len: int


class BufferLayout(Transform[BufferLayoutOption]):
    """Re-factorize one buffer's tile axis; sets :attr:`Buffer.list_len`."""

    def analyze(self, ir: KernelIR) -> list[BufferLayoutOption]:
        """Return every logical-tile divisor relayout for each on-chip buffer."""
        options: list[BufferLayoutOption] = []
        for name, buf in ir.all_buffers().items():
            if buf.location not in ("sbuf", "psum"):
                continue
            if tensor_has_access_pattern(ir.tree, name):
                continue
            logical_tiles = buf.logical_tile_count()
            for b in range(1, logical_tiles + 1):
                if logical_tiles % b == 0 and b != buf.list_len:
                    options.append(BufferLayoutOption(tensor=name, list_len=b))
        return options

    def apply(self, ir: KernelIR, option: BufferLayoutOption) -> KernelIR:
        """Re-check legality, deep-copy, set ``list_len``, rebuild the dependency sidecar."""
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        self._set_list_len(new_ir, option.tensor, option.list_len)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def _check_legality(self, ir: KernelIR, option: BufferLayoutOption) -> None:
        """Structural renderability guards only (never resource capacity)."""
        buffers = ir.all_buffers()
        if option.tensor not in buffers:
            raise TransformLegalityError(f"BufferLayout: no buffer named {option.tensor!r}")
        buf = buffers[option.tensor]
        if buf.location == "shared_hbm":
            raise TransformLegalityError(f"BufferLayout: {option.tensor} is shared_hbm (no tile axis)")
        if tensor_has_access_pattern(ir.tree, option.tensor):
            raise TransformLegalityError(f"BufferLayout: {option.tensor} participates in an explicit access pattern")
        logical_tiles = buf.logical_tile_count()
        if option.list_len < 1 or logical_tiles % option.list_len != 0:
            raise TransformLegalityError(
                f"BufferLayout: list_len {option.list_len} must be a positive divisor of T={logical_tiles}"
            )
        if option.list_len == buf.list_len:
            raise TransformLegalityError(f"BufferLayout: {option.tensor} already has list_len={option.list_len}")

    def _set_list_len(self, ir: KernelIR, name: str, list_len: int) -> None:
        """Replace the owning block's alloc entry for ``name`` with a list_len-updated copy."""
        for nid in ir.tree.blocks():
            block = ir.tree.data(nid)
            assert isinstance(block, BlockNode)
            new_allocs = tuple(replace(b, list_len=list_len) if b.name == name else b for b in block.alloc_buffers)
            if new_allocs != block.alloc_buffers:
                ir.tree.graph.nodes[nid]["data"] = replace(block, alloc_buffers=new_allocs)


__all__ = ["BufferLayout", "BufferLayoutOption"]
