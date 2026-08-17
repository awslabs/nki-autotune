"""Public per-buffer declaration placement at a lifetime-safe LCA scope."""

from __future__ import annotations

from dataclasses import dataclass

from nkigym.ir import KernelIR
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import BlockNode, KernelTree
from nkigym.search.buffer_placement import buffer_placement_targets, place_buffer
from nkigym.search.serialization import inherit_analysis_result
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption, copy_for_rewrite
from nkigym.transforms.helper.access_pattern import tensor_has_access_pattern


@dataclass(frozen=True)
class BufferPlacementOption(TransformOption):
    """Move the declaration of one on-chip ``tensor`` to its lifetime-safe LCA.

    Attributes:
        tensor: Buffer name whose declaration should move.
    """

    tensor: str


class BufferPlacement(Transform[BufferPlacementOption]):
    """Move only one selected on-chip declaration to its lifetime-safe LCA scope."""

    def analyze(self, ir: KernelIR) -> list[BufferPlacementOption]:
        """Offer on-chip buffers whose declaration would move."""
        tensors = tuple(
            name
            for name, buffer in ir.all_buffers().items()
            if buffer.location in ("sbuf", "psum") and not tensor_has_access_pattern(ir.tree, name)
        )
        changed = self._would_change_many(ir.tree, tensors)
        return [BufferPlacementOption(tensor=tensor) for tensor in tensors if tensor in changed]

    def apply(self, ir: KernelIR, option: BufferPlacementOption) -> KernelIR:
        """Re-check legality, move one declaration on a deep copy, and rebuild dependencies."""
        self._check_legality(ir, option)
        new_ir = copy_for_rewrite(ir)
        place_buffer(new_ir.tree, option.tensor)
        new_ir.dependency = Dependency(new_ir.tree)
        inherit_analysis_result(ir, new_ir, "code-motion")
        return new_ir

    def _check_legality(self, ir: KernelIR, option: BufferPlacementOption) -> None:
        """Reject unknown, HBM, explicit-pattern, and no-op placement choices."""
        buffers = ir.all_buffers()
        if option.tensor not in buffers:
            raise TransformLegalityError(f"BufferPlacement: no buffer named {option.tensor!r}")
        if buffers[option.tensor].location == "shared_hbm":
            raise TransformLegalityError(f"BufferPlacement: {option.tensor} is shared_hbm (must remain at root)")
        if tensor_has_access_pattern(ir.tree, option.tensor):
            raise TransformLegalityError(f"BufferPlacement: {option.tensor} participates in an explicit access pattern")
        if option.tensor not in self._would_change_many(ir.tree, (option.tensor,)):
            raise TransformLegalityError(f"BufferPlacement: {option.tensor} is already at its target scope (no-op)")

    def _would_change_many(self, tree: KernelTree, tensors: tuple[str, ...]) -> set[str]:
        """Return declarations whose computed lifetime-safe block differs."""
        selected = frozenset(tensors)
        changed: set[str] = set()
        if selected:
            before = _declaration_blocks(tree, selected)
            after = buffer_placement_targets(tree, tensors)
            changed = {tensor for tensor in tensors if after[tensor] != before[tensor]}
        return changed


def _declaration_blocks(tree: KernelTree, tensors: frozenset[str]) -> dict[str, int]:
    """Return the unique owning block for each selected declaration."""
    declarations: dict[str, int] = {}
    for nid in tree.blocks():
        block = tree.data(nid)
        assert isinstance(block, BlockNode)
        for buffer in block.alloc_buffers:
            if buffer.name in tensors:
                if buffer.name in declarations:
                    raise AssertionError(f"buffer {buffer.name!r} is declared by multiple blocks")
                declarations[buffer.name] = nid
    missing = tensors - declarations.keys()
    if missing:
        raise KeyError(f"buffers declared by no block: {sorted(missing)}")
    return declarations


__all__ = ["BufferPlacement", "BufferPlacementOption"]
