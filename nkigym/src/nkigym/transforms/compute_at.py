"""``ComputeAt`` — sink a producer block under a consumer's loop.

See ``compute_at_legality.md`` (conditions 1-4, 5a, 6). Structural move is
the shared ``_move``; this face owns the output-block guard (condition 4:
the kernel's final store cannot be sunk). Ordering legality (condition 5a)
is delegated to ``_check_move_preserves_dependencies``: it simulates the move
and rejects any dependency edge that would point backward afterward.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from nkigym.codegen.compact import compact_shapes
from nkigym.ir import KernelIR
from nkigym.ir.buffer_placement import place_buffers
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms._code_motion import _check_move_preserves_dependencies, _move
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class ComputeAtOption(TransformOption):
    """Sink producer ``block_nid`` under ``target_loop_nid`` at ``index``."""

    block_nid: int
    target_loop_nid: int
    index: int


class ComputeAt(Transform):
    """Sink a producer block under a consumer's loop."""

    def apply(self, ir: KernelIR, option: ComputeAtOption) -> KernelIR:
        """Re-check legality, deep-copy, sink, re-derive geometry, return."""
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        _move(
            new_ir,
            block_nid=option.block_nid,
            target_loop_nid=option.target_loop_nid,
            index=option.index,
            is_reverse=False,
        )
        place_buffers(new_ir.tree)
        compact_shapes(new_ir.tree)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def analyze(self, ir: KernelIR) -> list[ComputeAtOption]:
        """Enumerate (producer, target loop, index) triples passing legality."""
        options: list[ComputeAtOption] = []
        leaf_blocks = [
            nid
            for nid in ir.tree.blocks()
            if nid != ir.tree.root
            and sum(1 for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)) == 1
        ]
        for block_nid in leaf_blocks:
            for target_nid in ir.tree.preorder():
                if not isinstance(ir.tree.data(target_nid), ForNode):
                    continue
                for index in self._legal_indices(ir, block_nid, target_nid):
                    opt = ComputeAtOption(block_nid=block_nid, target_loop_nid=target_nid, index=index)
                    try:
                        self._check_legality(ir, opt)
                    except TransformLegalityError:
                        continue
                    options.append(opt)
        return options

    def _legal_indices(self, ir: KernelIR, block_nid: int, target_nid: int) -> list[int]:
        """Slots in the insertion gap (lp, fc] among the target loop's children."""
        children = ir.tree.children(target_nid)
        producers = ir.dependency.producers(block_nid)
        consumers = ir.dependency.consumers(block_nid)
        lp = -1
        fc = len(children)
        for i, child in enumerate(children):
            sub = ir.tree.descendants(child) | {child}
            if sub & producers:
                lp = i
            if sub & consumers and i < fc:
                fc = i
        return list(range(lp + 1, fc + 1))

    def _check_legality(self, ir: KernelIR, option: ComputeAtOption) -> None:
        """Conditions 1-4, then move-sim ordering (5a). (6 enumerated by _legal_indices.)"""
        if option.target_loop_nid not in ir.tree.graph:
            raise TransformLegalityError(f"target_loop_nid={option.target_loop_nid} not in tree")
        if not isinstance(ir.tree.data(option.target_loop_nid), ForNode):
            raise TransformLegalityError(
                f"ComputeAt requires target_loop_nid to be a ForNode; got "
                f"{type(ir.tree.data(option.target_loop_nid)).__name__}"
            )
        if option.block_nid not in ir.tree.graph:
            raise TransformLegalityError(f"block_nid={option.block_nid} not in tree")
        if option.target_loop_nid in ir.tree.descendants(option.block_nid):
            raise TransformLegalityError(
                f"target_loop_nid={option.target_loop_nid} is a descendant of moved block "
                f"{option.block_nid} (cannot sink under its own loop)"
            )
        leaf = next(
            (ir.tree.data(d) for d in ir.tree.descendants(option.block_nid) if isinstance(ir.tree.data(d), ISANode)),
            None,
        )
        if leaf is not None:
            for region in leaf.operand_bindings.values():
                if region.tensor == ir.return_name:
                    raise TransformLegalityError(
                        f"ComputeAt cannot sink the output block (writes return {ir.return_name})"
                    )
        _check_move_preserves_dependencies(ir, option.block_nid, option.target_loop_nid, option.index, is_reverse=False)


__all__ = ["ComputeAt", "ComputeAtOption"]
