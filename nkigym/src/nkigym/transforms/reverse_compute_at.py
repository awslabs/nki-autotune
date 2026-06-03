"""``ReverseComputeAt`` — lift a consumer block under a producer's loop.

See ``compute_at_legality.md`` (conditions 1-3, 5b, 6). Structural move is
the shared ``_move``; this face owns the producer-direction legality.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from nkigym.codegen.compact import compact_shapes
from nkigym.ir import KernelIR
from nkigym.ir.buffer_placement import place_buffers
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms._code_motion import _move
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class ReverseComputeAtOption(TransformOption):
    """Lift consumer ``block_nid`` under ``target_loop_nid`` at ``index``."""

    block_nid: int
    target_loop_nid: int
    index: int


class ReverseComputeAt(Transform):
    """Lift a consumer block under a producer's loop."""

    def apply(self, ir: KernelIR, option: ReverseComputeAtOption) -> KernelIR:
        """Re-check legality, deep-copy, lift, re-derive geometry, return."""
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        _move(
            new_ir,
            block_nid=option.block_nid,
            target_loop_nid=option.target_loop_nid,
            index=option.index,
            is_reverse=True,
        )
        place_buffers(new_ir.tree)
        compact_shapes(new_ir.tree)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def analyze(self, ir: KernelIR) -> list[ReverseComputeAtOption]:
        """Enumerate (consumer, target loop, index) triples passing legality."""
        options: list[ReverseComputeAtOption] = []
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
                    opt = ReverseComputeAtOption(block_nid=block_nid, target_loop_nid=target_nid, index=index)
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

    def _check_legality(self, ir: KernelIR, option: ReverseComputeAtOption) -> None:
        """Conditions 1-3, 5b. (6 enumerated by _legal_indices.)"""
        if option.target_loop_nid not in ir.tree.graph:
            raise TransformLegalityError(f"target_loop_nid={option.target_loop_nid} not in tree")
        if not isinstance(ir.tree.data(option.target_loop_nid), ForNode):
            raise TransformLegalityError(
                f"ReverseComputeAt requires target_loop_nid to be a ForNode; got "
                f"{type(ir.tree.data(option.target_loop_nid)).__name__}"
            )
        if option.block_nid not in ir.tree.graph:
            raise TransformLegalityError(f"block_nid={option.block_nid} not in tree")
        if option.target_loop_nid in ir.tree.descendants(option.block_nid):
            raise TransformLegalityError(
                f"target_loop_nid={option.target_loop_nid} is a descendant of moved block "
                f"{option.block_nid} (cannot lift under its own loop)"
            )
        self._check_producers_visited(ir, option)

    def _check_producers_visited(self, ir: KernelIR, option: ReverseComputeAtOption) -> None:
        """Condition 5b: every producer is under the target, encloses the target, OR is an earlier root-sibling."""
        target_root = self._root_sibling_of(ir, option.target_loop_nid)
        root_order = ir.tree.children(ir.tree.root)
        target_index = root_order.index(target_root)
        target_descendants = ir.tree.descendants(option.target_loop_nid)
        for producer in ir.dependency.producers(option.block_nid):
            if producer in target_descendants:
                continue
            if option.target_loop_nid in ir.tree.descendants(producer):
                continue
            producer_root = self._root_sibling_of(ir, producer)
            if producer_root not in root_order:
                raise TransformLegalityError(f"producer block {producer} not under a root-sibling")
            if root_order.index(producer_root) < target_index:
                continue
            raise TransformLegalityError(
                f"producer block {producer} runs after the target loop "
                f"(root index {root_order.index(producer_root)} >= target {target_index})"
            )

    @staticmethod
    def _root_sibling_of(ir: KernelIR, nid: int) -> int:
        """Return the direct child of tree.root that is nid or an ancestor of it."""
        if nid in ir.tree.children(ir.tree.root):
            return nid
        for anc in ir.tree.ancestors(nid):
            if anc in ir.tree.children(ir.tree.root):
                return anc
        raise TransformLegalityError(f"node {nid} has no root-sibling ancestor")


__all__ = ["ReverseComputeAt", "ReverseComputeAtOption"]
