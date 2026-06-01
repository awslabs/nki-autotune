"""``ReverseComputeAt`` transform — lift a consumer block under a producer's loop.

See ``nkigym/src/nkigym/transforms/compute_at_legality.md`` for the six
legality conditions and worked examples.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from nkigym.codegen.compact import compact_shapes
from nkigym.ir import KernelIR
from nkigym.ir.buffer_placement import place_buffers
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms._code_motion import _compute_at_impl
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class ReverseComputeAtOption(TransformOption):
    """Lift consumer ``block_nid`` under ``target_loop_nid``.

    Attributes:
        block_nid: root-child :class:`BlockNode` to lift (the consumer).
        target_loop_nid: a :class:`ForNode` in a producer block's nest;
            the consumer is moved to execute inside this loop.
        index: insertion position among the target loop body's children
            (TVM convention: -1 = last legal slot).
    """

    block_nid: int
    target_loop_nid: int
    index: int = -1


class ReverseComputeAt(Transform):
    """Lift a consumer block under a producer's loop."""

    def apply(self, ir: KernelIR, option: ReverseComputeAtOption) -> KernelIR:
        """Re-check legality, deep-copy, lift the block, re-derive geometry, return new IR."""
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        _compute_at_impl(
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
        """Enumerate (consumer block, target loop) pairs passing legality."""
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
                opt = ReverseComputeAtOption(block_nid=block_nid, target_loop_nid=target_nid)
                try:
                    self._check_legality(ir, opt)
                except TransformLegalityError:
                    continue
                options.append(opt)
        return options

    def _check_legality(self, ir: KernelIR, option: ReverseComputeAtOption) -> None:
        """Raise :class:`TransformLegalityError` on any rule violation."""
        if option.target_loop_nid not in ir.tree.graph:
            raise TransformLegalityError(f"ReverseComputeAt.target_loop_nid={option.target_loop_nid} not in tree")
        target = ir.tree.data(option.target_loop_nid)
        if not isinstance(target, ForNode):
            raise TransformLegalityError(
                f"ReverseComputeAt requires target_loop_nid to be a ForNode; got {type(target).__name__}"
            )
        if option.block_nid not in ir.tree.graph:
            raise TransformLegalityError(f"ReverseComputeAt.block_nid={option.block_nid} not in tree")
        if option.target_loop_nid in ir.tree.descendants(option.block_nid):
            raise TransformLegalityError(
                f"ReverseComputeAt: target_loop_nid={option.target_loop_nid} is a descendant of the moved "
                f"block={option.block_nid} (cannot lift a block under its own loop)"
            )
        self._check_producers_visited(ir, option)

    def _check_producers_visited(self, ir: KernelIR, option: ReverseComputeAtOption) -> None:
        """Condition 5b: every producer of the moved consumer must be a descendant of the
        target loop, OR live in a root-sibling whose pre-order index is before the target's.

        A producer that runs entirely before the target loop (earlier root-sibling) writes
        its output before the lifted consumer reads it. A producer under the target shares
        the iteration. Any other producer would run after the lifted consumer — illegal.
        """
        target_root = self._root_sibling_of(ir, option.target_loop_nid)
        root_order = ir.tree.children(ir.tree.root)
        target_index = root_order.index(target_root)
        target_descendants = ir.tree.descendants(option.target_loop_nid)
        for producer in ir.dependency.producers(option.block_nid):
            if producer in target_descendants:
                continue
            if option.target_loop_nid in ir.tree.descendants(producer):
                """Producer encloses the target loop — the lifted consumer inserts into the
                producer's own body and reads what each iteration just produced. Satisfied."""
                continue
            producer_root = self._root_sibling_of(ir, producer)
            if producer_root not in root_order:
                raise TransformLegalityError(f"ReverseComputeAt: producer block {producer} is not under a root-sibling")
            if root_order.index(producer_root) < target_index:
                continue
            raise TransformLegalityError(
                f"ReverseComputeAt: producer block {producer} runs after the target loop "
                f"(root index {root_order.index(producer_root)} >= target {target_index}); "
                f"not all producers are visited before the lifted consumer"
            )

    @staticmethod
    def _root_sibling_of(ir: KernelIR, nid: int) -> int:
        """Return the direct child of ``tree.root`` that is ``nid`` or an ancestor of it."""
        if nid in ir.tree.children(ir.tree.root):
            return nid
        for anc in ir.tree.ancestors(nid):
            if anc in ir.tree.children(ir.tree.root):
                return anc
        raise TransformLegalityError(f"node {nid} has no root-sibling ancestor")


__all__ = ["ReverseComputeAt", "ReverseComputeAtOption"]
