"""``ComputeAt`` — sink a producer block under a consumer's loop.

See ``compute_at_legality.md`` (conditions 1-4, 5a, 6). Structural move is
the shared ``_move``; this face owns the consumer-direction legality plus
the output-block guard (condition 4: the kernel's final store cannot be
sunk).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from nkigym.codegen.compact import compact_shapes
from nkigym.ir import KernelIR
from nkigym.ir.buffer_placement import place_buffers
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import BlockNode, ForNode, ISANode, role_of
from nkigym.ops.base import AxisRole
from nkigym.transforms._code_motion import _move
from nkigym.transforms._domain_solve import _enclosing_block, _loopvar_to_dim
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
        """Conditions 1-4, 5a. (6 enumerated by _legal_indices.)"""
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
        self._check_no_writer_under_accumulation(ir, option)
        self._check_consumers_visited(ir, option)

    def _check_no_writer_under_accumulation(self, ir: KernelIR, option: ComputeAtOption) -> None:
        """Reject sinking a block that writes a tensor under a reduction loop.

        Sinking an initializer/writer inside a reduction loop re-runs the init
        each accumulation step, destroying the reduction (e.g. a PSUM memset
        sunk under the matmul's K loop re-zeros the accumulator every K-tile, so
        only the last tile's product survives). The only op that legitimately
        lives inside a reduction loop is the reducer itself, never a separate
        writer block.

        The guard fires when the moved block has any write region AND the
        target loop binds an iter-var whose role on its owning block is
        ``AxisRole.ACCUMULATION``.
        """
        moved = ir.tree.data(option.block_nid)
        assert isinstance(moved, BlockNode)
        written = {region.tensor for region in moved.writes}
        if not written:
            return
        owner_nid = _enclosing_block(ir.tree, option.target_loop_nid)
        owner = ir.tree.data(owner_nid)
        assert isinstance(owner, BlockNode)
        target = ir.tree.data(option.target_loop_nid)
        assert isinstance(target, ForNode)
        axis = _loopvar_to_dim(ir.tree, owner_nid).get(target.loop_var)
        if axis is None or role_of(owner, axis) != AxisRole.ACCUMULATION:
            return
        raise TransformLegalityError(
            f"ComputeAt cannot sink writer of {sorted(written)} under accumulation loop "
            f"{target.loop_var} (axis {axis}); re-runs the init each reduction step"
        )

    def _check_consumers_visited(self, ir: KernelIR, option: ComputeAtOption) -> None:
        """Condition 5a: every consumer is under the target, encloses the target, OR is a later root-sibling."""
        target_root = self._root_sibling_of(ir, option.target_loop_nid)
        root_order = ir.tree.children(ir.tree.root)
        target_index = root_order.index(target_root)
        target_descendants = ir.tree.descendants(option.target_loop_nid)
        for consumer in ir.dependency.consumers(option.block_nid):
            if consumer in target_descendants:
                continue
            if option.target_loop_nid in ir.tree.descendants(consumer):
                continue
            consumer_root = self._root_sibling_of(ir, consumer)
            if consumer_root not in root_order:
                raise TransformLegalityError(f"consumer block {consumer} not under a root-sibling")
            if root_order.index(consumer_root) > target_index:
                continue
            raise TransformLegalityError(
                f"consumer block {consumer} runs before the target loop "
                f"(root index {root_order.index(consumer_root)} <= target {target_index})"
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


__all__ = ["ComputeAt", "ComputeAtOption"]
