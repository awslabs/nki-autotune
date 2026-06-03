"""``Reorder`` transform — swap an adjacent parent-child ForNode pair via payload swap."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Var
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import BlockNode, ForNode, role_of
from nkigym.ops.base import AxisRole
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class ReorderOption(TransformOption):
    """Swap the payloads of two adjacent parent-child ForNodes."""

    outer_nid: int
    inner_nid: int


class Reorder(Transform):
    """Swap an adjacent parent-child ForNode pair via payload swap."""

    def analyze(self, ir: KernelIR) -> list[ReorderOption]:
        options: list[ReorderOption] = []
        for nid in ir.tree.preorder():
            data = ir.tree.data(nid)
            if not isinstance(data, ForNode):
                continue
            kids = ir.tree.children(nid)
            if len(kids) != 1:
                continue
            kid_data = ir.tree.data(kids[0])
            if not isinstance(kid_data, ForNode):
                continue
            opt = ReorderOption(outer_nid=nid, inner_nid=kids[0])
            if self._is_legal(ir, opt):
                options.append(opt)
        return options

    def apply(self, ir: KernelIR, option: ReorderOption) -> KernelIR:
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        outer_data = new_ir.tree.data(option.outer_nid)
        inner_data = new_ir.tree.data(option.inner_nid)
        new_ir.tree.graph.nodes[option.outer_nid]["data"] = inner_data
        new_ir.tree.graph.nodes[option.inner_nid]["data"] = outer_data
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def _is_legal(self, ir: KernelIR, option: ReorderOption) -> bool:
        try:
            self._check_legality(ir, option)
        except TransformLegalityError:
            return False
        return True

    def _check_legality(self, ir: KernelIR, option: ReorderOption) -> None:
        for nid in (option.outer_nid, option.inner_nid):
            if nid not in ir.tree.graph:
                raise TransformLegalityError(f"Reorder: nid {nid} not in tree")
        outer = ir.tree.data(option.outer_nid)
        inner = ir.tree.data(option.inner_nid)
        if not isinstance(outer, ForNode) or not isinstance(inner, ForNode):
            raise TransformLegalityError(
                f"Reorder: both targets must be ForNode; got {type(outer).__name__}, {type(inner).__name__}"
            )
        kids = ir.tree.children(option.outer_nid)
        if kids != [option.inner_nid]:
            raise TransformLegalityError(f"Reorder: inner must be sole child of outer; got children {kids}")
        outer_loop_var = outer.loop_var
        inner_loop_var = inner.loop_var
        for descendant in ir.tree.blocks(option.inner_nid):
            block = ir.tree.data(descendant)
            assert isinstance(block, BlockNode)
            for loop_var in (outer_loop_var, inner_loop_var):
                axis = _axis_for_loop_var(block, loop_var)
                if axis is None:
                    continue
                if role_of(block, axis) == AxisRole.SEQUENTIAL:
                    raise TransformLegalityError(
                        f"Reorder rejected: descendant block has SEQUENTIAL role on loop_var {loop_var!r}"
                    )


def _axis_for_loop_var(block: BlockNode, loop_var: str) -> str | None:
    """Return the iter_var axis bound by ``loop_var``, if any."""
    for iv, value in zip(block.iter_vars, block.iter_values):
        if isinstance(value, Var) and value.name == loop_var:
            return iv.axis
    return None


__all__ = ["Reorder", "ReorderOption"]
