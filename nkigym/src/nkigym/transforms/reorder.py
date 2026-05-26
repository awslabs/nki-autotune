"""``Reorder`` transform — swap an adjacent parent-child ForNode pair via payload swap."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from nkigym.ir import KernelIR
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import ForNode, ISANode, role_of
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.base import AxisRole
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class ReorderOption(TransformOption):
    """Per-application payload for :class:`Reorder`.

    Attributes:
        outer_nid: nid of the parent :class:`ForNode` to swap.
        inner_nid: nid of its sole :class:`ForNode` child.
    """

    outer_nid: int
    inner_nid: int


class Reorder(Transform):
    """Swap an adjacent parent-child ForNode pair via payload swap.

    See ``docs/superpowers/specs/2026-05-26-reorder-transform-design.md``.
    """

    def analyze(self, ir: KernelIR) -> list[ReorderOption]:
        """Enumerate every legal adjacent-pair ForNode swap."""
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
        """Re-check legality, deep-copy ``ir``, swap the two payloads, return."""
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        outer_data = new_ir.tree.data(option.outer_nid)
        inner_data = new_ir.tree.data(option.inner_nid)
        new_ir.tree.graph.nodes[option.outer_nid]["data"] = inner_data
        new_ir.tree.graph.nodes[option.inner_nid]["data"] = outer_data
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def _is_legal(self, ir: KernelIR, option: ReorderOption) -> bool:
        """Wrapper around :meth:`_check_legality` that returns a bool.

        Used by :meth:`analyze` to filter candidate options without raising.
        Production-path callers must use :meth:`_check_legality` directly so
        illegal options raise loudly. Mirrors :meth:`Fuse._is_legal`.
        """
        legal = True
        try:
            self._check_legality(ir, option)
        except TransformLegalityError:
            legal = False
        return legal

    def _check_legality(self, ir: KernelIR, option: ReorderOption) -> None:
        """Raise :class:`TransformLegalityError` on any rule violation."""
        """1. Both nids exist."""
        if option.outer_nid not in ir.tree.graph:
            raise TransformLegalityError(f"Reorder.outer_nid={option.outer_nid} is not a node in the IR tree")
        if option.inner_nid not in ir.tree.graph:
            raise TransformLegalityError(f"Reorder.inner_nid={option.inner_nid} is not a node in the IR tree")
        """2. Both are ForNodes."""
        outer = ir.tree.data(option.outer_nid)
        inner = ir.tree.data(option.inner_nid)
        if not isinstance(outer, ForNode) or not isinstance(inner, ForNode):
            raise TransformLegalityError(
                f"Reorder requires both targets to be ForNode; got "
                f"outer={type(outer).__name__}, inner={type(inner).__name__}"
            )
        """3. Inner is the sole child of outer (perfect-nest of two)."""
        kids = ir.tree.children(option.outer_nid)
        if kids != [option.inner_nid]:
            raise TransformLegalityError(
                f"Reorder requires inner_nid={option.inner_nid} to be the sole child of "
                f"outer_nid={option.outer_nid}; got children {kids}"
            )
        """4. No descendant ISA leaf has SEQUENTIAL role on either swapped dim."""
        for leaf_nid in ir.tree.leaves(option.inner_nid):
            leaf = ir.tree.data(leaf_nid)
            if not isinstance(leaf, ISANode) or leaf.op_cls is NKIAlloc:
                continue
            leaf_dims = set(leaf.axis_map.values())
            for swap_dim in (outer.dim, inner.dim):
                if swap_dim in leaf_dims and role_of(leaf, swap_dim) == AxisRole.SEQUENTIAL:
                    raise TransformLegalityError(
                        f"Reorder rejected: leaf {leaf.op_cls.__name__} has SEQUENTIAL role " f"on dim {swap_dim!r}"
                    )


__all__ = ["Reorder", "ReorderOption"]
