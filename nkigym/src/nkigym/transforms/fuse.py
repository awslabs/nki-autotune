"""``Fuse`` transform — collapse adjacent same-axis ForNodes (or absorb them into a tensorize tile)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import prod

from nkigym.ir import KernelIR
from nkigym.ir.dependency import Dependency
from nkigym.ir.expr import Const, Expr, Mul, Var, from_affine, substitute, to_affine
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.transforms._tree_ops import _block_local_descendants, _replace_in_parent_children
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class FuseOption(TransformOption):
    """Per-application payload for :class:`Fuse`.

    Attributes:
        target_nids: Adjacent axis-chain entries to fuse, parent->child order.
            ``len >= 2``.
        target_axis: ``None`` for outer-trip flavour. Abstract iter_var
            axis name for tensorize flavour.
    """

    target_nids: tuple[int, ...]
    target_axis: str | None = None


class Fuse(Transform):
    """Collapse a parent->child chain of same-loop-axis entries into one."""

    def analyze(self, ir: KernelIR) -> list[FuseOption]:
        options: list[FuseOption] = []
        for nid in ir.tree.preorder():
            data = ir.tree.data(nid)
            if isinstance(data, ForNode):
                chain: list[int] = [nid]
                cur = nid
                while True:
                    kids = ir.tree.children(cur)
                    if len(kids) != 1:
                        break
                    kid_data = ir.tree.data(kids[0])
                    if not isinstance(kid_data, ForNode):
                        break
                    """Two adjacent ForNodes are fusion candidates iff their loop_vars share a stem."""
                    if not _same_loop_axis(data.loop_var, kid_data.loop_var):
                        break
                    chain.append(kids[0])
                    cur = kids[0]
                for end in range(2, len(chain) + 1):
                    sub = tuple(chain[:end])
                    options.append(FuseOption(target_nids=sub, target_axis=None))
        return options

    def apply(self, ir: KernelIR, option: FuseOption) -> KernelIR:
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        if option.target_axis is None:
            self._do_outer_trip(new_ir, option)
        else:
            self._do_tensorize(new_ir, option)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def _check_legality(self, ir: KernelIR, option: FuseOption) -> None:
        if len(option.target_nids) < 2:
            raise TransformLegalityError(f"Fuse.target_nids must have len >= 2; got {option.target_nids}")
        for nid in option.target_nids:
            if nid not in ir.tree.graph:
                raise TransformLegalityError(f"Fuse.target_nids contains unknown nid {nid}")
        nodes = [ir.tree.data(nid) for nid in option.target_nids]
        if option.target_axis is None:
            if not all(isinstance(n, ForNode) for n in nodes):
                raise TransformLegalityError(
                    f"Fuse outer-trip flavour: every target must be ForNode; got {[type(n).__name__ for n in nodes]}"
                )
            for parent_nid, child_nid in zip(option.target_nids, option.target_nids[1:]):
                kids = ir.tree.children(parent_nid)
                if kids != [child_nid]:
                    raise TransformLegalityError(
                        f"Fuse outer-trip flavour: nid {parent_nid} must have a single child {child_nid}; got {kids}"
                    )
        else:
            """Tensorize flavour: prefix is ForNodes; last is the ISA leaf."""
            if not isinstance(nodes[-1], ISANode):
                raise TransformLegalityError(
                    f"Fuse tensorize flavour: last target must be ISANode; got {type(nodes[-1]).__name__}"
                )
            for n in nodes[:-1]:
                if not isinstance(n, ForNode):
                    raise TransformLegalityError(
                        f"Fuse tensorize flavour: prefix must be all ForNodes; got {type(n).__name__}"
                    )
            for parent_nid, child_nid in zip(option.target_nids, option.target_nids[1:]):
                kids = ir.tree.children(parent_nid)
                if kids != [child_nid]:
                    raise TransformLegalityError(
                        f"Fuse tensorize flavour: nid {parent_nid} must have a single child {child_nid}; got {kids}"
                    )

    def _do_outer_trip(self, ir: KernelIR, option: FuseOption) -> None:
        nids = option.target_nids
        """Capture loop_vars and extents BEFORE removing nodes."""
        old_loop_vars_in_order = [ir.tree.data(nid).loop_var for nid in nids]
        old_extents_in_order = [ir.tree.data(nid).extent for nid in nids]

        first = ir.tree.data(nids[0])
        assert isinstance(first, ForNode)
        parent_nid = ir.tree.parent(nids[0])
        assert parent_nid is not None
        deepest_kids = ir.tree.children(nids[-1])
        new_extent = prod(old_extents_in_order)
        block_nid, block = _find_enclosing_block(ir.tree, nids[0])

        new_loop_var = _fused_loop_var(first.loop_var)
        new_nid = ir.tree.add_node(ForNode(loop_var=new_loop_var, extent=new_extent), parent=None)
        for child_nid in deepest_kids:
            ir.tree.graph.add_edge(new_nid, child_nid)
        _replace_in_parent_children(ir.tree, parent_nid, [nids[0]], [new_nid])
        for nid in nids:
            ir.tree.graph.remove_node(nid)

        """Replace iter_values that reference any old loop_var.

        When fusing a chain, the iter_value that used the chain's loop_vars
        should now bind directly to the fused loop_var. We detect which
        iter_value to replace by checking if it contains any of the old
        loop_vars.

        For reads/writes/operand_bindings, we still need to substitute all
        occurrences of old loop_vars with the new fused loop_var.
        """
        old_loop_var_set = set(old_loop_vars_in_order)

        def _contains_old_var(expr: Expr) -> bool:
            """Check if expr contains any old loop_var."""
            affine = to_affine(expr)
            return bool(old_loop_var_set & affine.keys())

        new_iter_values = tuple(
            Var(name=new_loop_var) if _contains_old_var(value) else value for value in block.iter_values
        )

        """For regions, substitute all old loop_vars → new_loop_var."""
        substitutions: dict[str, Expr] = {
            old_loop_var: Var(name=new_loop_var) for old_loop_var in old_loop_vars_in_order
        }

        new_block = BlockNode(
            iter_vars=block.iter_vars,
            iter_values=new_iter_values,
            reads=tuple(_substitute_region(r, substitutions, fused_var=new_loop_var) for r in block.reads),
            writes=tuple(_substitute_region(w, substitutions, fused_var=new_loop_var) for w in block.writes),
            alloc_buffers=block.alloc_buffers,
            annotations=dict(block.annotations),
        )
        ir.tree.graph.nodes[block_nid]["data"] = new_block

        """Propagate substitutions into descendant ISANode operand_bindings within this block's scope."""
        for desc_nid in _block_local_descendants(ir.tree, block_nid):
            desc_data = ir.tree.data(desc_nid)
            if isinstance(desc_data, ISANode):
                new_bindings = {
                    slot: _substitute_region(region, substitutions, fused_var=new_loop_var)
                    for slot, region in desc_data.operand_bindings.items()
                }
                new_isa = ISANode(op_cls=desc_data.op_cls, operand_bindings=new_bindings, kwargs=dict(desc_data.kwargs))
                ir.tree.graph.nodes[desc_nid]["data"] = new_isa

    def _do_tensorize(self, ir: KernelIR, option: FuseOption) -> None:
        """Tensorize Fuse: absorb a chain of same-axis ForNodes above an ISA leaf into the leaf's tile width.

        ``option.target_nids[-1]`` is the ISA leaf; the prefix is the
        ForNode chain to absorb. The leaf's operand_bindings on
        ``target_axis`` widen by the product of the absorbed extents.
        """
        leaf_nid = option.target_nids[-1]
        leaf = ir.tree.data(leaf_nid)
        assert isinstance(leaf, ISANode)
        for_chain = option.target_nids[:-1]
        chain_root = for_chain[0]
        chain_root_parent = ir.tree.parent(chain_root)
        assert chain_root_parent is not None
        block_nid, block = _find_enclosing_block(ir.tree, leaf_nid)

        absorbed_extent = prod(ir.tree.data(nid).extent for nid in for_chain)
        absorbed_loop_vars = [ir.tree.data(nid).loop_var for nid in for_chain]
        for nid in for_chain:
            ir.tree.graph.remove_node(nid)
        ir.tree.graph.add_edge(chain_root_parent, leaf_nid)

        new_bindings = {
            slot: _widen_region_axis(region, leaf.op_cls, slot, option.target_axis, absorbed_extent)
            for slot, region in leaf.operand_bindings.items()
        }
        new_leaf = ISANode(op_cls=leaf.op_cls, operand_bindings=new_bindings, kwargs=dict(leaf.kwargs))
        ir.tree.graph.nodes[leaf_nid]["data"] = new_leaf

        """Drop absorbed loop_vars from the block's iter_values by substituting Const(0) for each."""
        substitutions: dict[str, Expr] = {lv: Const(value=0) for lv in absorbed_loop_vars}
        new_iter_values = tuple(substitute(value, substitutions) for value in block.iter_values)
        new_block = BlockNode(
            iter_vars=block.iter_vars,
            iter_values=new_iter_values,
            reads=tuple(_substitute_region(r, substitutions) for r in block.reads),
            writes=tuple(_substitute_region(w, substitutions) for w in block.writes),
            alloc_buffers=block.alloc_buffers,
            annotations=dict(block.annotations),
        )
        ir.tree.graph.nodes[block_nid]["data"] = new_block

        """Propagate substitutions into descendant ISANode operand_bindings within this block's scope."""
        for desc_nid in _block_local_descendants(ir.tree, block_nid):
            desc_data = ir.tree.data(desc_nid)
            if isinstance(desc_data, ISANode):
                new_bindings_desc = {
                    slot: _substitute_region(region, substitutions)
                    for slot, region in desc_data.operand_bindings.items()
                }
                new_isa_desc = ISANode(
                    op_cls=desc_data.op_cls, operand_bindings=new_bindings_desc, kwargs=dict(desc_data.kwargs)
                )
                ir.tree.graph.nodes[desc_nid]["data"] = new_isa_desc


def _widen_region_axis(region: BufferRegion, op_cls, slot: str, target_axis: str, new_width: int) -> BufferRegion:
    """Widen the slice for ``target_axis`` on ``region`` to ``new_width`` if the slot's axes contain it."""
    axes = op_cls.OPERAND_AXES.get(slot)
    if axes is None or target_axis not in axes:
        return region
    axis_index = axes.index(target_axis)
    if axis_index >= len(region.ranges):
        return region
    new_ranges: list[tuple[Expr, Expr]] = []
    for i, (lo, hi) in enumerate(region.ranges):
        if i == axis_index:
            new_ranges.append((lo, Const(value=new_width)))
        else:
            new_ranges.append((lo, hi))
    return BufferRegion(tensor=region.tensor, ranges=tuple(new_ranges))


def _find_enclosing_block(tree: KernelTree, nid: int) -> tuple[int, BlockNode]:
    for ancestor in reversed(tree.ancestors(nid)):
        data = tree.data(ancestor)
        if isinstance(data, BlockNode):
            return ancestor, data
    raise TransformLegalityError(f"no enclosing BlockNode for nid {nid}")


def _same_loop_axis(a: str, b: str) -> bool:
    """Two loop_vars are 'same axis' if their split-stem matches.

    For canonical loop_var ``i_<concrete>_0`` and post-Split offspring
    ``i_<concrete>_0_0`` / ``i_<concrete>_0_1``, the stem is everything
    before the trailing ``_<int>`` suffix.
    """
    return _stem(a) == _stem(b)


def _stem(loop_var: str) -> str:
    parts = loop_var.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return loop_var


def _fused_loop_var(first_loop_var: str) -> str:
    return _stem(first_loop_var) + "_fused"


def _substitute_region(region: BufferRegion, subs: dict[str, Expr], fused_var: str | None = None) -> BufferRegion:
    """Substitute loop_vars in a BufferRegion.

    For outer-trip Fuse, if `fused_var` is set, detect subexpressions that
    contain the old loop_vars and replace them with `Var(fused_var)`.
    """
    if fused_var is not None:
        old_loop_vars = set(subs.keys())

        def _replace_if_contains_old(expr: Expr) -> Expr:
            """If expr contains any old loop_var, replace it with Var(fused_var).

            This handles the case where expr is like `(i_old_0 * 8 + i_old_1) * stride`,
            which should become `Var(fused_var) * stride`.
            """
            affine = to_affine(expr)
            if old_loop_vars & affine.keys():
                """expr contains old loop_vars. Extract the stride if any."""
                """Check if expr is of form `base * stride` where base contains old vars."""
                if isinstance(expr, Mul) and isinstance(expr.right, Const):
                    """expr = base * stride"""
                    base_affine = to_affine(expr.left)
                    if old_loop_vars & base_affine.keys():
                        """base contains old vars; replace base with fused_var."""
                        return Mul(left=Var(name=fused_var), right=expr.right)
                """expr directly contains old vars without a constant multiplier."""
                return Var(name=fused_var)
            return expr

        new_ranges = tuple((_replace_if_contains_old(lo), hi) for lo, hi in region.ranges)
    else:
        """Simple substitution + normalization."""
        new_ranges = tuple(
            (from_affine(to_affine(substitute(lo, subs))), from_affine(to_affine(substitute(hi, subs))))
            for lo, hi in region.ranges
        )
    return BufferRegion(tensor=region.tensor, ranges=new_ranges)


__all__ = ["Fuse", "FuseOption"]
