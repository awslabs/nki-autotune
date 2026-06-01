"""``Fuse`` transform — collapse adjacent same-axis ForNodes (or absorb them into a tensorize tile)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import prod

from nkigym.ir import KernelIR
from nkigym.ir.dependency import Dependency
from nkigym.ir.expr import Expr
from nkigym.ir.tree import BlockNode, ForNode, ISANode, KernelTree
from nkigym.transforms._normalize import _dim_from_loopvar, normalize_block
from nkigym.transforms._tile_region import retile_region
from nkigym.transforms._tree_ops import _replace_in_parent_children
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class FuseOption(TransformOption):
    """Per-application payload for :class:`Fuse`.

    Attributes:
        target_nids: Adjacent axis-chain entries to fuse, parent->child order.
            ``len >= 2``.
        target_axis: ``None`` for outer-trip flavour. The concrete iter_var
            axis name (e.g. ``"d1"``) for tensorize flavour; matches
            ``IterVar.axis``.
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
                    """Two adjacent ForNodes are fusion candidates iff they bind the same dim."""
                    if _dim_from_loopvar(data.loop_var) != _dim_from_loopvar(kid_data.loop_var):
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
        """Outer-trip Fuse: merge a parent->child chain of same-dim ForNodes into one loop.

        Only the loop topology changes: the chain is replaced by a single
        ForNode whose extent is the product of the chain extents (the access
        tile width is unchanged). :func:`normalize_block` then assigns the
        dense name and rebuilds the iter_values + region offsets from the new
        loop structure.
        """
        nids = option.target_nids
        first = ir.tree.data(nids[0])
        assert isinstance(first, ForNode)
        parent_nid = ir.tree.parent(nids[0])
        assert parent_nid is not None
        deepest_kids = ir.tree.children(nids[-1])
        new_extent = prod(ir.tree.data(nid).extent for nid in nids)
        block_nid, _block = _find_enclosing_block(ir.tree, nids[0])

        new_nid = ir.tree.add_node(ForNode(loop_var=f"{first.loop_var}__fused", extent=new_extent), parent=None)
        for child_nid in deepest_kids:
            ir.tree.graph.add_edge(new_nid, child_nid)
        _replace_in_parent_children(ir.tree, parent_nid, [nids[0]], [new_nid])
        for nid in nids:
            ir.tree.graph.remove_node(nid)

        normalize_block(ir.tree, block_nid)

    def _do_tensorize(self, ir: KernelIR, option: FuseOption) -> None:
        """Tensorize Fuse: absorb a chain of same-axis ForNodes above an ISA leaf into the tile width.

        ``option.target_nids[-1]`` is the ISA leaf; the prefix is the ForNode
        chain to absorb. The chain is removed and the affected-axis access
        width grows by the product of the absorbed extents;
        :func:`normalize_block` then drops any now-trip-1 loops, re-densifies
        names, and recomputes the region offsets from the surviving loops.
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
        for nid in for_chain:
            ir.tree.graph.remove_node(nid)
        ir.tree.graph.add_edge(chain_root_parent, leaf_nid)

        inverse_axis_map = {concrete: abstract for abstract, concrete in block.axis_map.items()}
        abstract_axis = inverse_axis_map.get(option.target_axis)

        def _widen(lo: Expr, width: int) -> tuple[Expr, int]:
            """Keep the offset (normalize recomputes it); grow the tile width."""
            return lo, width * absorbed_extent

        new_bindings = {
            slot: retile_region(region, leaf.op_cls.OPERAND_AXES[slot], abstract_axis, _widen)
            for slot, region in leaf.operand_bindings.items()
        }
        ir.tree.graph.nodes[leaf_nid]["data"] = ISANode(
            op_cls=leaf.op_cls, operand_bindings=new_bindings, kwargs=dict(leaf.kwargs)
        )

        """Block reads/writes are keyed by tensor name, not slot; map tensor->axes via the leaf
        so each region uses its own operand's axes (matmul stationary lacks N -> no-op)."""
        tensor_to_axes = {leaf.operand_bindings[s].tensor: leaf.op_cls.OPERAND_AXES[s] for s in leaf.operand_bindings}
        new_block = BlockNode(
            iter_vars=block.iter_vars,
            iter_values=block.iter_values,
            reads=tuple(retile_region(r, tensor_to_axes.get(r.tensor, ()), abstract_axis, _widen) for r in block.reads),
            writes=tuple(
                retile_region(w, tensor_to_axes.get(w.tensor, ()), abstract_axis, _widen) for w in block.writes
            ),
            alloc_buffers=block.alloc_buffers,
            annotations=dict(block.annotations),
            axis_map=block.axis_map,
        )
        ir.tree.graph.nodes[block_nid]["data"] = new_block

        normalize_block(ir.tree, block_nid)


def _find_enclosing_block(tree: KernelTree, nid: int) -> tuple[int, BlockNode]:
    """Walk ancestors of ``nid`` until we hit a BlockNode."""
    for ancestor in reversed(tree.ancestors(nid)):
        data = tree.data(ancestor)
        if isinstance(data, BlockNode):
            return ancestor, data
    raise TransformLegalityError(f"no enclosing BlockNode for nid {nid}")


__all__ = ["Fuse", "FuseOption"]
