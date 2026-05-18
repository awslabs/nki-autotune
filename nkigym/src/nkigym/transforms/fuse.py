"""``Fuse`` transform — collapse adjacent axis-chain entries into one."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import prod

from nkigym.ir import KernelIR
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import ForNode, ISANode
from nkigym.ops.alloc import NKIAlloc
from nkigym.transforms._tree_ops import _replace_in_parent_children
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class FuseOption(TransformOption):
    """Per-application payload for :class:`Fuse`.

    Attributes:
        target_nids: Adjacent axis-chain entries to fuse, parent->child order.
            ``len >= 2``.
        target_axis: ``None`` selects the outer-trip flavor — every
            entry in ``target_nids`` is a :class:`ForNode`. Set to an
            abstract axis name (e.g. ``"F"``) for the tensorize flavor —
            ``target_nids[-1]`` is an :class:`ISANode` and the trailing
            ForNode chain is absorbed into its
            ``tensorize_sizes[target_axis]``.
    """

    target_nids: tuple[int, ...]
    target_axis: str | None = None


class Fuse(Transform):
    """Collapse a parent->child chain of same-axis entries into one.

    See ``docs/superpowers/specs/2026-05-16-transforms-split-fuse-design.md``.
    """

    def analyze(self, ir: KernelIR) -> list[FuseOption]:
        """Enumerate every legal fuse option (outer-trip and tensorize)."""
        options: list[FuseOption] = []
        for nid in ir.tree.preorder():
            data = ir.tree.data(nid)
            if isinstance(data, ForNode):
                chain = [nid]
                cur = nid
                while True:
                    kids = ir.tree.children(cur)
                    if len(kids) != 1:
                        break
                    kid_data = ir.tree.data(kids[0])
                    if not isinstance(kid_data, ForNode):
                        break
                    if kid_data.dim != data.dim:
                        break
                    chain.append(kids[0])
                    cur = kids[0]
                for end in range(2, len(chain) + 1):
                    sub = tuple(chain[:end])
                    opt = FuseOption(target_nids=sub, target_axis=None)
                    if self._is_legal(ir, opt):
                        options.append(opt)
            elif isinstance(data, ISANode):
                if data.op_cls is NKIAlloc:
                    continue
                for axis, dim in data.axis_map.items():
                    chain_above: list[int] = []
                    walker = ir.tree.parent(nid)
                    prev = nid
                    while walker is not None and walker != ir.tree.root:
                        wdata = ir.tree.data(walker)
                        if not isinstance(wdata, ForNode):
                            break
                        if wdata.dim != dim:
                            break
                        kids = ir.tree.children(walker)
                        if kids != [prev]:
                            break
                        chain_above.insert(0, walker)
                        prev = walker
                        walker = ir.tree.parent(walker)
                    for start in range(len(chain_above)):
                        sub = tuple(chain_above[start:] + [nid])
                        if len(sub) < 2:
                            continue
                        opt = FuseOption(target_nids=sub, target_axis=axis)
                        if self._is_legal(ir, opt):
                            options.append(opt)
        return options

    def apply(self, ir: KernelIR, option: FuseOption) -> KernelIR:
        """Re-check legality, deep-copy ``ir``, perform the fuse, return new IR."""
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        self._do_apply(new_ir, option)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def _is_legal(self, ir: KernelIR, option: FuseOption) -> bool:
        """Wrapper around :meth:`_check_legality` that returns a bool.

        Used by :meth:`analyze` to filter candidate options without raising.
        Production-path callers must use :meth:`_check_legality` directly so
        illegal options raise loudly.
        """
        legal = True
        try:
            self._check_legality(ir, option)
        except TransformLegalityError:
            legal = False
        return legal

    def _check_legality(self, ir: KernelIR, option: FuseOption) -> None:
        """Raise :class:`TransformLegalityError` if ``option`` is invalid for ``ir``."""
        if len(option.target_nids) < 2:
            raise TransformLegalityError(f"Fuse.target_nids must have len >= 2; got {option.target_nids}")
        for nid in option.target_nids:
            if nid not in ir.tree.graph:
                raise TransformLegalityError(f"Fuse.target_nids contains unknown nid {nid}")
        if option.target_axis is None:
            self._check_outer_trip(ir, option)
        else:
            self._check_tensorize(ir, option)

    def _check_outer_trip(self, ir: KernelIR, option: FuseOption) -> None:
        """Outer-trip legality: chain of ForNodes, same dim, parent->child, single-child."""
        nodes = [ir.tree.data(nid) for nid in option.target_nids]
        if not all(isinstance(n, ForNode) for n in nodes):
            raise TransformLegalityError(
                f"Fuse outer-trip flavor: every target must be ForNode; got " f"{[type(n).__name__ for n in nodes]}"
            )
        first = nodes[0]
        for n in nodes[1:]:
            if n.dim != first.dim:
                raise TransformLegalityError(
                    f"Fuse outer-trip flavor: all entries must share dim; got {first.dim!r} vs {n.dim!r}"
                )
        for parent_nid, child_nid in zip(option.target_nids, option.target_nids[1:]):
            kids = ir.tree.children(parent_nid)
            if kids != [child_nid]:
                raise TransformLegalityError(
                    f"Fuse outer-trip flavor: nid {parent_nid} must have a single ForNode child {child_nid}; "
                    f"got children {kids}"
                )

    def _check_tensorize(self, ir: KernelIR, option: FuseOption) -> None:
        """Tensorize legality: trailing ISANode + chain of same-dim ForNodes above it."""
        leaf_nid = option.target_nids[-1]
        leaf = ir.tree.data(leaf_nid)
        if not isinstance(leaf, ISANode):
            raise TransformLegalityError(
                f"Fuse tensorize flavor: last target must be ISANode; got {type(leaf).__name__}"
            )
        if leaf.op_cls is NKIAlloc:
            raise TransformLegalityError(
                "Fuse tensorize flavor: cannot fuse NKIAlloc tensorize_sizes " "(would change buffer placement)"
            )
        if option.target_axis not in leaf.axis_map:
            raise TransformLegalityError(
                f"Fuse.target_axis={option.target_axis!r} not in leaf.axis_map={list(leaf.axis_map)}"
            )
        concrete_dim = leaf.axis_map[option.target_axis]

        for_chain_nids = option.target_nids[:-1]
        for nid in for_chain_nids:
            data = ir.tree.data(nid)
            if not isinstance(data, ForNode):
                raise TransformLegalityError(
                    f"Fuse tensorize flavor: prefix entries must be ForNode; got {type(data).__name__}"
                )
            if data.dim != concrete_dim:
                raise TransformLegalityError(
                    f"Fuse tensorize flavor: prefix dim must match leaf axis concrete dim "
                    f"({concrete_dim!r}); got {data.dim!r}"
                )

        for parent_nid, child_nid in zip(option.target_nids, option.target_nids[1:]):
            kids = ir.tree.children(parent_nid)
            if kids != [child_nid]:
                raise TransformLegalityError(
                    f"Fuse tensorize flavor: nid {parent_nid} must have a single child {child_nid}; "
                    f"got children {kids}"
                )

        chain_trip_product = prod(ir.tree.data(nid).trip for nid in for_chain_nids)
        if chain_trip_product < 2:
            raise TransformLegalityError(
                f"Fuse tensorize flavor: chain trip product must be >= 2; got {chain_trip_product}"
            )

        new_tensorize = leaf.tensorize_sizes[option.target_axis] * chain_trip_product
        max_tile = leaf.op_cls.MAX_TILE_SIZE.get(option.target_axis)
        if max_tile is not None and new_tensorize > max_tile:
            raise TransformLegalityError(
                f"Fuse tensorize flavor: new tensorize {new_tensorize} > "
                f"MAX_TILE_SIZE[{option.target_axis!r}]={max_tile}"
            )

    def _do_apply(self, ir: KernelIR, option: FuseOption) -> None:
        """Mutate ``ir.tree`` in place per ``option``."""
        if option.target_axis is None:
            self._do_apply_outer_trip(ir, option)
        else:
            self._do_apply_tensorize(ir, option)

    def _do_apply_outer_trip(self, ir: KernelIR, option: FuseOption) -> None:
        """Replace a chain of same-dim ForNodes with one ForNode whose trip is the product.

        Sibling order under the chain root's parent is preserved: the new fused
        ForNode occupies the position the chain root held in the parent's child list.
        """
        nids = option.target_nids
        first = ir.tree.data(nids[0])
        assert isinstance(first, ForNode)
        parent_nid = ir.tree.parent(nids[0])
        assert parent_nid is not None
        deepest_kids = ir.tree.children(nids[-1])
        new_trip = prod(ir.tree.data(nid).trip for nid in nids)

        """Build the new fused ForNode DETACHED."""
        new_nid = ir.tree.add_node(ForNode(dim=first.dim, trip=new_trip), parent=None)

        """Reparent deepest_kids under the new node."""
        for child in deepest_kids:
            ir.tree.graph.add_edge(new_nid, child)

        """Swap nids[0] (the chain root, parent's child) for new_nid at the same position."""
        _replace_in_parent_children(ir.tree, parent_nid, [nids[0]], [new_nid])

        """Drop the chain. Each ``remove_node`` also removes any remaining incident edges."""
        for nid in nids:
            ir.tree.graph.remove_node(nid)

    def _do_apply_tensorize(self, ir: KernelIR, option: FuseOption) -> None:
        """Remove the prefix ForNodes and bump leaf.tensorize_sizes[target_axis]."""
        leaf_nid = option.target_nids[-1]
        leaf = ir.tree.data(leaf_nid)
        assert isinstance(leaf, ISANode)
        assert option.target_axis is not None
        for_chain_nids = option.target_nids[:-1]
        chain_root_parent = ir.tree.parent(for_chain_nids[0])
        assert chain_root_parent is not None

        new_tensorize = leaf.tensorize_sizes[option.target_axis] * prod(
            ir.tree.data(nid).trip for nid in for_chain_nids
        )

        """Detach the chain (and the leaf-edge it carries)."""
        for nid in for_chain_nids:
            ir.tree.graph.remove_node(nid)

        """Reattach the leaf under chain_root_parent (it was detached when the immediate parent was removed)."""
        ir.tree.graph.add_edge(chain_root_parent, leaf_nid)

        """Rewrite the leaf's tensorize_sizes entry."""
        new_tensorize_sizes = dict(leaf.tensorize_sizes)
        new_tensorize_sizes[option.target_axis] = new_tensorize
        new_leaf = ISANode(
            op_cls=leaf.op_cls,
            reads=leaf.reads,
            writes=leaf.writes,
            rmw=leaf.rmw,
            tensorize_sizes=new_tensorize_sizes,
            axis_map=dict(leaf.axis_map),
            kwargs=dict(leaf.kwargs),
            location=leaf.location,
            dtype=leaf.dtype,
        )
        ir.tree.graph.nodes[leaf_nid]["data"] = new_leaf


__all__ = ["Fuse", "FuseOption"]
