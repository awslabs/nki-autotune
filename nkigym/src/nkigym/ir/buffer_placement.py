"""Lifetime-safe LCA buffer placement over a :class:`KernelTree`.

Each :class:`Buffer` starts from the lowest common ancestor of every block
that reads or writes it. If that block is nested under a loop, the declaration
may remain inside the loop only when every touching region uses the loop as the
same buffer-instance selector. Otherwise values are live across iterations and
the declaration is lifted above that loop.

:func:`place_buffers` is a pure recompute: it gathers every Buffer
currently declared anywhere in the tree, clears all ``alloc_buffers``,
recomputes each Buffer's lifetime-safe scope, and re-attaches. It is idempotent
and safe to call after a transform has moved blocks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from nkigym.ir.arith.expr import to_affine
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode, KernelTree


def place_buffers(tree: KernelTree, anchor_loop_nids_by_tensor: Mapping[str, frozenset[int]] | None = None) -> None:
    """Recompute and apply lifetime-safe LCA placement in place.

    Gathers all Buffers from existing ``alloc_buffers``, clears them,
    then re-attaches each at or above the LCA block of its touchers. Buffers
    remain in first-seen order so each ``alloc_buffers`` list is deterministic.
    """
    buffers = _gather_buffers(tree)
    _clear_alloc_buffers(tree)
    touchers = _touchers_by_tensor(tree)
    placement: dict[int, list[Buffer]] = {}
    for name, buf in buffers.items():
        touch = touchers.get(name)
        if buf.location == "shared_hbm":
            block_nid = tree.root
        else:
            lca = tree.root if not touch else _lca(tree, touch)
            anchor_loop_nids = anchor_loop_nids_by_tensor.get(name) if anchor_loop_nids_by_tensor is not None else None
            if anchor_loop_nids is None:
                anchor_loop_nids = _anchor_loop_nids(tree, name)
            block_nid = _safe_enclosing_block(tree, lca, anchor_loop_nids)
        placement.setdefault(block_nid, []).append(buf)
    for block_nid, bufs in placement.items():
        blk = tree.data(block_nid)
        assert isinstance(blk, BlockNode)
        tree.graph.nodes[block_nid]["data"] = replace(blk, alloc_buffers=tuple(bufs))


def place_buffer(tree: KernelTree, tensor: str, anchor_loop_nids: frozenset[int] | None = None) -> None:
    """Recompute and apply LCA-of-users placement for one buffer.

    Every non-selected buffer remains attached to its current owning block.
    This is required by per-buffer transforms: earlier structural rewrites may
    have made several declarations eligible for tighter placement, but selecting
    one buffer must not materialize placement changes for the others.
    """
    buffers = _gather_buffers(tree)
    if tensor not in buffers:
        raise KeyError(f"buffer {tensor!r} is declared by no block")
    buf = buffers[tensor]
    source_nid = _declaring_block(tree, tensor)
    touch = _touchers_by_tensor(tree).get(tensor)
    if buf.location == "shared_hbm":
        target_nid = tree.root
    else:
        lca = tree.root if not touch else _lca(tree, touch)
        resolved_anchors = anchor_loop_nids if anchor_loop_nids is not None else _anchor_loop_nids(tree, tensor)
        target_nid = _safe_enclosing_block(tree, lca, resolved_anchors)
    if source_nid != target_nid:
        source = tree.block(source_nid)
        remaining = tuple(candidate for candidate in source.alloc_buffers if candidate.name != tensor)
        tree.graph.nodes[source_nid]["data"] = replace(source, alloc_buffers=remaining)

        order = {name: index for index, name in enumerate(buffers)}
        target = tree.block(target_nid)
        placed = tuple(sorted((*target.alloc_buffers, buf), key=lambda candidate: order[candidate.name]))
        tree.graph.nodes[target_nid]["data"] = replace(target, alloc_buffers=placed)


def _safe_enclosing_block(tree: KernelTree, lca_nid: int, anchor_loop_nids: frozenset[int]) -> int:
    """Return the deepest block at or above every non-anchor ancestor loop."""
    target_nid = _enclosing_block(tree, lca_nid)
    for ancestor in tree.ancestors(target_nid):
        if isinstance(tree.data(ancestor), ForNode) and ancestor not in anchor_loop_nids:
            target_nid = _enclosing_block(tree, ancestor)
            break
    return target_nid


def _declaring_block(tree: KernelTree, tensor: str) -> int:
    """Return the unique block that declares ``tensor``."""
    owners = [nid for nid in tree.blocks() if any(buffer.name == tensor for buffer in tree.block(nid).alloc_buffers)]
    if len(owners) != 1:
        raise AssertionError(f"buffer {tensor!r} must have exactly one declaration, found {owners}")
    return owners[0]


def _enclosing_block(tree: KernelTree, nid: int) -> int:
    """Return ``nid`` if it is a BlockNode, else its nearest BlockNode ancestor.

    A buffer is declared on a block, but the LCA of its touchers can be a
    ForNode when two co-located blocks share an enclosing loop (e.g. a store
    lifted next to its tensor_copy under a shared d2 loop). The owning block is
    then the nearest BlockNode at or above that loop.
    """
    cur = nid
    while not isinstance(tree.data(cur), BlockNode):
        parent = tree.parent(cur)
        assert parent is not None, f"node {nid} has no enclosing BlockNode"
        cur = parent
    return cur


def _gather_buffers(tree: KernelTree) -> dict[str, Buffer]:
    """Collect every Buffer currently declared in any block, keyed by name, first-seen order."""
    out: dict[str, Buffer] = {}
    for nid in tree.blocks():
        blk = tree.data(nid)
        assert isinstance(blk, BlockNode)
        for buf in blk.alloc_buffers:
            if buf.name not in out:
                out[buf.name] = buf
    return out


def _clear_alloc_buffers(tree: KernelTree) -> None:
    """Set every block's ``alloc_buffers`` to the empty tuple."""
    for nid in tree.blocks():
        blk = tree.data(nid)
        assert isinstance(blk, BlockNode)
        if blk.alloc_buffers:
            tree.graph.nodes[nid]["data"] = replace(blk, alloc_buffers=())


def _touchers_by_tensor(tree: KernelTree) -> dict[str, set[int]]:
    """Map each buffer name to the set of block nids that read or write it."""
    touchers: dict[str, set[int]] = {}
    for nid in tree.blocks():
        blk = tree.data(nid)
        assert isinstance(blk, BlockNode)
        for region in (*blk.reads, *blk.writes):
            touchers.setdefault(region.tensor, set()).add(nid)
    return touchers


def _anchor_loop_nids(tree: KernelTree, tensor: str) -> frozenset[int]:
    """Return common outer loops that select one reusable buffer instance.

    An anchor encloses every ISA access and has the same coefficient on every
    region axis across all touchers. Anchors form an outer prefix of the common
    loop chain; once one loop is inconsistent, values can remain live across it
    and every inner loop must share that wider lifetime.
    """
    pairs = _regions_touching(tree, tensor)
    anchors: set[int] = set()
    if pairs:
        per_leaf = [
            {ancestor for ancestor in tree.ancestors(leaf) if isinstance(tree.data(ancestor), ForNode)}
            for leaf, _region in pairs
        ]
        common = set.intersection(*per_leaf)
        candidates = [nid for nid in tree.ancestors(pairs[0][0]) if nid in common]
        regions = [region for _leaf, region in pairs]
        for nid in candidates:
            loop_var = tree.loop(nid).loop_var
            if not _offsets_consistently(loop_var, regions):
                break
            anchors.add(nid)
    return frozenset(anchors)


def _offsets_consistently(loop_var: str, regions: list[BufferRegion]) -> bool:
    """Return whether ``loop_var`` has one coefficient per axis across all regions."""
    n_axes = max(len(region.ranges) for region in regions)
    return all(len({_axis_coeff(region, axis, loop_var) for region in regions}) == 1 for axis in range(n_axes))


def _axis_coeff(region: BufferRegion, axis: int, loop_var: str) -> int:
    """Return ``loop_var``'s coefficient in one region-axis lower bound."""
    coeff = 0
    if axis < len(region.ranges):
        lower, _width = region.ranges[axis]
        coeff = to_affine(lower).get(loop_var, 0)
    return coeff


def _regions_touching(tree: KernelTree, tensor: str) -> list[tuple[int, BufferRegion]]:
    """Return every ISA leaf and operand region naming ``tensor``."""
    regions: list[tuple[int, BufferRegion]] = []
    for nid in tree.preorder():
        data = tree.data(nid)
        if not isinstance(data, ISANode):
            continue
        for region in data.operand_bindings.values():
            if region.tensor == tensor:
                regions.append((nid, region))
    return regions


def _lca(tree: KernelTree, nids: set[int]) -> int:
    """Lowest common ancestor of ``nids`` (deepest common ancestor).

    For a single-element set, returns that element.
    """
    if len(nids) == 1:
        return next(iter(nids))
    ancestor_sets: list[set[int]] = []
    for nid in nids:
        anc = set(tree.ancestors(nid))
        anc.add(nid)
        ancestor_sets.append(anc)
    common = ancestor_sets[0].intersection(*ancestor_sets[1:])
    lca_nid = tree.root
    max_depth = -1
    for candidate in common:
        depth = len(tree.ancestors(candidate))
        if depth > max_depth:
            max_depth = depth
            lca_nid = candidate
    return lca_nid


__all__ = ["place_buffer", "place_buffers"]
