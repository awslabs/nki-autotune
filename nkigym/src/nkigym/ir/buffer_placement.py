"""LCA-of-users buffer placement over a :class:`KernelTree`.

Each :class:`Buffer` is declared on the lowest-common-ancestor
:class:`BlockNode` of every block that reads or writes it — the
shallowest block whose scope dominates all uses (lifetime-correct).

:func:`place_buffers` is a pure recompute: it gathers every Buffer
currently declared anywhere in the tree, clears all ``alloc_buffers``,
recomputes each Buffer's LCA over its touchers, and re-attaches. It is
idempotent and safe to call after a transform has moved blocks (e.g.
``compute_at``), so allocations descend automatically when producer and
consumers become co-located.
"""

from __future__ import annotations

from dataclasses import replace

from nkigym.ir.tree import BlockNode, Buffer, KernelTree


def place_buffers(tree: KernelTree) -> None:
    """Recompute and apply LCA-of-users placement in place.

    Gathers all Buffers from existing ``alloc_buffers``, clears them,
    then re-attaches each to the LCA block of its touchers. Buffers in
    a stable iteration order (first-seen) so a block's ``alloc_buffers``
    list is deterministic.
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
            block_nid = _enclosing_block(tree, lca)
        placement.setdefault(block_nid, []).append(buf)
    for block_nid, bufs in placement.items():
        blk = tree.data(block_nid)
        assert isinstance(blk, BlockNode)
        tree.graph.nodes[block_nid]["data"] = replace(blk, alloc_buffers=tuple(bufs))


def place_buffer(tree: KernelTree, tensor: str) -> None:
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
        target_nid = _enclosing_block(tree, lca)
    if source_nid != target_nid:
        source = tree.block(source_nid)
        remaining = tuple(candidate for candidate in source.alloc_buffers if candidate.name != tensor)
        tree.graph.nodes[source_nid]["data"] = replace(source, alloc_buffers=remaining)

        order = {name: index for index, name in enumerate(buffers)}
        target = tree.block(target_nid)
        placed = tuple(sorted((*target.alloc_buffers, buf), key=lambda candidate: order[candidate.name]))
        tree.graph.nodes[target_nid]["data"] = replace(target, alloc_buffers=placed)


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
