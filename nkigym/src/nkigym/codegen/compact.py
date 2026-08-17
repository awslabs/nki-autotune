"""Selected-buffer geometry compaction over a transformed schedule tree."""

from __future__ import annotations

from dataclasses import replace

from nkigym.ir.arith.expr import Const, Expr, substitute, to_affine
from nkigym.ir.graph_index import ordered_tree_topology
from nkigym.ir.tree import PARTITION_DIM, BlockNode, Buffer, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.search.buffer_placement import _anchor_loop_nids_from_regions, _regions_by_tensor


def compact_buffer_shapes(tree: KernelTree, tensors: frozenset[str]) -> dict[str, Buffer]:
    """Compact selected buffer declarations using shared tree access indexes."""
    declared: dict[str, Buffer] = {}
    for block_nid in tree.blocks():
        block = tree.data(block_nid)
        assert isinstance(block, BlockNode)
        for buf in block.alloc_buffers:
            if buf.name in tensors:
                if buf.name in declared:
                    raise AssertionError(f"buffer {buf.name!r} is declared by multiple blocks")
                declared[buf.name] = buf
    if missing := tensors - declared.keys():
        raise KeyError(f"buffers declared by no block: {sorted(missing)}")
    regions = _regions_by_tensor(tree, tensors)
    ancestors = ordered_tree_topology(tree.graph, tree.root)[1]
    leaf_extents = {
        leaf_nid: _leaf_loop_extents(tree, leaf_nid) for pairs in regions.values() for leaf_nid, _region in pairs
    }
    compacted: dict[str, Buffer] = {}
    for tensor, buffer in declared.items():
        anchor_nids = _anchor_loop_nids_from_regions(tree, regions.get(tensor, []), ancestors)
        anchors = {tree.loop(nid).loop_var for nid in anchor_nids}
        compacted[tensor] = _compact_one(buffer, anchors, regions.get(tensor, []), leaf_extents)
    for block_nid in tree.blocks():
        block = tree.block(block_nid)
        updated = tuple(compacted.get(buffer.name, buffer) for buffer in block.alloc_buffers)
        if updated != block.alloc_buffers:
            tree.graph.nodes[block_nid]["data"] = replace(block, alloc_buffers=updated)
    return compacted


def _compact_one(
    buf: Buffer, anchors: set[str], accesses: list[tuple[int, BufferRegion]], leaf_extents: dict[int, dict[str, int]]
) -> Buffer:
    """Return a copy of ``buf`` whose logical shape is the bbox of its access regions.

    shared_hbm buffers keep their declared shape (params/outputs are never
    resized). For sbuf/psum, each logical axis extent is the max of
    ``lo + width`` over the interior-loop box, with anchor loop vars zeroed
    and interior loop vars ranging over their leaf-local extents.
    """
    if buf.location == "shared_hbm":
        return buf
    if not accesses:
        return buf
    n_axes = len(buf.shape)
    new_shape = list(buf.shape)
    for axis in range(n_axes):
        widest = 0
        for leaf_nid, region in accesses:
            if axis >= len(region.ranges):
                continue
            lo, width = region.ranges[axis]
            extents = leaf_extents[leaf_nid]
            span = _axis_span(lo, width, axis, buf.location, anchors, extents, buf.partition_extent())
            widest = max(widest, span)
        new_shape[axis] = widest
    return replace(buf, shape=tuple(new_shape))


def _axis_span(
    lo: Expr, width: Expr, axis: int, location: str, anchors: set[str], extents: dict[str, int], partition: int
) -> int:
    """Max value of ``lo + width`` over the interior-loop box, anchors zeroed.

    Axis 0 of sbuf/psum carries a bare partition-tile index; its compacted
    extent is reported in element space.
    """
    assert isinstance(width, Const), f"region width must be Const; got {width!r}"
    zeroed = substitute(lo, {a: Const(value=0) for a in anchors})
    coeffs = to_affine(zeroed)
    hi = coeffs.get(None, 0)
    for var, coeff in coeffs.items():
        if var is None:
            continue
        trips = extents.get(var, 1)
        if coeff > 0:
            hi += coeff * (trips - 1)
    is_partition = axis == 0 and location in ("sbuf", "psum") and width.value == partition
    if is_partition:
        return (hi + 1) * partition
    return hi + width.value


def _leaf_loop_extents(tree: KernelTree, leaf_nid: int) -> dict[str, int]:
    """Loop-var → extent for every ForNode that encloses ``leaf_nid`` (its ancestors).

    Built from the leaf's own ancestor chain, so a loop_var reused across
    subtrees with different extents (e.g. canonical trip-1 vs trip-16
    ``i_d1_0``) resolves to the extent in THIS leaf's scope.
    """
    out: dict[str, int] = {}
    for anc in tree.ancestors(leaf_nid):
        data = tree.data(anc)
        if isinstance(data, ForNode):
            out[data.loop_var] = data.extent
    return out


__all__ = ["compact_buffer_shapes"]
