"""Buffer geometry compaction over a transformed schedule tree.

Primary entry points:

* :func:`compact_buffer_shape` — recompute one selected :class:`Buffer`'s
  logical shape as the bounding box of its access regions without moving its
  declaration or changing its allocation layout.
* :func:`compact_shapes` — apply the same shape calculation to every declared
  buffer.
* :func:`rebased_region` — a read-time projection that subtracts a
  buffer's anchor loop vars (the loops enclosing all of its touchers) from
  a region's ``lo``, so a compacted buffer is indexed within its single
  live instance. Never written back (tree regions stay global-frame for
  ``Dependency``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from nkigym.ir.arith.expr import Const, Expr, substitute, to_affine
from nkigym.ir.buffer_placement import _anchor_loop_nids, _regions_touching
from nkigym.ir.tree import PARTITION_DIM, BlockNode, Buffer, BufferRegion, ForNode, ISANode, KernelTree


def compact_buffer_shape(tree: KernelTree, tensor: str) -> Buffer:
    """Recompute only ``tensor``'s logical shape while preserving its declaration and layout."""
    anchor_loop_nids = _anchor_loop_nids(tree, tensor)
    anchors = {tree.loop(nid).loop_var for nid in anchor_loop_nids}
    compacted: Buffer | None = None
    for block_nid in tree.blocks():
        block = tree.data(block_nid)
        assert isinstance(block, BlockNode)
        if not any(b.name == tensor for b in block.alloc_buffers):
            continue
        if compacted is not None:
            raise AssertionError(f"buffer {tensor!r} is declared by multiple blocks")
        new_bufs: list[Buffer] = []
        for buf in block.alloc_buffers:
            updated = _compact_one(tree, buf, anchors) if buf.name == tensor else buf
            new_bufs.append(updated)
            if buf.name == tensor:
                compacted = updated
        tree.graph.nodes[block_nid]["data"] = replace(block, alloc_buffers=tuple(new_bufs))
    if compacted is None:
        raise KeyError(f"buffer {tensor!r} is declared by no block")
    return compacted


def rebase_regions_of(tree: KernelTree, tensor: str) -> None:
    """Rewrite every ISA-leaf region naming ``tensor`` into single-instance local frame.

    Subtracts ``tensor``'s anchor loop vars (loops selecting which instance is
    live) from each region axis ``lo``, materializing on the tree what
    ``rebased_region`` used to compute at render time. shared_hbm buffers and
    buffers with no anchors are left unchanged (identity).
    """
    buf = next((b for nid in tree.blocks() for b in tree.block(nid).alloc_buffers if b.name == tensor), None)
    if buf is None or buf.location == "shared_hbm":
        return
    anchors = _anchor_loop_vars(tree, tensor)
    if not anchors:
        return
    subs: dict[str, Expr] = {a: Const(value=0) for a in anchors}
    for nid in tree.preorder():
        data = tree.data(nid)
        if not isinstance(data, ISANode):
            continue
        new_bindings = {
            slot: (
                BufferRegion(tensor=region.tensor, ranges=tuple((substitute(lo, subs), w) for lo, w in region.ranges))
                if region.tensor == tensor
                else region
            )
            for slot, region in data.operand_bindings.items()
        }
        if new_bindings != data.operand_bindings:
            tree.graph.nodes[nid]["data"] = replace(data, operand_bindings=new_bindings)


def compact_shapes(tree: KernelTree, anchor_loop_nids_by_tensor: Mapping[str, frozenset[int]] | None = None) -> None:
    """Recompute and write back every Buffer's logical shape (bbox over its LCA scope)."""
    for block_nid in tree.blocks():
        block = tree.data(block_nid)
        assert isinstance(block, BlockNode)
        if not block.alloc_buffers:
            continue
        new_bufs: list[Buffer] = []
        for buf in block.alloc_buffers:
            anchor_loop_nids = (
                anchor_loop_nids_by_tensor.get(buf.name) if anchor_loop_nids_by_tensor is not None else None
            )
            if anchor_loop_nids is None:
                anchor_loop_nids = _anchor_loop_nids(tree, buf.name)
            anchors = {tree.loop(nid).loop_var for nid in anchor_loop_nids}
            new_bufs.append(_compact_one(tree, buf, anchors))
        tree.graph.nodes[block_nid]["data"] = replace(block, alloc_buffers=new_bufs)


def _anchor_loop_vars(tree: KernelTree, tensor: str) -> set[str]:
    """Return loop vars that select one reusable instance of ``tensor``."""
    return {tree.loop(nid).loop_var for nid in _anchor_loop_nids(tree, tensor)}


def _compact_one(tree: KernelTree, buf: Buffer, anchors: set[str]) -> Buffer:
    """Return a copy of ``buf`` whose logical shape is the bbox of its access regions.

    shared_hbm buffers keep their declared shape (params/outputs are never
    resized). For sbuf/psum, each logical axis extent is the max of
    ``lo + width`` over the interior-loop box, with anchor loop vars zeroed
    and interior loop vars ranging over their leaf-local extents.
    """
    if buf.location == "shared_hbm":
        return buf
    pairs = _regions_touching(tree, buf.name)
    if not pairs:
        return buf
    n_axes = len(buf.shape)
    new_shape = list(buf.shape)
    for axis in range(n_axes):
        widest = 0
        for leaf_nid, region in pairs:
            if axis >= len(region.ranges):
                continue
            lo, width = region.ranges[axis]
            extents = _leaf_loop_extents(tree, leaf_nid)
            span = _axis_span(lo, width, axis, buf.location, anchors, extents)
            widest = max(widest, span)
        new_shape[axis] = widest
    return replace(buf, shape=tuple(new_shape))


def _axis_span(lo: Expr, width: Expr, axis: int, location: str, anchors: set[str], extents: dict[str, int]) -> int:
    """Max value of ``lo + width`` over the interior-loop box, anchors zeroed.

    Axis 0 of sbuf/psum carries a bare partition-tile index with width 128;
    its compacted extent is reported in element space (list_len * 128).
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
    is_partition = axis == 0 and location in ("sbuf", "psum") and width.value == PARTITION_DIM
    if is_partition:
        return (hi + 1) * PARTITION_DIM
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


def rebased_region(region: BufferRegion, buf: Buffer, tree: KernelTree) -> BufferRegion:
    """Subtract the buffer's anchor loop vars from each axis ``lo``.

    A buffer's anchors are the loops enclosing all of its touchers (see
    :func:`_anchor_loop_vars`). shared_hbm params/outputs are never rebased:
    they address absolute HBM, so the enclosing loop index is part of the
    address (symmetric with :func:`_compact_one`, which never resizes them).
    Canonical sbuf/psum buffers (touchers in disjoint nests) project to
    themselves. For a compacted sbuf/psum buffer whose touchers share
    enclosing loops, those loop vars are subtracted so the index addresses
    the single resident instance (e.g. ``[i_d0_0, (i_d1_0)*128 : +128]`` →
    ``[0, 0:128]``).
    """
    if buf.location == "shared_hbm":
        return region
    anchors = _anchor_loop_vars(tree, buf.name)
    if not anchors:
        return region
    subs: dict[str, Expr] = {a: Const(value=0) for a in anchors}
    new_ranges = tuple((substitute(lo, subs), width) for lo, width in region.ranges)
    return BufferRegion(tensor=region.tensor, ranges=new_ranges)


__all__ = ["compact_buffer_shape", "compact_shapes", "rebase_regions_of", "rebased_region"]
