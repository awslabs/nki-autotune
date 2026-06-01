"""Buffer geometry compaction over a transformed schedule tree.

Two entry points:

* :func:`compact_shapes` — recompute each :class:`Buffer`'s logical shape
  as the bounding box of its access regions within its declaration (LCA)
  block, and write it back on the tree. Idempotent; materialized like
  :func:`nkigym.ir.buffer_placement.place_buffers`.
* :func:`rebased_region` — a read-time projection that subtracts the
  declaration block's anchor loop vars from a region's ``lo``, so a
  compacted buffer is indexed within its single live instance. Never
  written back (tree regions stay global-frame for ``Dependency``).
"""

from __future__ import annotations

from dataclasses import replace

from nkigym.ir.expr import Const, Expr, substitute, to_affine
from nkigym.ir.tree import PARTITION_DIM, BlockNode, Buffer, BufferRegion, ForNode, ISANode, KernelTree


def compact_shapes(tree: KernelTree) -> None:
    """Recompute and write back every Buffer's logical shape (bbox over its LCA scope)."""
    for block_nid in tree.blocks():
        block = tree.data(block_nid)
        assert isinstance(block, BlockNode)
        if not block.alloc_buffers:
            continue
        anchors = _anchor_loop_vars(tree, block_nid)
        new_bufs = tuple(_compact_one(tree, buf, anchors) for buf in block.alloc_buffers)
        tree.graph.nodes[block_nid]["data"] = replace(block, alloc_buffers=new_bufs)


def _anchor_loop_vars(tree: KernelTree, decl_block_nid: int) -> set[str]:
    """Loop vars bound at or above the declaration block (ForNode ancestors)."""
    out: set[str] = set()
    for anc in tree.ancestors(decl_block_nid):
        data = tree.data(anc)
        if isinstance(data, ForNode):
            out.add(data.loop_var)
    return out


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
    its compacted extent is reported in element space (num_tiles * 128).
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


def _regions_touching(tree: KernelTree, tensor: str) -> list[tuple[int, BufferRegion]]:
    """Every (ISA-leaf nid, operand BufferRegion) pair naming ``tensor``."""
    out: list[tuple[int, BufferRegion]] = []
    for nid in tree.preorder():
        data = tree.data(nid)
        if isinstance(data, ISANode):
            for region in data.operand_bindings.values():
                if region.tensor == tensor:
                    out.append((nid, region))
    return out


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
    """Subtract the declaration block's anchor loop vars from each axis ``lo``.

    Params (shared_hbm, declared at root → no anchors) project to
    themselves. For a compacted sbuf/psum buffer declared inside loops, the
    enclosing loop vars are subtracted so the index addresses the single
    resident instance (e.g. ``[i_d0_0, (i_d1_0)*128 : +128]`` → ``[0, 0:128]``).
    """
    decl_block_nid = _declaring_block(tree, buf.name)
    if decl_block_nid is None:
        return region
    anchors = _anchor_loop_vars(tree, decl_block_nid)
    if not anchors:
        return region
    subs = {a: Const(value=0) for a in anchors}
    new_ranges = tuple((substitute(lo, subs), width) for lo, width in region.ranges)
    return BufferRegion(tensor=region.tensor, ranges=new_ranges)


def _declaring_block(tree: KernelTree, tensor: str) -> int | None:
    """Return the block nid whose alloc_buffers declares ``tensor``, or None (a param)."""
    for nid in tree.blocks():
        block = tree.data(nid)
        assert isinstance(block, BlockNode)
        for buf in block.alloc_buffers:
            if buf.name == tensor:
                return nid
    return None


__all__ = ["compact_shapes", "rebased_region"]
