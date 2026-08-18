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
from math import gcd, lcm

from nkigym.ir.arith.expr import Const, to_affine
from nkigym.ir.dimension_analysis import TensorDims
from nkigym.ir.graph_index import ordered_tree_topology
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode, KernelTree, partition_extent

_STORAGE_DTYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "float8_e4m3": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int8": 1,
    "int32": 4,
    "tfloat32": 4,
    "uint8": 1,
    "uint32": 4,
}


def layout_satisfies_output_alignment(tree: KernelTree, buffer: Buffer) -> bool:
    """Return whether packed tiles satisfy every producer output alignment."""
    alignment = 1
    for nid in tree.preorder():
        node = tree.data(nid)
        if isinstance(node, ISANode):
            for slot, required in node.op_cls.OUTPUT_TILE_ALIGNMENT_BYTES.items():
                region = node.operand_bindings.get(slot)
                if region is not None and region.tensor == buffer.name:
                    alignment = lcm(alignment, required)
    return layout_satisfies_alignment(buffer, alignment)


def layout_satisfies_alignment(buffer: Buffer, alignment: int) -> bool:
    """Return whether packed physical tiles satisfy one byte alignment."""
    if alignment == 1 or buffer.tiles_per_list() * buffer.versions == 1:
        return True
    dtype = buffer.physical_dtype()
    if dtype not in _STORAGE_DTYPE_BYTES:
        raise AssertionError(f"{buffer.name}: unsupported physical dtype {dtype!r}")
    return buffer.physical_shape()[2] * _STORAGE_DTYPE_BYTES[dtype] % alignment == 0


def collect_buffers(tensors: dict[str, TensorDims], param_names: list[str], tree: KernelTree) -> dict[str, Buffer]:
    """Return one allocation per intermediate tensor with a consistent partition width."""
    out: dict[str, Buffer] = {}
    for name, tensor in tensors.items():
        if name in param_names:
            continue
        widths = {
            width.value
            for block_nid in tree.blocks()
            for region in (*tree.block(block_nid).reads, *tree.block(block_nid).writes)
            if region.tensor == name
            for _lower, width in region.ranges[:1]
            if isinstance(width, Const)
        }
        partition_size = None
        if tensor.location != "shared_hbm":
            partition_size = partition_extent(tensor.shape[0])
            for width in widths:
                partition_size = gcd(partition_size, width)
        buffer = Buffer(
            name=name,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            location=tensor.location,
            storage_dtype=tensor.storage_dtype,
            partition_size=partition_size,
        )
        if buffer.location != "shared_hbm" and not layout_satisfies_output_alignment(tree, buffer):
            buffer = replace(buffer, list_len=buffer.logical_tile_count())
        if buffer.location != "shared_hbm" and not layout_satisfies_output_alignment(tree, buffer):
            raise ValueError(f"{name}: no canonical allocation satisfies producer output alignment")
        out[name] = buffer
    return out


def place_buffers(tree: KernelTree, anchor_loop_nids_by_tensor: Mapping[str, frozenset[int]] | None = None) -> None:
    """Recompute and apply lifetime-safe LCA placement in place.

    Gathers all Buffers from existing ``alloc_buffers``, clears them,
    then re-attaches each at or above the LCA block of its touchers. Buffers
    remain in first-seen order so each ``alloc_buffers`` list is deterministic.
    """
    buffers = _gather_buffers(tree)
    targets = buffer_placement_targets(tree, tuple(buffers), anchor_loop_nids_by_tensor)
    _clear_alloc_buffers(tree)
    placement: dict[int, list[Buffer]] = {}
    for name, buf in buffers.items():
        placement.setdefault(targets[name], []).append(buf)
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
    anchors = {tensor: anchor_loop_nids} if anchor_loop_nids is not None else None
    target_nid = buffer_placement_targets(tree, (tensor,), anchors)[tensor]
    if source_nid != target_nid:
        source = tree.block(source_nid)
        remaining = tuple(candidate for candidate in source.alloc_buffers if candidate.name != tensor)
        tree.graph.nodes[source_nid]["data"] = replace(source, alloc_buffers=remaining)

        order = {name: index for index, name in enumerate(buffers)}
        target = tree.block(target_nid)
        placed = tuple(sorted((*target.alloc_buffers, buf), key=lambda candidate: order[candidate.name]))
        tree.graph.nodes[target_nid]["data"] = replace(target, alloc_buffers=placed)


def buffer_placement_targets(
    tree: KernelTree, tensors: tuple[str, ...], anchor_loop_nids_by_tensor: Mapping[str, frozenset[int]] | None = None
) -> dict[str, int]:
    """Return lifetime-safe declaration blocks for selected tensors."""
    buffers = _gather_buffers(tree)
    missing = set(tensors) - buffers.keys()
    if missing:
        raise KeyError(f"buffers declared by no block: {sorted(missing)}")
    touchers = _touchers_by_tensor(tree)
    regions = _regions_by_tensor(tree, frozenset(tensors))
    _order, ancestors, _descendants = ordered_tree_topology(tree.graph, tree.root)
    targets: dict[str, int] = {}
    for tensor in tensors:
        buffer = buffers[tensor]
        if buffer.location == "shared_hbm":
            target = tree.root
        else:
            touch = touchers.get(tensor)
            lca = tree.root if not touch else _lca(touch, ancestors)
            anchors = anchor_loop_nids_by_tensor.get(tensor) if anchor_loop_nids_by_tensor is not None else None
            if anchors is None:
                anchors = _anchor_loop_nids_from_regions(tree, regions.get(tensor, []), ancestors)
            target = _safe_enclosing_block(tree, lca, anchors, ancestors)
        targets[tensor] = target
    return targets


def _safe_enclosing_block(
    tree: KernelTree, lca_nid: int, anchor_loop_nids: frozenset[int], ancestors: dict[int, tuple[int, ...]]
) -> int:
    """Return the deepest block at or above every non-anchor ancestor loop."""
    target_nid = _enclosing_block(tree, lca_nid, ancestors)
    for ancestor in ancestors[target_nid]:
        if isinstance(tree.data(ancestor), ForNode) and ancestor not in anchor_loop_nids:
            target_nid = _enclosing_block(tree, ancestor, ancestors)
            break
    return target_nid


def _declaring_block(tree: KernelTree, tensor: str) -> int:
    """Return the unique block that declares ``tensor``."""
    owners = [nid for nid in tree.blocks() if any(buffer.name == tensor for buffer in tree.block(nid).alloc_buffers)]
    if len(owners) != 1:
        raise AssertionError(f"buffer {tensor!r} must have exactly one declaration, found {owners}")
    return owners[0]


def _enclosing_block(tree: KernelTree, nid: int, ancestors: dict[int, tuple[int, ...]]) -> int:
    """Return ``nid`` if it is a BlockNode, else its nearest BlockNode ancestor.

    A buffer is declared on a block, but the LCA of its touchers can be a
    ForNode when two co-located blocks share an enclosing loop (e.g. a store
    lifted next to its tensor_copy under a shared d2 loop). The owning block is
    then the nearest BlockNode at or above that loop.
    """
    blocks = (
        candidate for candidate in reversed((*ancestors[nid], nid)) if isinstance(tree.data(candidate), BlockNode)
    )
    return next(blocks)


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


def _anchor_loop_nids_from_regions(
    tree: KernelTree, pairs: list[tuple[int, BufferRegion]], ancestors: dict[int, tuple[int, ...]]
) -> frozenset[int]:
    """Return common outer loops that select one reusable buffer instance."""
    anchors: set[int] = set()
    if pairs:
        per_leaf = [
            {ancestor for ancestor in ancestors[leaf] if isinstance(tree.data(ancestor), ForNode)}
            for leaf, _region in pairs
        ]
        common = set.intersection(*per_leaf)
        candidates = [nid for nid in ancestors[pairs[0][0]] if nid in common]
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


def _regions_by_tensor(tree: KernelTree, tensors: frozenset[str]) -> dict[str, list[tuple[int, BufferRegion]]]:
    """Index selected ISA operand regions by tensor in one traversal."""
    regions: dict[str, list[tuple[int, BufferRegion]]] = {}
    for nid in tree.preorder():
        data = tree.data(nid)
        if isinstance(data, ISANode):
            for region in data.operand_bindings.values():
                if region.tensor in tensors:
                    regions.setdefault(region.tensor, []).append((nid, region))
    return regions


def _lca(nids: set[int], ancestors: dict[int, tuple[int, ...]]) -> int:
    """Lowest common ancestor of ``nids`` (deepest common ancestor).

    For a single-element set, returns that element.
    """
    if len(nids) == 1:
        return next(iter(nids))
    paths = [(*ancestors[nid], nid) for nid in nids]
    return next(level[0] for level in reversed(tuple(zip(*paths))) if len(set(level)) == 1)


__all__ = [
    "buffer_placement_targets",
    "collect_buffers",
    "layout_satisfies_alignment",
    "layout_satisfies_output_alignment",
    "place_buffer",
    "place_buffers",
]
