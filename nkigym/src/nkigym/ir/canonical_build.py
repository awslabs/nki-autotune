"""Canonical :class:`BlockNode`-rooted tree construction.

Consumes the private :class:`_AnalysisResult` produced by
:func:`nkigym.ir.dimension_analysis.analyze_dimensions` and emits a
fully-shaped :class:`KernelTree` whose root is a :class:`BlockNode`
(empty iter_vars/reads/writes) containing one leaf ``BlockNode`` per
non-alloc op, in source order.

Every compute op (including memset) becomes a sibling leaf block under
the root block, preserving source order.

Buffer placement: each :class:`Buffer` is placed on the lowest common
ancestor (LCA) block of all blocks that read or write it. For canonical
IR (every op is a sibling under the root), this resolves to the root
block whenever a buffer is touched by multiple leaf blocks.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from nkigym.ir.expr import Const, Var
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode, IterVar, KernelTree
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.base import AxisRole

if TYPE_CHECKING:
    from nkigym.ir.dimension_analysis import TensorDims, _AnalysisResult, _OpRecord


def build_canonical_blocknode_tree(analysis: "_AnalysisResult") -> KernelTree:
    """Build the canonical :class:`BlockNode`-rooted tree.

    Tree.root is already an empty BlockNode from KernelTree.__init__.
    We build leaf blocks under it, compute LCA buffer placement, then
    attach buffers to their placement blocks.
    """
    tree = KernelTree()
    op_records = list(analysis.ops)
    buffers_by_name = _collect_buffers(analysis.tensors, analysis.param_names)
    compute_records = [rec for rec in op_records if rec.op_cls is not NKIAlloc]

    """Build leaf blocks under tree.root, tracking which block touches which buffer."""
    touchers: dict[str, set[int]] = {}
    for rec in compute_records:
        block_nid = _build_subblock(tree, tree.root, rec, analysis)
        """Collect tensor names from reads and writes of this block."""
        touched = set()
        blk = tree.data(block_nid)
        for region in blk.reads:
            if region.tensor not in analysis.param_names:
                touched.add(region.tensor)
        for region in blk.writes:
            if region.tensor not in analysis.param_names:
                touched.add(region.tensor)
        for tname in touched:
            touchers.setdefault(tname, set()).add(block_nid)

    """Place buffers at their LCA block."""
    placement: dict[int, list[Buffer]] = {}
    for name, buf in buffers_by_name.items():
        if name not in touchers:
            """Buffer not touched by any block — place at root (shouldn't happen in canonical)."""
            lca = tree.root
        else:
            lca = _lca(tree, touchers[name])
        placement.setdefault(lca, []).append(buf)

    """Write alloc_buffers onto each placement block (replace frozen BlockNode payload)."""
    for block_nid, bufs in placement.items():
        blk = tree.data(block_nid)
        tree.graph.nodes[block_nid]["data"] = replace(blk, alloc_buffers=tuple(bufs))

    return tree


def _lca(tree: KernelTree, nids: set[int]) -> int:
    """Compute the lowest common ancestor of a set of node ids.

    Returns the deepest node that is an ancestor of all nids.
    For a single-element set, returns that element.
    """
    if len(nids) == 1:
        return next(iter(nids))

    """Collect ancestor sets for each nid (including the node itself)."""
    ancestor_sets: list[set[int]] = []
    for nid in nids:
        ancestors = set(tree.ancestors(nid))
        ancestors.add(nid)
        ancestor_sets.append(ancestors)

    """The common ancestors are the intersection of all ancestor sets."""
    common = ancestor_sets[0].intersection(*ancestor_sets[1:])

    """The LCA is the one with the longest path from root (deepest in tree)."""
    lca_nid = tree.root
    max_depth = 0
    for candidate in common:
        depth = len(list(tree.ancestors(candidate)))
        if depth >= max_depth:
            max_depth = depth
            lca_nid = candidate
    return lca_nid


def _collect_buffers(tensors: dict[str, "TensorDims"], param_names: list[str]) -> dict[str, Buffer]:
    """Return one :class:`Buffer` per allocated tensor (excluding kernel parameters)."""
    out: dict[str, Buffer] = {}
    for name, td in tensors.items():
        if name not in param_names:
            out[name] = Buffer(name=name, shape=tuple(td.shape), dtype=td.dtype, location=td.location)
    return out


def _build_subblock(tree: KernelTree, parent_nid: int, rec: "_OpRecord", analysis: "_AnalysisResult") -> int:
    """Construct one :class:`BlockNode` + its loop chain + ISA leaf; return the block's nid."""
    iter_vars: list[IterVar] = []
    iter_values: list = []
    loop_var_names: dict[str, str] = {}
    for abstract, concrete in rec.axis_map.items():
        extent = analysis.dim_sizes[concrete]
        role = rec.op_cls.AXIS_ROLES.get(abstract, AxisRole.PARALLEL)
        iter_vars.append(IterVar(axis=concrete, dom=(0, extent), role=role))
        loop_var = f"i_{concrete}_0"
        loop_var_names[abstract] = loop_var
        iter_values.append(Var(name=loop_var))
    reads, writes = _operand_regions(rec, loop_var_names, analysis)
    block = BlockNode(
        iter_vars=tuple(iter_vars),
        iter_values=tuple(iter_values),
        reads=tuple(reads),
        writes=tuple(writes),
        alloc_buffers=(),
    )
    block_nid = tree.add_node(block, parent=parent_nid)
    parent_for_loops: int = block_nid
    for abstract, concrete in rec.axis_map.items():
        extent = analysis.dim_sizes[concrete]
        max_tile = rec.op_cls.MAX_TILE_SIZE.get(abstract)
        tile = extent if max_tile is None else max_tile
        trip = extent // tile
        loop_var = loop_var_names[abstract]
        for_nid = tree.add_node(ForNode(loop_var=loop_var, extent=trip), parent=parent_for_loops)
        parent_for_loops = for_nid
    operand_bindings = _operand_bindings(rec, loop_var_names, analysis)
    op_kwargs = {k: v for k, v in rec.kwargs.items() if k not in rec.op_cls.OPERAND_AXES}
    tree.add_node(
        ISANode(op_cls=rec.op_cls, operand_bindings=operand_bindings, kwargs=op_kwargs), parent=parent_for_loops
    )
    return block_nid


def _operand_regions(
    rec: "_OpRecord", loop_var_names: dict[str, str], analysis: "_AnalysisResult"
) -> tuple[list[BufferRegion], list[BufferRegion]]:
    """Build (reads, writes) BufferRegion lists from ``rec.operand_names`` and OPERAND_AXES."""
    reads: list[BufferRegion] = []
    writes: list[BufferRegion] = []
    for slot, axes in rec.op_cls.OPERAND_AXES.items():
        if slot not in rec.operand_names:
            continue
        region = _build_region(rec, slot, axes, loop_var_names, analysis)
        if slot in rec.op_cls.INPUT_OPERANDS:
            reads.append(region)
        elif slot in rec.op_cls.RMW_OPERANDS:
            reads.append(region)
            writes.append(region)
        else:
            writes.append(region)
    return reads, writes


def _operand_bindings(
    rec: "_OpRecord", loop_var_names: dict[str, str], analysis: "_AnalysisResult"
) -> dict[str, BufferRegion]:
    """Build the per-slot :class:`BufferRegion` map for the ISA leaf."""
    out: dict[str, BufferRegion] = {}
    for slot, axes in rec.op_cls.OPERAND_AXES.items():
        if slot not in rec.operand_names:
            continue
        out[slot] = _build_region(rec, slot, axes, loop_var_names, analysis)
    return out


def _build_region(
    rec: "_OpRecord", slot: str, axes: tuple[str, ...], loop_var_names: dict[str, str], analysis: "_AnalysisResult"
) -> BufferRegion:
    """Construct a :class:`BufferRegion` for ``slot`` using its axes and per-tile widths.

    SBUF/PSUM operands use 3D layout with partition axis special: axis 0 has
    width=128, lo is the bare partition-coord Var (not multiplied).
    """
    from nkigym.ir.expr import Mul

    tensor_name = rec.operand_names[slot]
    tensor_location = analysis.tensors[tensor_name].location
    ranges: list = []
    for axis_index, abstract in enumerate(axes):
        max_tile = rec.op_cls.MAX_TILE_SIZE.get(abstract)
        if max_tile is None:
            extent_per_tile = analysis.dim_sizes[rec.axis_map[abstract]]
        else:
            extent_per_tile = max_tile
        loop_var = loop_var_names.get(abstract)

        """Partition axis (axis 0) of SBUF/PSUM operands: tile is 128, lo is bare Var (not multiplied)."""
        _PARTITION_DIM = 128
        if (
            axis_index == 0
            and tensor_location in ("sbuf", "psum")
            and extent_per_tile == _PARTITION_DIM
            and loop_var is not None
        ):
            ranges.append((Var(name=loop_var), Const(value=_PARTITION_DIM)))
        elif loop_var is None:
            ranges.append((Const(value=0), Const(value=extent_per_tile)))
        else:
            lo = Mul(left=Var(name=loop_var), right=Const(value=extent_per_tile))
            ranges.append((lo, Const(value=extent_per_tile)))
    return BufferRegion(tensor=tensor_name, ranges=tuple(ranges))


__all__ = ["build_canonical_blocknode_tree"]
