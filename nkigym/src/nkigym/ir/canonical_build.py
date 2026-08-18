"""Canonical :class:`BlockNode`-rooted tree construction.

Consumes the private :class:`_AnalysisResult` produced by
:func:`nkigym.ir.dimension_analysis.analyze_dimensions` and emits a
fully-shaped :class:`KernelTree` whose root is a :class:`BlockNode`
(empty iter_vars/reads/writes) containing one leaf ``BlockNode`` per
non-alloc op, in source order.

Every compute op (including memset) becomes a sibling leaf block under
the root block, preserving source order.

Buffer placement is delegated to
:func:`nkigym.search.buffer_placement.place_buffers`.
"""

from __future__ import annotations

from dataclasses import replace

from nkigym.ir.arith.expr import Const, Var
from nkigym.ir.dimension_analysis import _AnalysisResult, _OpRecord
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode, IterVar, KernelTree
from nkigym.ops.base import AxisRole
from nkigym.ops.memset import NKIMemset


def build_canonical_blocknode_tree(analysis: "_AnalysisResult") -> KernelTree:
    """Build the canonical :class:`BlockNode`-rooted tree.

    Tree.root is already an empty BlockNode from KernelTree.__init__.
    Build leaf blocks under it, seed all Buffers on the root, then run
    LCA placement to distribute them to their lifetime-dominating blocks.
    """
    from nkigym.search.buffer_placement import collect_buffers, place_buffers

    tree = KernelTree()
    op_records = list(analysis.ops)
    for rec in op_records:
        if rec.op_cls.SYNTHESIZE_RMW_INITIALIZER and rec.op_cls.rmw_operands(rec.kwargs):
            _build_memset_subblock(tree, tree.root, rec, analysis)
        _build_subblock(tree, tree.root, rec, analysis)
    buffers_by_name = collect_buffers(analysis.tensors, analysis.param_names, tree)
    """Seed every Buffer on the root block, then let place_buffers redistribute by LCA."""
    root_blk = tree.data(tree.root)
    tree.graph.nodes[tree.root]["data"] = replace(root_blk, alloc_buffers=tuple(buffers_by_name.values()))
    place_buffers(tree)
    return tree


def _build_subblock(tree: KernelTree, parent_nid: int, rec: "_OpRecord", analysis: "_AnalysisResult") -> int:
    """Construct one :class:`BlockNode` + its loop chain + ISA leaf; return the block's nid."""
    from nkigym.search.axis_groups import build_access_patterns, canonical_tile_size, canonical_trip_count

    iter_vars: list[IterVar] = []
    iter_values: list = []
    loop_var_names: dict[str, str] = {}
    for abstract, concrete in rec.axis_map.items():
        extent = analysis.dim_sizes[concrete]
        role = rec.op_cls.AXIS_ROLES.get(abstract, AxisRole.PARALLEL)
        iter_vars.append(IterVar(axis=concrete, dom=(0, extent), role=role))
        loop_var = f"i_{concrete}_0"
        loop_var_names[abstract] = loop_var
        if canonical_trip_count(rec, abstract, analysis) > 1:
            iter_values.append(Var(name=loop_var))
        else:
            iter_values.append(Const(value=0))
    reads, writes = _operand_regions(rec, loop_var_names, analysis)
    block = BlockNode(
        iter_vars=tuple(iter_vars),
        iter_values=tuple(iter_values),
        reads=tuple(reads),
        writes=tuple(writes),
        alloc_buffers=(),
        axis_map=dict(rec.axis_map),
    )
    block_nid = tree.add_node(block, parent=parent_nid)
    parent_for_loops: int = block_nid
    for abstract, concrete in rec.axis_map.items():
        trip = canonical_trip_count(rec, abstract, analysis)
        if trip > 1:
            loop_var = loop_var_names[abstract]
            for_nid = tree.add_node(ForNode(loop_var=loop_var, extent=trip), parent=parent_for_loops)
            parent_for_loops = for_nid
    operand_bindings = _operand_bindings(rec, loop_var_names, analysis)
    access_patterns = build_access_patterns(rec, loop_var_names, analysis, canonical_tile_size, canonical_trip_count)
    op_kwargs = dict(rec.kwargs)
    for abstract, (key, slot) in getattr(rec.op_cls, "SPLIT_OFFSET_KWARGS", {}).items():
        groups = rec.op_cls.operand_axis_groups(slot)
        dimension = next(i for i, group in enumerate(groups) if abstract in group)
        op_kwargs[key] = operand_bindings[slot].ranges[dimension][0]
    if rec.op_cls.FIRST_WRITE_AXES:
        op_kwargs["accumulate"] = rec.op_cls.FIRST_WRITE_AXES
    tree.add_node(
        ISANode(
            op_cls=rec.op_cls, operand_bindings=operand_bindings, kwargs=op_kwargs, access_patterns=access_patterns
        ),
        parent=parent_for_loops,
    )
    return block_nid


def _build_memset_subblock(tree: KernelTree, parent_nid: int, rec: "_OpRecord", analysis: "_AnalysisResult") -> int:
    """Synthesize a memset sibling block zeroing the RMW (accumulator) operand of ``rec``.

    Emitted immediately before the RMW op's own block, mirroring the
    decomposed-canonical form (memset is a sibling, not a nested init).
    The dependency edge falls out by sibling pre-order: memset writes the
    PSUM region, the matmul RMW-reads+writes it (WAW/RAW after memset).

    ``rec``'s RMW slot axes (e.g. matmul dst ``(M, N)``) are remapped onto
    memset's own abstract axes ``(P, F)`` positionally, so the synthesized
    record renders correctly against the PSUM tensor.
    """
    rmw_slot = next(iter(rec.op_cls.rmw_operands(rec.kwargs)))
    rmw_axes = rec.op_cls.OPERAND_AXES[rmw_slot]
    memset_concrete = [rec.axis_map[a] for a in rmw_axes if a in rec.axis_map]
    memset_axis_map = {abstract: concrete for abstract, concrete in zip(NKIMemset.OPERAND_AXES["dst"], memset_concrete)}
    memset_rec = _OpRecord(
        op_cls=NKIMemset,
        operand_names={"dst": rec.operand_names[rmw_slot]},
        axis_map=memset_axis_map,
        kwargs={"value": 0.0},
    )
    return _build_subblock(tree, parent_nid, memset_rec, analysis)


def _operand_regions(
    rec: "_OpRecord", loop_var_names: dict[str, str], analysis: "_AnalysisResult"
) -> tuple[list[BufferRegion], list[BufferRegion]]:
    """Build (reads, writes) BufferRegion lists from ``rec.operand_names`` and OPERAND_AXES."""
    reads: list[BufferRegion] = []
    writes: list[BufferRegion] = []
    rmw_operands = rec.op_cls.rmw_operands(rec.kwargs)
    for slot, axes in rec.op_cls.OPERAND_AXES.items():
        if slot not in rec.operand_names:
            continue
        region = _build_region(rec, slot, axes, loop_var_names, analysis)
        if slot in rec.op_cls.INPUT_OPERANDS:
            reads.append(region)
        elif slot in rmw_operands:
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
    """Construct one dependency region from the operation's physical axis groups."""
    from nkigym.search.axis_groups import build_operand_region, canonical_tile_size, canonical_trip_count

    _ = axes
    return build_operand_region(rec, slot, loop_var_names, analysis, canonical_tile_size, canonical_trip_count)


__all__ = ["build_canonical_blocknode_tree"]
