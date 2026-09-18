"""Canonical :class:`BlockNode`-rooted tree construction.

Consumes the private :class:`_AnalysisResult` produced by
:func:`nkigym.ir.dimension_analysis.analyze_dimensions` and emits a
fully-shaped :class:`KernelTree` whose root is a :class:`BlockNode`
(empty iter_vars/reads/writes) containing one leaf ``BlockNode`` per
non-alloc op, in source order.

Every compute op (including memset) becomes one leaf block in source order.
Explicit repeated bodies group those blocks under sequential loop scopes.

Buffer placement is delegated to
:func:`nkigym.ir.buffer_placement.place_buffers`.
"""

from __future__ import annotations

from dataclasses import replace

from nkigym.ir.arith.expr import Const, Var
from nkigym.ir.dimension_analysis import _AnalysisResult, _OpRecord
from nkigym.ir.operand_layout import (
    build_access_patterns,
    build_operand_region,
    canonical_tile_size,
    canonical_trip_count,
)
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode, IterVar, KernelTree
from nkigym.ops.base import AxisRole
from nkigym.ops.memset import NKIMemset


def build_canonical_blocknode_tree(analysis: "_AnalysisResult") -> KernelTree:
    """Build the canonical :class:`BlockNode`-rooted tree.

    Tree.root is already an empty BlockNode from KernelTree.__init__.
    Build leaf blocks, wrap explicit repetitions, seed all Buffers on the root,
    then use LCA placement to distribute their lifetime-dominating declarations.
    """
    from nkigym.ir.buffer_placement import collect_buffers, place_buffers

    tree = KernelTree()
    groups: list[list[int]] = []
    for rec in analysis.ops:
        group = []
        if rec.op_cls.SYNTHESIZE_RMW_INITIALIZER and rec.op_cls.rmw_operands(rec.kwargs):
            group.append(_build_memset_subblock(tree, tree.root, rec, analysis))
        group.append(_build_subblock(tree, tree.root, rec, analysis))
        groups.append(group)
    _wrap_repetitions(tree, analysis, groups)
    buffers_by_name = collect_buffers(analysis.tensors, analysis.param_names, tree)
    """Seed every Buffer on the root block, then let place_buffers redistribute by LCA."""
    root_blk = tree.data(tree.root)
    tree.graph.nodes[tree.root]["data"] = replace(root_blk, alloc_buffers=tuple(buffers_by_name.values()))
    place_buffers(tree)
    return tree


def _wrap_repetitions(tree: KernelTree, analysis: _AnalysisResult, groups: list[list[int]]) -> None:
    """Wrap traced operation intervals in ordinary sequential loop scopes."""
    for first, last, count in analysis.repetitions:
        if count == 1:
            continue
        selected = [nid for group in groups[first:last] for nid in group]
        children = tree.children(tree.root)
        position = children.index(selected[0])
        if children[position : position + len(selected)] != selected:
            raise ValueError("repeated operation intervals must be properly nested")
        if isinstance(count, tuple):
            region = BufferRegion(tensor=count[0], ranges=((Const(value=0), Const(value=1)),) * 2)
            body = tree.add_node(replace(tree.block(tree.root), reads=(region,), annotations={"predicate": count}))
            replacement = body
        else:
            axis = f"d{len(analysis.dim_sizes)}"
            analysis.dim_sizes[axis] = count
            loop_var = f"i_{axis}_0"
            replacement = tree.add_node(ForNode(loop_var=loop_var, extent=count))
            body = tree.add_node(
                replace(
                    tree.block(tree.root),
                    iter_vars=(IterVar(axis=axis, dom=(0, count), role=AxisRole.SEQUENTIAL),),
                    iter_values=(Var(name=loop_var),),
                    axis_map={"R": axis},
                ),
                parent=replacement,
            )
        children[position : position + len(selected)] = [replacement]
        tree.graph.remove_edges_from((tree.root, nid) for nid in tree.children(tree.root))
        tree.graph.add_edges_from((tree.root, nid) for nid in children)
        tree.graph.add_edges_from((body, nid) for nid in selected)
        groups[first:last] = [[replacement], *([] for _ in range(last - first - 1))]


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
        iter_values.append(Var(name=loop_var) if canonical_trip_count(rec, abstract, analysis) > 1 else Const(value=0))
    operand_bindings = {
        slot: build_operand_region(rec, slot, loop_var_names, analysis, canonical_tile_size, canonical_trip_count)
        for slot in rec.op_cls.OPERAND_AXES
        if slot in rec.operand_names
    }
    rmw_operands = rec.op_cls.rmw_operands(rec.kwargs)
    block = BlockNode(
        iter_vars=tuple(iter_vars),
        iter_values=tuple(iter_values),
        reads=tuple(
            region
            for slot, region in operand_bindings.items()
            if slot in rec.op_cls.INPUT_OPERANDS or slot in rmw_operands
        ),
        writes=tuple(region for slot, region in operand_bindings.items() if slot not in rec.op_cls.INPUT_OPERANDS),
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

    The accumulator's physical dimension IDs are mapped onto memset's
    ``(P, F)`` axes, preserving the complete extent of packed axis groups.
    """
    rmw_slot = next(iter(rec.op_cls.rmw_operands(rec.kwargs)))
    memset_concrete = analysis.tensors[rec.operand_names[rmw_slot]].dim_ids
    memset_axis_map = {abstract: concrete for abstract, concrete in zip(NKIMemset.OPERAND_AXES["dst"], memset_concrete)}
    memset_rec = _OpRecord(
        op_cls=NKIMemset,
        operand_names={"dst": rec.operand_names[rmw_slot]},
        axis_map=memset_axis_map,
        kwargs={"value": 0.0},
    )
    return _build_subblock(tree, parent_nid, memset_rec, analysis)
