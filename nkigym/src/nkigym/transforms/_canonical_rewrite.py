"""Canonical graph-rewrite helpers shared by operation transforms."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from weakref import WeakKeyDictionary

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Const, Mul, Var
from nkigym.ir.buffer_placement import place_buffers
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import PARTITION_DIM, BlockNode, Buffer, BufferRegion, ForNode, ISANode, IterVar, KernelTree
from nkigym.ops.base import AxisRole, NKIOp
from nkigym.transforms._tree_ops import invalidate_stale_software_pipelines


@dataclass(frozen=True)
class CanonicalSpec:
    """Payloads for one canonical single-ISA block."""

    block: BlockNode
    loops: tuple[ForNode, ...]
    leaf: ISANode


@dataclass(frozen=True)
class _CanonicalContext:
    """Derived buffer and axis maps shared by canonical matching."""

    buffers: dict[str, Buffer]
    extents: dict[str, int]


_CONTEXTS: WeakKeyDictionary[KernelTree, _CanonicalContext] = WeakKeyDictionary()
_SINGLE_LEAVES: WeakKeyDictionary[KernelTree, dict[int, int | None]] = WeakKeyDictionary()


def _canonical_context(ir: KernelIR) -> _CanonicalContext:
    """Return cached canonical-matching facts for the current tree."""
    context = _CONTEXTS.get(ir.tree)
    if context is None:
        context = _CanonicalContext(buffers=ir.all_buffers(), extents=axis_extents(ir))
        _CONTEXTS[ir.tree] = context
    return context


def _invalidate_canonical_context(ir: KernelIR) -> None:
    """Discard derived canonical facts after mutating a tree."""
    _CONTEXTS.pop(ir.tree, None)
    _SINGLE_LEAVES.pop(ir.tree, None)


def single_leaf(tree: KernelTree, block_nid: int) -> int | None:
    """Return the sole ISA leaf owned by ``block_nid``."""
    cache = _SINGLE_LEAVES.setdefault(tree, {})
    if block_nid in cache:
        result = cache[block_nid]
    else:
        result = None
        if block_nid in tree.graph and isinstance(tree.data(block_nid), BlockNode):
            descendants = tree.descendants(block_nid)
            leaves = [
                nid
                for nid in descendants
                if isinstance(tree.data(nid), ISANode)
                and not any(
                    isinstance(tree.data(ancestor), BlockNode) and ancestor != block_nid
                    for ancestor in tree.ancestors(nid)
                    if ancestor in descendants
                )
            ]
            if len(leaves) == 1:
                result = leaves[0]
        cache[block_nid] = result
    return result


def owning_block(tree: KernelTree, leaf_nid: int) -> int:
    """Return the nearest block that owns ``leaf_nid``."""
    result: int | None = None
    for ancestor in reversed(tree.ancestors(leaf_nid)):
        if isinstance(tree.data(ancestor), BlockNode):
            result = ancestor
            break
    if result is None:
        raise ValueError(f"ISA leaf {leaf_nid} has no owning block")
    return result


def is_canonical_block(ir: KernelIR, block_nid: int) -> bool:
    """Return whether ``block_nid`` exactly matches canonical construction."""
    result = False
    leaf_nid = single_leaf(ir.tree, block_nid)
    if leaf_nid is not None:
        leaf = ir.tree.isa(leaf_nid)
        operand_names = {slot: region.tensor for slot, region in leaf.operand_bindings.items()}
        spec = canonical_spec(
            ir, leaf.op_cls, operand_names, ir.tree.block(block_nid).axis_map, leaf.kwargs, _canonical_context(ir)
        )
        chain = block_chain(ir.tree, block_nid)
        if spec is not None and chain is not None:
            block = replace(ir.tree.block(block_nid), alloc_buffers=())
            result = (block, *chain[1:]) == (spec.block, *spec.loops, spec.leaf)
    return result


def block_chain(tree: KernelTree, block_nid: int) -> tuple[BlockNode | ForNode | ISANode, ...] | None:
    """Return one unbranched block-to-ISA payload chain."""
    result: tuple[BlockNode | ForNode | ISANode, ...] | None = None
    if block_nid in tree.graph and isinstance(tree.data(block_nid), BlockNode):
        payloads: list[BlockNode | ForNode | ISANode] = [tree.block(block_nid)]
        current = block_nid
        complete = False
        valid = True
        while valid and not complete:
            children = tree.children(current)
            valid = len(children) == 1
            if valid:
                current = children[0]
                payload = tree.data(current)
                valid = not isinstance(payload, BlockNode)
                if valid:
                    payloads.append(payload)
                    complete = isinstance(payload, ISANode)
        if valid and complete:
            result = tuple(payloads)
    return result


def axis_extents(ir: KernelIR) -> dict[str, int]:
    """Collect concrete axis extents, requiring declarations to agree."""
    extents: dict[str, int] = {}
    for block_nid in ir.tree.blocks():
        for iter_var in ir.tree.block(block_nid).iter_vars:
            extent = iter_var.dom[1] - iter_var.dom[0]
            prior = extents.get(iter_var.axis)
            if prior is not None and prior != extent:
                raise ValueError(f"axis {iter_var.axis} has conflicting extents {prior} and {extent}")
            extents[iter_var.axis] = extent
    return extents


def canonical_spec(
    ir: KernelIR,
    op_cls: type[NKIOp],
    operand_names: dict[str, str],
    axis_map: dict[str, str],
    kwargs: dict[str, Any],
    context: _CanonicalContext | None = None,
) -> CanonicalSpec | None:
    """Build canonical payloads for one operation."""
    resolved = context if context is not None else _canonical_context(ir)
    extents = resolved.extents
    buffers = resolved.buffers
    valid = all(name in buffers for name in operand_names.values())
    valid = valid and all(
        abstract in axis_map and axis_map[abstract] in extents
        for slot, axes in op_cls.OPERAND_AXES.items()
        if slot in operand_names
        for abstract in axes[: len(buffers[operand_names[slot]].shape)]
    )
    tiles: dict[str, int] = {}
    if valid:
        for abstract, concrete in axis_map.items():
            extent = extents[concrete]
            minimum = op_cls.MIN_TILE_SIZE.get(abstract, 1)
            maximum = op_cls.MAX_TILE_SIZE.get(abstract)
            tile = extent if maximum is None else min(extent, maximum)
            if tile <= 0 or extent < minimum or extent % tile != 0:
                valid = False
                break
            tiles[abstract] = tile

    result: CanonicalSpec | None = None
    if valid:
        iter_vars: list[IterVar] = []
        iter_values: list[Const | Var] = []
        loops: list[ForNode] = []
        loop_vars: dict[str, str] = {}
        for abstract, concrete in axis_map.items():
            extent = extents[concrete]
            trip = extent // tiles[abstract]
            loop_var = f"i_{concrete}_0"
            loop_vars[abstract] = loop_var
            iter_vars.append(
                IterVar(axis=concrete, dom=(0, extent), role=op_cls.AXIS_ROLES.get(abstract, AxisRole.PARALLEL))
            )
            iter_values.append(Var(name=loop_var) if trip > 1 else Const(value=0))
            if trip > 1:
                loops.append(ForNode(loop_var=loop_var, extent=trip))

        bindings = {
            slot: _canonical_region(
                tensor=operand_names[slot],
                axes=axes,
                axis_map=axis_map,
                loop_vars=loop_vars,
                tiles=tiles,
                extents=extents,
                buffers=buffers,
            )
            for slot, axes in op_cls.OPERAND_AXES.items()
            if slot in operand_names
        }
        reads: list[BufferRegion] = []
        writes: list[BufferRegion] = []
        rmw_operands = op_cls.rmw_operands(kwargs)
        for slot, region in bindings.items():
            if slot in op_cls.INPUT_OPERANDS:
                reads.append(region)
            elif slot in rmw_operands:
                reads.append(region)
                writes.append(region)
            else:
                writes.append(region)
        result = CanonicalSpec(
            block=BlockNode(
                iter_vars=tuple(iter_vars),
                iter_values=tuple(iter_values),
                reads=tuple(reads),
                writes=tuple(writes),
                alloc_buffers=(),
                axis_map=dict(axis_map),
            ),
            loops=tuple(loops),
            leaf=ISANode(op_cls=op_cls, operand_bindings=bindings, kwargs=dict(kwargs)),
        )
    return result


def required_spec(
    ir: KernelIR, op_cls: type[NKIOp], operand_names: dict[str, str], axis_map: dict[str, str], kwargs: dict[str, Any]
) -> CanonicalSpec:
    """Return a canonical specification or fail on malformed rewrite input."""
    spec = canonical_spec(ir, op_cls, operand_names, axis_map, kwargs)
    if spec is None:
        raise AssertionError(f"could not construct canonical {op_cls.__name__} block")
    return spec


def append_block(tree: KernelTree, spec: CanonicalSpec) -> int:
    """Append one detached canonical block and return its node id."""
    block_nid = tree.add_node(spec.block)
    parent = block_nid
    for loop in spec.loops:
        parent = tree.add_node(loop, parent=parent)
    tree.add_node(spec.leaf, parent=parent)
    return block_nid


def rewrite_block(tree: KernelTree, block_nid: int, spec: CanonicalSpec) -> None:
    """Replace a block's local subtree while retaining its node id."""
    descendants = list(tree.descendants(block_nid))
    tree.graph.remove_nodes_from(descendants)
    tree.graph.nodes[block_nid]["data"] = spec.block
    parent = block_nid
    for loop in spec.loops:
        parent = tree.add_node(loop, parent=parent)
    tree.add_node(spec.leaf, parent=parent)


def append_root_buffers(ir: KernelIR, buffers: tuple[Buffer, ...]) -> None:
    """Append new buffers to the root before placement recomputation."""
    _invalidate_canonical_context(ir)
    root = ir.tree.block(ir.tree.root)
    ir.tree.graph.nodes[ir.tree.root]["data"] = replace(root, alloc_buffers=(*root.alloc_buffers, *buffers))


def replace_buffer(ir: KernelIR, replacement: Buffer) -> None:
    """Replace one declared buffer by name."""
    _invalidate_canonical_context(ir)
    found = 0
    for block_nid in ir.tree.blocks():
        block = ir.tree.block(block_nid)
        updated = tuple(replacement if buffer.name == replacement.name else buffer for buffer in block.alloc_buffers)
        if updated != block.alloc_buffers:
            ir.tree.graph.nodes[block_nid]["data"] = replace(block, alloc_buffers=updated)
            found += 1
    if found != 1:
        raise AssertionError(f"expected one declaration of {replacement.name!r}, found {found}")


def remove_buffers(ir: KernelIR, names: set[str]) -> None:
    """Remove temporary declarations named by ``names``."""
    _invalidate_canonical_context(ir)
    removed: set[str] = set()
    for block_nid in ir.tree.blocks():
        block = ir.tree.block(block_nid)
        retained = tuple(buffer for buffer in block.alloc_buffers if buffer.name not in names)
        removed.update(buffer.name for buffer in block.alloc_buffers if buffer.name in names)
        if retained != block.alloc_buffers:
            ir.tree.graph.nodes[block_nid]["data"] = replace(block, alloc_buffers=retained)
    if removed != names:
        raise AssertionError(f"expected to remove buffers {names}, removed {removed}")


def fresh_name(ir: KernelIR, stem: str) -> str:
    """Return a deterministic buffer name not present in ``ir``."""
    names = set(ir.all_buffers())
    candidate = stem
    suffix = 1
    while candidate in names:
        candidate = f"{stem}_{suffix}"
        suffix += 1
    return candidate


def replace_input_binding(ir: KernelIR, leaf_nid: int, operand: str, tensor: str) -> None:
    """Rebind one input operand and its owning block read region."""
    leaf = ir.tree.isa(leaf_nid)
    if operand not in leaf.op_cls.INPUT_OPERANDS or operand not in leaf.operand_bindings:
        raise ValueError(f"{leaf.op_cls.__name__}.{operand} is not a bound input operand")
    old_region = leaf.operand_bindings[operand]
    new_region = replace(old_region, tensor=tensor)
    bindings = dict(leaf.operand_bindings)
    bindings[operand] = new_region
    ir.tree.graph.nodes[leaf_nid]["data"] = replace(leaf, operand_bindings=bindings)

    block_nid = owning_block(ir.tree, leaf_nid)
    block = ir.tree.block(block_nid)
    replaced = False
    reads: list[BufferRegion] = []
    for region in block.reads:
        if region == old_region and not replaced:
            reads.append(new_region)
            replaced = True
        else:
            reads.append(region)
    if not replaced:
        raise AssertionError(
            f"expected a {old_region.tensor} read in block {block_nid} for {leaf.op_cls.__name__}.{operand}"
        )
    ir.tree.graph.nodes[block_nid]["data"] = replace(block, reads=tuple(reads))


def finalize_rewrite(ir: KernelIR) -> None:
    """Recompute buffer placement and dependencies after a graph rewrite."""
    _invalidate_canonical_context(ir)
    invalidate_stale_software_pipelines(ir)
    place_buffers(ir.tree)
    ir.dependency = Dependency(ir.tree)


def _canonical_region(
    *,
    tensor: str,
    axes: tuple[str, ...],
    axis_map: dict[str, str],
    loop_vars: dict[str, str],
    tiles: dict[str, int],
    extents: dict[str, int],
    buffers: dict[str, Buffer],
) -> BufferRegion:
    """Build one canonical operand region."""
    ranges: list[tuple[Const | Var | Mul, Const]] = []
    buffer = buffers[tensor]
    present_axes = tuple(axis for axis in axes if axis in axis_map)
    for axis_index, abstract in enumerate(present_axes):
        concrete = axis_map[abstract]
        tile = tiles[abstract]
        trip = extents[concrete] // tile
        if trip == 1:
            lo: Const | Var | Mul = Const(value=0)
        elif axis_index == 0 and buffer.location in {"sbuf", "psum"} and tile == PARTITION_DIM:
            lo = Var(name=loop_vars[abstract])
        else:
            lo = Mul(left=Var(name=loop_vars[abstract]), right=Const(value=tile))
        ranges.append((lo, Const(value=tile)))
    return BufferRegion(tensor=tensor, ranges=tuple(ranges))


__all__ = [
    "CanonicalSpec",
    "append_block",
    "append_root_buffers",
    "axis_extents",
    "block_chain",
    "canonical_spec",
    "finalize_rewrite",
    "fresh_name",
    "is_canonical_block",
    "owning_block",
    "remove_buffers",
    "replace_buffer",
    "replace_input_binding",
    "required_spec",
    "rewrite_block",
    "single_leaf",
]
