"""BlockNode-driven body emitter.

Walks the canonical / transformed schedule tree and renders each
:class:`BlockNode` as a Python source fragment. Each block emits, in
order:

1. ``nl.ndarray(...)`` declarations — one per :attr:`BlockNode.alloc_buffers`.
2. The block's body — its ``ForNode`` chain ending in one :class:`ISANode`,
   or child sub-blocks if nested.

Operand slices are rendered via :func:`render_buffer_region` from the
ISA leaf's :attr:`ISANode.operand_bindings`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import partial
from typing import Any, cast

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Add, Const, Expr, Mod, Mul, Var, _format_raw, format_expr, substitute, to_affine
from nkigym.ir.tree import AccessPattern, BlockNode, Buffer, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.search.axis_groups import access_pattern_allocation_view
from nkigym.search.program_sharding import operation_axis_iterations, operation_axis_value, program_sharded_loops

_INDENT = "    "


class _RenderIR:
    """Read-only IR view with one buffer snapshot for source emission."""

    def __init__(self, ir: KernelIR) -> None:
        """Snapshot buffers after all schedule transformations are complete."""
        self.ir = ir
        self.buffers, self.shard_loops = ir.all_buffers(), program_sharded_loops(ir)

    def __getattr__(self, name: str) -> Any:
        """Delegate immutable schedule metadata to the source IR."""
        return getattr(self.ir, name)

    def buffer(self, name: str) -> Buffer:
        """Resolve one buffer from the render-local map."""
        return self.buffers[name]


def emit_body(ir: KernelIR) -> str:
    """Emit the kernel body for the entire tree.

    The root is a BlockNode (empty iter_vars, holds kernel-lifetime buffers).
    Emit it directly at depth=1 (one indent level inside the kernel function).
    A ``{loop_nid: annotation}`` map of every ``software_pipeline`` annotation
    is built once and threaded down so a pipelined loop rotates its
    multi-version buffer accesses (see :func:`_emit_subtree`).

    Buffer declarations are placed at their tightest materialized scope
    (:func:`_alloc_emit_anchors`): each ``nl.ndarray`` is emitted immediately before
    the first child that uses it, without crossing the buffer's owning-block
    placement boundary. Within that block, offset-carrying loops hoist the
    declaration to cover its live range. The
    ``{node_nid: [Buffer, ...]}`` map is threaded down so each node emits, before
    each child, the declarations anchored to that child.
    """
    code: list[str] = []
    ir = cast(KernelIR, _RenderIR(ir))
    pipeline_map = _pipeline_loops(ir)
    emit_before = _alloc_emit_anchors(ir)
    _emit_block(
        ir,
        ir.tree.root,
        depth=1,
        code=code,
        pipeline_map=pipeline_map,
        rotations={},
        substitutions={},
        emit_before=emit_before,
    )
    return "\n".join(code) + "\n"


def _alloc_emit_anchors(ir: KernelIR) -> dict[int, list[Buffer]]:
    """Map each tree node to the buffers emitted immediately before it.

    Scratch-buffer scope starts from the touchers' LCA, constrained by the owning
    block and hoisted over loops carried in its offsets; ``shared_hbm`` buffers use
    the root. The declaration anchors to the first child in dataflow order whose
    subtree contains a touching leaf, immediately before first use. A lone toucher
    anchors to that ISA leaf. Kernel parameters are never declared. Buffers are
    walked in ``all_buffers`` order for deterministic anchor lists.
    """
    params = set(ir.param_buffers)
    version_loops = {
        name: loop_nid
        for loop_nid, annotation in _pipeline_loops(ir).items()
        for name in annotation["versioned_buffers"]
    }
    ancestors = _ancestor_index(ir.tree)
    owners = {buffer.name: nid for nid in ir.tree.blocks() for buffer in ir.tree.block(nid).alloc_buffers}
    leaves_by_tensor: dict[str, list[int]] = {}
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if isinstance(data, ISANode):
            for region in data.operand_bindings.values():
                leaves_by_tensor.setdefault(region.tensor, []).append(nid)
    out: dict[int, list[Buffer]] = {}
    for name, buf in cast(_RenderIR, ir).buffers.items():
        if name in params:
            continue
        if leaves := leaves_by_tensor.get(name, ()):
            scope = (
                ir.tree.root
                if buf.location == "shared_hbm"
                else _hoisted_scope(ir.tree, name, leaves, owners[name], ancestors)
            )
            version_loop = version_loops.get(name)
            if version_loop is not None and (scope == version_loop or version_loop in ancestors[scope]):
                parent = ir.tree.parent(version_loop)
                assert parent is not None, f"pipeline loop {version_loop} has no declaration scope"
                scope = parent
            anchor = _anchor_child(ir.tree, scope, leaves, ancestors)
            out.setdefault(anchor, []).append(buf)
    return out


def _ancestor_index(tree: KernelTree) -> dict[int, tuple[int, ...]]:
    """Return root-first ancestor chains from one tree traversal."""
    result: dict[int, tuple[int, ...]] = {tree.root: ()}
    pending = [tree.root]
    while pending:
        parent = pending.pop()
        child_ancestors = (*result[parent], parent)
        children = tree.children(parent)
        for child in children:
            result[child] = child_ancestors
        pending.extend(reversed(children))
    return result


def _anchor_child(tree: KernelTree, scope: int, leaves: list[int], ancestors: dict[int, tuple[int, ...]]) -> int:
    """Return the node to emit a buffer's declaration before.

    When ``scope`` is an ISA leaf (lone toucher), the buffer anchors to that leaf.
    Otherwise the anchor is the first child of ``scope`` (in child order) whose
    subtree contains one of ``leaves`` — the first dataflow use of the buffer.
    """
    if isinstance(tree.data(scope), ISANode):
        return scope
    positions = {child: index for index, child in enumerate(tree.children(scope))}
    direct_children: list[int] = []
    for leaf in leaves:
        path = (*ancestors[leaf], leaf)
        if scope not in path:
            raise AssertionError(f"scope {scope} does not enclose touching leaf {leaf}")
        direct_children.append(path[path.index(scope) + 1])
    return min(direct_children, key=positions.__getitem__)


def _lca_nodes(tree: KernelTree, nids: list[int], ancestors: dict[int, tuple[int, ...]]) -> int:
    """Lowest common ancestor of ``nids`` — the deepest node on every root->nid path.

    Each node's path is its ancestors (root-first) plus itself; the LCA is the
    last node shared by all paths. A single distinct nid is its own LCA.
    """
    unique = set(nids)
    if len(unique) == 1:
        return next(iter(unique))
    paths = [[*ancestors[nid], nid] for nid in unique]
    lca = tree.root
    for level in zip(*paths):
        if len(set(level)) == 1:
            lca = level[0]
        else:
            break
    return lca


def _carried_loop_vars(tree: KernelTree, name: str, leaves: list[int]) -> set[str]:
    """Loop vars appearing in any region offset of buffer ``name`` across its touchers.

    A var in a region's ``lo`` means the buffer's live slice varies with that loop, so
    the buffer is carried across it and its declaration must hoist above the loop.
    """
    carried: set[str] = set()
    for leaf in leaves:
        data = tree.data(leaf)
        if not isinstance(data, ISANode):
            continue
        for region in data.operand_bindings.values():
            if region.tensor != name:
                continue
            for lo, _width in region.ranges:
                carried |= {var for var in to_affine(lo) if var is not None}
    return carried


def _hoisted_scope(
    tree: KernelTree, name: str, leaves: list[int], owner: int, ancestors: dict[int, tuple[int, ...]]
) -> int:
    """Find the tightest declaration scope consistent with placement and offsets.

    The owning block is a material placement boundary. The renderer may tighten
    within that block's direct loop nest, but it must not cross into a nested block;
    doing so would silently place a root-owned structural-only buffer after
    CodeMotion. Within the allowed nest, an offset that references an enclosing
    loop carries the allocation across that loop, so the scope rises above the
    outermost such loop.
    """
    lca = _lca_nodes(tree, leaves, ancestors)
    chain = [*ancestors[lca], lca]
    if owner in chain:
        owner_index = chain.index(owner)
        local_chain = chain[owner_index + 1 :]
        if any(isinstance(tree.data(nid), BlockNode) for nid in local_chain):
            return owner
    else:
        local_chain = chain
    carried = _carried_loop_vars(tree, name, leaves)
    scope = lca
    for nid in local_chain:
        data = tree.data(nid)
        if isinstance(data, ForNode) and data.loop_var in carried:
            parent = tree.parent(nid)
            assert parent is not None, f"carried loop {nid} has no parent"
            scope = parent
            break
    return scope


def _pipeline_loops(ir: KernelIR) -> dict[int, dict[str, Any]]:
    """Map ``loop_nid -> software_pipeline annotation`` for every annotated block.

    Scans every BlockNode; an absent ``software_pipeline`` annotation
    contributes nothing, so an un-annotated kernel yields an empty map and the
    rotation threading is a no-op.
    """
    out: dict[int, dict[str, Any]] = {}
    for block_nid in ir.tree.blocks():
        block = ir.tree.data(block_nid)
        assert isinstance(block, BlockNode)
        annotation = block.annotations.get("software_pipeline")
        if annotation is not None:
            loop_nid = annotation["loop_nid"]
            expected_children = annotation.get("children")
            if loop_nid not in ir.tree.graph:
                raise AssertionError(f"software-pipeline loop {loop_nid} is no longer in the tree")
            if annotation.get("loop") is not None and ir.tree.data(loop_nid) != annotation["loop"]:
                raise AssertionError(f"software-pipeline loop {loop_nid} has stale loop metadata")
            if expected_children is not None and tuple(ir.tree.children(loop_nid)) != tuple(expected_children):
                raise AssertionError(f"software-pipeline loop {loop_nid} has stale staged children")
            out[annotation["loop_nid"]] = annotation
    return out


def _emit_block(
    ir: KernelIR,
    block_nid: int,
    depth: int,
    code: list[str],
    pipeline_map: dict[int, dict[str, Any]],
    rotations: dict[str, Expr],
    substitutions: dict[str, Expr],
    emit_before: dict[int, list[Buffer]],
) -> None:
    """Emit one BlockNode: each child's anchored buffer declarations, then the child."""
    block = ir.tree.data(block_nid)
    assert isinstance(block, BlockNode)
    indent = _INDENT * depth
    for child_nid in ir.tree.children(block_nid):
        for buf in emit_before.get(child_nid, ()):
            code.append(indent + _emit_alloc(buf))
        child_data = ir.tree.data(child_nid)
        if isinstance(child_data, BlockNode):
            _emit_block(ir, child_nid, depth, code, pipeline_map, rotations, substitutions, emit_before)
        else:
            _emit_subtree(ir, child_nid, depth, code, pipeline_map, rotations, substitutions, emit_before)


def _emit_subtree(
    ir: KernelIR,
    nid: int,
    depth: int,
    code: list[str],
    pipeline_map: dict[int, dict[str, Any]],
    rotations: dict[str, Expr],
    substitutions: dict[str, Expr],
    emit_before: dict[int, list[Buffer]],
) -> None:
    """Emit a ForNode, ISANode, or nested BlockNode subtree.

    A BlockNode may appear as a ForNode child once ``compute_at`` lifts / sinks a
    block into a loop body; delegate it to :func:`_emit_block`.

    A ForNode emits, before each of its children, any buffers anchored to that
    child (``emit_before[child]``) — so a buffer used only within the loop is
    declared inside it, immediately before its first use.

    When ``nid`` is a pipelined loop (a key of ``pipeline_map``), the loop is
    emitted monolithically and the buffers versioned by that pipeline are added
    to ``rotations`` before recursing.
    """
    indent = _INDENT * depth
    node = ir.tree.data(nid)
    if isinstance(node, ForNode):
        if nid in pipeline_map:
            _emit_pipelined_loop(ir, nid, node, depth, code, pipeline_map, rotations, substitutions, emit_before)
        else:
            child_indent = _INDENT * (depth + 1)
            if nid in cast(_RenderIR, ir).shard_loops:
                local_var, per_program = f"{node.loop_var}_local", f"{node.extent} // nl.num_programs(0)"
                code.append(indent + f"for {local_var} in range({per_program}):")
                code.append(child_indent + f"{node.loop_var} = nl.program_id(0) * ({per_program}) + {local_var}")
            else:
                code.append(indent + f"for {node.loop_var} in range({node.extent}):")
            child_substitutions = {name: value for name, value in substitutions.items() if name != node.loop_var}
            for child_nid in ir.tree.children(nid):
                for buf in emit_before.get(child_nid, ()):
                    code.append(child_indent + _emit_alloc(buf))
                _emit_subtree(ir, child_nid, depth + 1, code, pipeline_map, rotations, child_substitutions, emit_before)
    elif isinstance(node, ISANode):
        code.extend(indent + line for line in _emit_isa_call(nid, node, ir, rotations, substitutions).splitlines())
    elif isinstance(node, BlockNode):
        _emit_block(ir, nid, depth, code, pipeline_map, rotations, substitutions, emit_before)
    else:
        raise TypeError(f"unexpected subtree node type {type(node).__name__}")


def _emit_pipelined_loop(
    ir: KernelIR,
    loop_nid: int,
    loop: ForNode,
    depth: int,
    code: list[str],
    pipeline_map: dict[int, dict[str, Any]],
    rotations: dict[str, Expr],
    substitutions: dict[str, Expr],
    emit_before: dict[int, list[Buffer]],
) -> None:
    """Emit a software pipeline as fill, steady-state, and drain phases."""
    annotation = pipeline_map[loop_nid]
    children = ir.tree.children(loop_nid)
    stages = tuple(annotation["stages"])
    order = tuple(annotation["order"])
    programs = cast(_RenderIR, ir).shard_loops.get(loop_nid, 1)
    minimum_extent = loop.extent // programs
    extent_source = str(loop.extent) if programs == 1 else f"{loop.extent} // nl.num_programs(0)"
    extent = Const(value=loop.extent) if programs == 1 else Var(name=extent_source)
    offset = None if programs == 1 else Var(name=f"nl.program_id(0) * ({extent_source})")
    emit_units = partial(
        _emit_pipeline_units,
        ir,
        children,
        order,
        loop_var=loop.loop_var,
        versioned_buffers=annotation["versioned_buffers"],
        code=code,
        pipeline_map=pipeline_map,
        rotations=rotations,
        substitutions=substitutions,
        emit_before=emit_before,
    )
    if len(stages) != len(children) or sorted(order) != list(range(len(children))):
        raise AssertionError(f"malformed software-pipeline annotation on loop {loop_nid}")
    max_stage = max(stages)
    if min(stages) != 0:
        raise AssertionError(f"software-pipeline stages must start at zero: {stages}")
    if max_stage == 0:
        code.append(_INDENT * depth + f"for {loop.loop_var} in range({extent_source}):")
        child_substitutions = {name: value for name, value in substitutions.items() if name != loop.loop_var}
        if offset is not None:
            child_substitutions[loop.loop_var] = Add(left=offset, right=Var(name=loop.loop_var))
        for child_nid in children:
            for buf in emit_before.get(child_nid, ()):
                code.append(_INDENT * (depth + 1) + _emit_alloc(buf))
            _emit_subtree(
                ir,
                child_nid,
                depth + 1,
                code,
                {key: value for key, value in pipeline_map.items() if key != loop_nid},
                rotations,
                child_substitutions,
                emit_before,
            )
        return

    prefix_end = min(max_stage, minimum_extent + max_stage)
    for tick in range(prefix_end):
        logical = {
            index: _pipeline_iteration(offset, Const(value=tick - stage))
            for index, stage in enumerate(stages)
            if 0 <= tick - stage < minimum_extent
        }
        emit_units(logical_iterations=logical, depth=depth)

    if minimum_extent > max_stage:
        code.append(_INDENT * depth + f"for {loop.loop_var} in range({extent_source} - {max_stage}):")
        logical = {
            index: _pipeline_iteration(
                offset,
                (
                    Var(name=loop.loop_var)
                    if max_stage == stage
                    else Add(left=Var(name=loop.loop_var), right=Const(value=max_stage - stage))
                ),
            )
            for index, stage in enumerate(stages)
        }
        emit_units(logical_iterations=logical, depth=depth + 1)

    for tick in range(max_stage):
        logical = {
            index: _pipeline_iteration(offset, Add(left=extent, right=Const(value=tick - stage)))
            for index, stage in enumerate(stages)
            if stage > tick
        }
        emit_units(logical_iterations=logical, depth=depth)


def _pipeline_iteration(offset: Expr | None, iteration: Expr) -> Expr:
    """Return one pipeline-local or program-global logical iteration."""
    return iteration if offset is None else Add(left=offset, right=iteration)


def _emit_pipeline_units(
    ir: KernelIR,
    children: list[int],
    order: tuple[int, ...],
    logical_iterations: Mapping[int, Expr],
    loop_var: str,
    versioned_buffers: tuple[str, ...],
    depth: int,
    code: list[str],
    pipeline_map: dict[int, dict[str, Any]],
    rotations: dict[str, Expr],
    substitutions: dict[str, Expr],
    emit_before: dict[int, list[Buffer]],
) -> None:
    """Emit the active child units for one pipeline tick."""
    indent = _INDENT * depth
    active = sorted(logical_iterations, key=lambda index: order[index])
    for index in active:
        child_nid = children[index]
        logical_iteration = logical_iterations[index]
        child_substitutions = {**substitutions, loop_var: logical_iteration}
        child_rotations = {**rotations, **_pipeline_rotations(ir, logical_iteration, versioned_buffers)}
        for buf in emit_before.get(child_nid, ()):
            code.append(indent + _emit_alloc(buf))
        _emit_subtree(ir, child_nid, depth, code, pipeline_map, child_rotations, child_substitutions, emit_before)


def _pipeline_rotations(ir: KernelIR, logical_iteration: Expr, versioned_buffers: tuple[str, ...]) -> dict[str, Expr]:
    """Return rotations for the buffers versioned by one pipeline."""
    out: dict[str, Expr] = {}
    for name in versioned_buffers:
        rotation = _version_rotation(ir.buffer(name), logical_iteration)
        if rotation is None:
            raise AssertionError(f"pipeline marks single-version buffer {name!r} as versioned")
        out[name] = rotation
    return out


def _version_rotation(buf: Buffer, logical_iteration: Expr) -> Expr | None:
    """Return the tile-axis version rotation for a multi-version buffer, or None.

    ``tiles_per_list`` is the per-version span inside each list allocation.
    When ``tiles_per_list == 1`` the rotation is the bare
    ``loop_var % versions`` (NO ``* 1`` — the validated kernel renders
    ``i_d1_0 % 2``, not ``i_d1_0 % 2 * 1``); only a >1 span wraps in
    ``Mul(..., Const(tiles_per_list))``.
    """
    if buf.versions <= 1:
        result = None
    else:
        if isinstance(logical_iteration, Const):
            mod: Expr = Const(value=logical_iteration.value % buf.versions)
        else:
            mod = Mod(left=logical_iteration, right=Const(value=buf.versions))
        tiles_per_list = buf.tiles_per_list()
        result = mod if tiles_per_list == 1 else Mul(left=mod, right=Const(value=tiles_per_list))
    return result


def _emit_alloc(buf: Buffer) -> str:
    """Emit the buffer declaration for ``buf``.

    ``shared_hbm`` buffers emit a single bare ``nl.ndarray`` of
    :meth:`Buffer.physical_shape` (no tile axis). Every sbuf/psum buffer emits a
    Python list of :attr:`Buffer.list_len` per-tile ndarrays
    (:meth:`Buffer.per_tile_physical_shape`) — uniformly, including ``list_len == 1``
    (a list-of-one), so the call site always indexes with a leading ``[list_idx]``.
    """
    if buf.location == "shared_hbm":
        shape = str(buf.physical_shape())
        result = f"{buf.name} = nl.ndarray({shape}, dtype=nl.{buf.physical_dtype()}, buffer=nl.{buf.location})"
    else:
        shape = str(tuple(buf.per_tile_physical_shape()))
        result = (
            f"{buf.name} = [nl.ndarray({shape}, dtype=nl.{buf.physical_dtype()}, "
            f"buffer=nl.{buf.location}) for _ in range({buf.list_len})]"
        )
    return result


def _emit_isa_call(
    leaf_nid: int, node: ISANode, ir: KernelIR, rotations: dict[str, Expr], substitutions: dict[str, Expr]
) -> str:
    """Emit ``nisa.<NAME>(slot=<region>, ..., kwarg=value, ...)`` for one ISA leaf.

    ``rotations`` maps a tensor name to its ``loop_var % versions`` tile-axis
    rotation when an enclosing loop is pipelined; ``rotations.get(...)`` is
    ``None`` for every single-version tensor, so the slice renders unchanged.
    """
    op_cls = node.op_cls
    if op_cls.INDIRECT_DMA_MODE is not None:
        return _emit_indirect_dma(node, ir, rotations, substitutions)
    parts: list[str] = []
    for slot in op_cls.OPERAND_AXES:
        if slot in node.operand_bindings:
            region = node.operand_bindings[slot]
            access_pattern = node.access_patterns.get(slot)
            if substitutions:
                region = BufferRegion(
                    tensor=region.tensor,
                    ranges=tuple(
                        (substitute(lower, substitutions), substitute(width, substitutions))
                        for lower, width in region.ranges
                    ),
                )
                if access_pattern is not None:
                    access_pattern = AccessPattern(
                        pattern=tuple(
                            (substitute(stride, substitutions), substitute(extent, substitutions))
                            for stride, extent in access_pattern.pattern
                        ),
                        offset=substitute(access_pattern.offset, substitutions),
                    )
            buf = ir.buffer(region.tensor)
            rotation = rotations.get(region.tensor)
            if slice_specs := getattr(op_cls, "INPUT_SLICES", {}).get(slot, ()):
                for axis, start_key, width_key, *alignment in slice_specs:
                    start, width = int(node.kwargs[start_key]), int(node.kwargs[width_key])
                    if alignment:
                        (output_slot,) = cast(tuple[str], tuple(alignment))
                        output = _substituted_region(node.operand_bindings[output_slot], substitutions)
                        region = region.with_partition_aligned_slice(axis, start, width, output)
                    else:
                        region = region.with_partition_aligned_slice(axis, start, width)
            if access_pattern is None:
                rendered = render_buffer_region(region, buf, rotation)
            else:
                rendered = render_access_pattern(region.tensor, access_pattern, buf, rotation)
            parts.append(f"{slot}={rendered}")
    internal_kwargs = getattr(op_cls, "CODEGEN_ONLY_KWARGS", frozenset())
    for k, v in node.kwargs.items():
        if k not in internal_kwargs:
            rendered = (
                _render_first_write(ir, leaf_nid, cast(tuple[str, ...], v), substitutions)
                if k == "accumulate" and isinstance(v, tuple)
                else _render_kwarg(k, substitute(v, substitutions) if isinstance(v, Expr) else v)
            )
            parts.append(f"{k}={rendered}")
    call = f"nisa.{op_cls.NAME}({', '.join(parts)})"
    ownership = node.kwargs.get("program_ownership")
    if isinstance(ownership, tuple):
        axis, programs = cast(tuple[str, int], ownership)
        iteration = format_expr(operation_axis_value(ir, leaf_nid, axis, substitutions))
        call = f"if nl.num_programs(0) == 1 or {iteration} % {programs} == nl.program_id(0):\n" f"{_INDENT}{call}"
    elif getattr(op_cls, "SINGLE_PROGRAM_ZERO", False):
        destination = next(part for part in parts if part.startswith("dst="))
        call = (
            f"if nl.num_programs(0) == 1:\n"
            f"{_INDENT}nisa.memset({destination}, value=0.0)\n"
            f"else:\n{_INDENT}{call}"
        )
    return call


def _emit_indirect_dma(node: ISANode, ir: KernelIR, rotations: dict[str, Expr], substitutions: dict[str, Expr]) -> str:
    """Render one row gather or scatter with an SBUF offset."""
    regions = {slot: _substituted_region(region, substitutions) for slot, region in node.operand_bindings.items()}
    source, indices, destination = (regions[key] for key in ("src", "indices", "dst"))
    mode = node.op_cls.INDIRECT_DMA_MODE
    gather = mode in {"gather", "scalar_gather"}
    data_region, hbm_region = (destination, source) if gather else (source, destination)
    partition, free = (_constant_width(data_region, axis) for axis in (0, 1))
    if mode == "scalar_gather":
        indices = indices.with_partition_aligned_slice(1, int(node.kwargs.get("index", 0)), 1)
    free_lower, hbm_buffer = hbm_region.ranges[1][0], ir.buffer(hbm_region.tensor)
    index_text = render_buffer_region(indices, ir.buffer(indices.tensor), rotations.get(indices.tensor))
    data_text = render_buffer_region(data_region, ir.buffer(data_region.tensor), rotations.get(data_region.tensor))
    row_stride = hbm_buffer.shape[1]
    if mode == "scalar_gather":
        row_lower = data_region.ranges[0][0]
        if ir.buffer(data_region.tensor).shape[0] > partition:
            row_lower = Mul(left=row_lower, right=Const(value=partition))
        free_lower = Add(left=Mul(left=row_lower, right=Const(value=free)), right=free_lower)
        row_stride = free
    offset_kind = "scalar_offset" if mode == "scalar_gather" else "vector_offset"
    indirect = (
        f"{hbm_region.tensor}.ap(pattern=[[{row_stride}, {partition}], [1, {free}]], "
        f"offset={format_expr(free_lower)}, {offset_kind}={index_text}, indirect_dim=0)"
    )
    if gather:
        operands = f"src={indirect}, dst={data_text}"
    else:
        operands = f"src={data_text}, dst={indirect}"
    return f"nisa.dma_copy({operands}, oob_mode=oob_mode.error, dge_mode=nisa.dge_mode.swdge)"


def _substituted_region(region: BufferRegion, substitutions: dict[str, Expr]) -> BufferRegion:
    """Apply loop substitutions to one operand region."""
    if not substitutions:
        return region
    return BufferRegion(
        tensor=region.tensor,
        ranges=tuple(
            (substitute(lower, substitutions), substitute(width, substitutions)) for lower, width in region.ranges
        ),
    )


def _constant_width(region: BufferRegion, axis: int) -> int:
    """Return one statically known region width."""
    width = region.ranges[axis][1]
    if not isinstance(width, Const):
        raise AssertionError(f"{region.tensor}: indirect DMA requires a constant tile width")
    return width.value


_NL_OP_KWARGS = frozenset({"comp_op0", "comp_op1", "dtype", "op", "op0", "op1", "reduce_op"})
"""ISA kwargs whose string value names an ``nl`` math operator. ``nisa`` ALU
ops (``tensor_tensor``, ``tensor_scalar``, ``activation``, ``tensor_reduce``)
take the operator as an ``nl`` reference (e.g. ``op=nl.add``), not a bare
string — so these render as ``nl.<value>`` while every other kwarg renders
via ``repr`` (e.g. memset's ``value=0.0``)."""


def _render_kwarg(key: str, value: Any) -> str:
    """Render one ISA kwarg value, mapping ALU-operator names to ``nl.<name>``."""
    value = "maximum" if key in {"op", "reduce_op"} and value == "max" else value
    if key == "reduce_cmd" or key in {"send_to_rank", "recv_from_rank"} and value == "program_peer":
        return f"nisa.reduce_cmd.{value}" if key == "reduce_cmd" else "1 - nl.program_id(0)"
    if key in _NL_OP_KWARGS and isinstance(value, str):
        return f"nl.{value}"
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return f"float('{value}')"
    return format_expr(value) if isinstance(value, Expr) else repr(value)


def _render_first_write(
    ir: KernelIR, leaf_nid: int, reduction_axes: tuple[str, ...], substitutions: dict[str, Expr]
) -> str:
    """Render accumulation from the final reduction-loop structure."""
    iterations = sum((operation_axis_iterations(ir, leaf_nid, axis, substitutions) for axis in reduction_axes), ())
    return " or ".join(f"{format_expr(iteration)} != 0" for iteration in iterations) or "False"


def _format_tile_index(lo: Expr, rotation: Expr | None) -> str:
    """Render the SBUF/PSUM tile-axis index, optionally + a version rotation.

    ``format_expr`` normalises through ``to_affine``, which RAISES
    ``NonAffineError`` on ``Mod(Var, Const)`` (a version rotation like
    ``i_d1_0 % 2``) — the modulo of a variable is not affine. So the rotation
    is rendered with the non-normalising ``_format_raw`` and combined with the
    (affine) ``lo`` here, dropping ``lo`` when it is the rebased ``Const(0)``.
    """
    if rotation is None:
        result = format_expr(lo)
    else:
        rot_str = _format_rotation(rotation)
        if isinstance(lo, Const) and lo.value == 0:
            result = rot_str
        else:
            result = f"{format_expr(lo)} + {rot_str}"
    return result


def _format_rotation(expr: Expr) -> str:
    """Render a modulo rotation with precedence preserved for shifted indices."""
    if isinstance(expr, (Const, Var)):
        result = _format_raw(expr)
    elif isinstance(expr, Add):
        result = f"{_format_rotation(expr.left)} + {_format_rotation(expr.right)}"
    elif isinstance(expr, (Mul, Mod)):
        left = _format_rotation(expr.left)
        right = _format_rotation(expr.right)
        if isinstance(expr.left, Add):
            left = f"({left})"
        if isinstance(expr.right, Add):
            right = f"({right})"
        operator = "*" if isinstance(expr, Mul) else "%"
        result = f"{left} {operator} {right}"
    else:
        raise TypeError(f"unsupported rotation expression {type(expr).__name__}")
    return result


def _format_local_tile_index(local_tile: str, rotation: Expr | None) -> str:
    """Add a pipeline version rotation to one list-local logical tile index."""
    if rotation is None:
        result = local_tile
    elif local_tile == "0":
        result = _format_rotation(rotation)
    else:
        result = f"{local_tile} + {_format_rotation(rotation)}"
    return result


def render_buffer_region(region: BufferRegion, buf: Buffer, rotation: Expr | None = None) -> str:
    """Render a :class:`BufferRegion` as a Python slice expression on its tensor.

    ``shared_hbm`` renders flat ``name[lo:hi, ...]``. Every sbuf/psum buffer renders
    as a list access ``name[list_idx][0:P, mid_idx, F]`` (uniform — there is no bare
    form). The partition axis (axis 0) carries the tile index ``t``; with
    ``a = tiles_per_list = logical_tiles // list_len``, branch on ``list_len``:

    * ``list_len == 1`` — a list-of-one: ``list_idx = 0``, ``mid_idx = t`` (the whole
      tile index). Preserves the pre-uniform packed middle, so a canonical multi-tile
      buffer renders ``buf[0][0:P, t, F]``.
    * ``a == 1`` (``list_len == T``, the full split) — ``list_idx = t``, ``mid_idx = 0``.
    * ``a > 1`` (``1 < list_len < T``) — ``list_idx = t // a``, ``mid_idx = t % a``,
      both via the non-normalising ``_format_raw`` (the aligned index is non-affine
      under ``FloorDiv``, so ``format_expr``/``to_affine`` would raise).

    ``rotation`` is added only to ``mid_idx``. Its stride is ``a``, so every list
    entry stores ``a`` logical tiles for each pipeline version while ``list_idx``
    remains a pure function of the logical tile.
    """
    list_subscript = ""
    parts: list[str] = []
    for axis_index, (lo, hi) in enumerate(region.ranges):
        if axis_index == 0 and buf.location != "shared_hbm":
            partition_extent = buf.partition_extent()
            if not isinstance(hi, Const) or hi.value != partition_extent:
                raise AssertionError(f"{buf.name}: SBUF/PSUM partition axis must use a partition-sized tile; got {hi}")
            a = buf.tiles_per_list()
            if buf.list_len == 1:
                list_subscript = "[0]"
                parts.append(f"0:{partition_extent}")
                parts.append(_format_tile_index(lo, rotation))
            elif a == 1:
                list_subscript = f"[{_format_tile_index(lo, None)}]"
                parts.append(f"0:{partition_extent}")
                parts.append(_format_local_tile_index("0", rotation))
            else:
                tile = f"({_format_raw(lo)})"
                list_subscript = f"[{tile} // {a}]"
                parts.append(f"0:{partition_extent}")
                parts.append(_format_local_tile_index(f"{tile} % {a}", rotation))
        else:
            lo_str = format_expr(lo)
            hi_str = format_expr(hi)
            parts.append(f"{lo_str}:{lo_str} + {hi_str}")
    return f"{region.tensor}{list_subscript}[{', '.join(parts)}]"


def render_access_pattern(tensor: str, access_pattern: AccessPattern, buf: Buffer, rotation: Expr | None = None) -> str:
    """Render one flattened multidimensional ``Tensor.ap`` view."""
    if buf.versions > 1 and rotation is None:
        raise AssertionError(f"{tensor}: versioned access pattern requires pipeline buffer rotation")
    if buf.location == "shared_hbm" and rotation is not None:
        raise AssertionError(f"{tensor}: shared HBM access pattern cannot use pipeline buffer rotation")
    list_index, access_pattern = access_pattern_allocation_view(access_pattern, buf)
    base = tensor if buf.location == "shared_hbm" else f"{tensor}[{_format_raw(list_index)}]"
    dimensions = ", ".join(
        f"[{format_expr(stride)}, {format_expr(extent)}]" for stride, extent in access_pattern.pattern
    )
    offset = _format_access_pattern_offset(access_pattern.offset, buf, rotation)
    return f"{base}.ap(pattern=[{dimensions}], offset={offset})"


def _format_access_pattern_offset(offset: Expr, buf: Buffer, rotation: Expr | None) -> str:
    """Render a flattened access-pattern offset with an optional tile rotation."""
    result = _format_raw(offset) if isinstance(offset, Mod) else format_expr(offset)
    if rotation is not None:
        free = buf.per_tile_physical_shape()[2]
        flattened = rotation if free == 1 else Mul(left=rotation, right=Const(value=free))
        rotation_text = _format_rotation(flattened)
        result = rotation_text if isinstance(offset, Const) and offset.value == 0 else f"{result} + {rotation_text}"
    return result


__all__ = ["emit_body", "render_access_pattern", "render_buffer_region"]
