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

import math
from collections.abc import Mapping
from functools import partial
from typing import Any, cast

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Add, Const, Expr, Mod, Mul, Var, _format_raw, expr_variables, format_expr, substitute
from nkigym.ir.operand_layout import access_pattern_allocation_view
from nkigym.ir.program_sharding import (
    configured_program_shards,
    operation_axis_iterations,
    operation_axis_value,
    owning_block,
)
from nkigym.ir.tree import AccessPattern, BlockNode, Buffer, BufferRegion, ForNode, ISANode
from nkigym.ops.base import AxisRole

_INDENT = "    "


class _RenderIR(KernelIR):
    """Render state sharing one buffer snapshot, pipeline map, and output list."""

    def __init__(self, ir: KernelIR) -> None:
        """Capture completed scheduling metadata and initialize one output buffer."""
        super().__init__(ir.func_name, ir.param_names, ir.return_names, ir.tree, ir.dependency, ir.param_buffers)
        self.buffers, self.shard_loops = (ir.all_buffers(), configured_program_shards(ir))
        self.unscoped_blocks: set[int] = {ir.tree.root}
        self.code: list[str] = []
        self.pipeline_map = _pipeline_loops(self)
        self.emit_before = _alloc_emit_anchors(self, self.pipeline_map)

    def buffer(self, name: str) -> Buffer:
        """Resolve one buffer from the render-local map."""
        return self.buffers[name]

    def _emit_child(self, nid: int, depth: int, rotations: dict[str, Expr], substitutions: dict[str, Expr]) -> None:
        """Emit one child after its first-use buffer declarations."""
        self.code.extend(_INDENT * depth + _emit_alloc(buf) for buf in self.emit_before.get(nid, ()))
        self._emit_subtree(nid, depth, rotations, substitutions)

    def _emit_block(
        self, block_nid: int, depth: int, rotations: dict[str, Expr], substitutions: dict[str, Expr]
    ) -> None:
        """Emit declarations and children within the block's allocation lifetime.

        The tracer executes Python loops eagerly, so Python indentation alone cannot
        end a scratch allocation's lifetime. A single-execution NKI region preserves
        each non-root allocation scope without changing the IR's iteration schedule.
        Captured register writes stay in their allocation's region: NKI registers
        hold SSA values, which cannot escape a nested region without yielded results.
        """
        block = self.tree.data(block_nid)
        assert isinstance(block, BlockNode)
        predicate = block.annotations.get("predicate")
        if predicate is not None and self.buffer(predicate[0]).location != "register":
            raise ValueError("conditional scopes require a scalar register")
        scoped = predicate is not None or (
            block_nid not in self.unscoped_blocks
            and any(
                buf.location != "shared_hbm"
                for child in self.tree.children(block_nid)
                for buf in self.emit_before.get(child, ())
            )
        )
        if scoped:
            self.code.append(_INDENT * depth + f"def _block_{block_nid}(_block_index):")
            depth += 1
        for child_nid in self.tree.children(block_nid):
            self._emit_child(child_nid, depth, rotations, substitutions)
        if scoped:
            bounds = "0, 1" if predicate is None else f"{predicate[0]}, 1" if predicate[1] else f"0, {predicate[0]}"
            self.code.append(_INDENT * (depth - 1) + f"fori_loop({bounds}, _block_{block_nid})")

    def _emit_subtree(self, nid: int, depth: int, rotations: dict[str, Expr], substitutions: dict[str, Expr]) -> None:
        """Emit a ForNode, ISANode, or nested BlockNode subtree.

        A BlockNode may appear as a ForNode child once ``compute_at`` lifts / sinks a
        block into a loop body; delegate it to :func:`_emit_block`.

        A ForNode emits, before each of its children, any buffers anchored to that
        child (``emit_before[child]``) — so a buffer used only within the loop is
        declared inside it, immediately before its first use.

        When ``nid`` is a pipelined loop (a key of ``pipeline_map``), the loop is
        emitted monolithically and the buffers versioned by that pipeline are added
        to ``rotations`` before recursing. Structural scopes with a sequential
        iteration variable use device ``fori_loop`` control flow.
        """
        indent = _INDENT * depth
        node = self.tree.data(nid)
        if isinstance(node, ForNode):
            if nid in self.pipeline_map:
                self._emit_pipelined_loop(nid, node, depth, rotations, substitutions)
            else:
                child_indent = _INDENT * (depth + 1)
                structured = any(
                    isinstance((block := self.tree.data(child)), BlockNode)
                    and not block.reads
                    and not block.writes
                    and any(
                        iv.role is AxisRole.SEQUENTIAL and node.loop_var in expr_variables(value)
                        for iv, value in zip(block.iter_vars, block.iter_values)
                    )
                    for child in self.tree.children(nid)
                )
                if nid in self.shard_loops:
                    if structured:
                        raise ValueError("sequential state-update loops cannot be sharded")
                    local_var, per_program = (f"{node.loop_var}_local", f"{node.extent} // nl.num_programs(0)")
                    self.code.append(indent + f"for {local_var} in range({per_program}):")
                    self.code.append(
                        child_indent + f"{node.loop_var} = nl.program_id(0) * ({per_program}) + {local_var}"
                    )
                elif structured:
                    self.code.append(indent + f"def _loop_{nid}({node.loop_var}):")
                else:
                    self.code.append(indent + f"for {node.loop_var} in range({node.extent}):")
                child_substitutions = {name: value for name, value in substitutions.items() if name != node.loop_var}
                for child_nid in self.tree.children(nid):
                    self._emit_child(child_nid, depth + 1, rotations, child_substitutions)
                if structured:
                    self.code.append(indent + f"fori_loop(0, {node.extent}, _loop_{nid})")
        elif isinstance(node, ISANode):
            self.code.extend(
                indent + line for line in _emit_isa_call(nid, node, self, rotations, substitutions).splitlines()
            )
        elif isinstance(node, BlockNode):
            self._emit_block(nid, depth, rotations, substitutions)
        else:
            raise TypeError(f"unexpected subtree node type {type(node).__name__}")

    def _emit_pipelined_loop(
        self, loop_nid: int, loop: ForNode, depth: int, rotations: dict[str, Expr], substitutions: dict[str, Expr]
    ) -> None:
        """Emit a software pipeline as fill, steady-state, and drain phases."""
        annotation = self.pipeline_map[loop_nid]
        children = self.tree.children(loop_nid)
        stages = tuple(annotation["stages"])
        order = tuple(annotation["order"])
        programs = self.shard_loops.get(loop_nid, 1)
        minimum_extent = loop.extent // programs
        extent_source = str(loop.extent) if programs == 1 else f"{loop.extent} // nl.num_programs(0)"
        extent = Const(value=loop.extent) if programs == 1 else Var(name=extent_source)
        offset = None if programs == 1 else Var(name=f"nl.program_id(0) * ({extent_source})")
        emit_units = partial(
            self._emit_pipeline_units,
            children,
            order=order,
            loop_var=loop.loop_var,
            versioned_buffers=annotation["versioned_buffers"],
            rotations=rotations,
            substitutions=substitutions,
        )
        if len(stages) != len(children) or sorted(order) != list(range(len(children))):
            raise AssertionError(f"malformed software-pipeline annotation on loop {loop_nid}")
        max_stage = max(stages)
        if min(stages) != 0:
            raise AssertionError(f"software-pipeline stages must start at zero: {stages}")
        if max_stage == 0:
            self.code.append(_INDENT * depth + f"for {loop.loop_var} in range({extent_source}):")
            buffered_iteration = _pipeline_iteration(offset, Var(name=loop.loop_var))
            emit_units(
                order=tuple(range(len(children))),
                logical_iterations={index: buffered_iteration for index in range(len(children))},
                depth=depth + 1,
            )
            return
        for tick in range(max_stage):
            logical = {
                index: _pipeline_iteration(offset, Const(value=tick - stage))
                for index, stage in enumerate(stages)
                if 0 <= tick - stage < minimum_extent
            }
            emit_units(logical_iterations=logical, depth=depth)
        if minimum_extent > max_stage:
            self.code.append(_INDENT * depth + f"for {loop.loop_var} in range({extent_source} - {max_stage}):")
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

    def _emit_pipeline_units(
        self,
        children: list[int],
        order: tuple[int, ...],
        logical_iterations: Mapping[int, Expr],
        loop_var: str,
        versioned_buffers: tuple[str, ...],
        depth: int,
        rotations: dict[str, Expr],
        substitutions: dict[str, Expr],
    ) -> None:
        """Emit the active child units for one pipeline tick."""
        active = sorted(logical_iterations, key=lambda index: order[index])
        for index in active:
            child_nid = children[index]
            logical_iteration = logical_iterations[index]
            child_substitutions = {**substitutions, loop_var: logical_iteration}
            child_rotations = {**rotations, **_pipeline_rotations(self, logical_iteration, versioned_buffers)}
            self._emit_child(child_nid, depth, child_rotations, child_substitutions)


def emit_body(ir: KernelIR) -> str:
    """Emit the kernel body for the entire tree.

    The root is a BlockNode (empty iter_vars, holds kernel-lifetime buffers).
    Emit it directly at depth=1 (one indent level inside the kernel function).
    A ``{loop_nid: annotation}`` map of every ``software_pipeline`` annotation
    is built once and threaded down so a pipelined loop rotates its
    multi-version buffer accesses (see :func:`_emit_subtree`).

    Each declaration is emitted in its owning block, immediately before
    the first child that uses it. Allocation placement is explicit in the IR. The
    ``{node_nid: [Buffer, ...]}`` map is threaded down so each node emits, before
    each child, the declarations anchored to that child.
    """
    context = _RenderIR(ir)
    context._emit_block(ir.tree.root, depth=1, rotations={}, substitutions={})
    return "\n".join(context.code) + "\n"


def _alloc_emit_anchors(ir: KernelIR, pipeline_map: dict[int, dict[str, Any]]) -> dict[int, list[Buffer]]:
    """Map each tree node to the buffers emitted immediately before it.

    Scratch-buffer declarations use their owning block; ``shared_hbm`` uses
    the root. The declaration anchors to the first child in dataflow order whose
    subtree contains a touching leaf, immediately before first use. A lone toucher
    anchors to that ISA leaf. Kernel parameters are never declared. Buffers are
    walked in ``all_buffers`` order for deterministic anchor lists.
    """
    version_loops = {
        name: loop_nid for loop_nid, annotation in pipeline_map.items() for name in annotation["versioned_buffers"]
    }
    ancestors = ir.dependency._topology()[1]
    owners = {buffer.name: nid for nid in ir.tree.blocks() for buffer in ir.tree.block(nid).alloc_buffers}
    leaves_by_tensor = ir.dependency.touches_by_tensor
    out: dict[int, list[Buffer]] = {}
    for name, buf in cast(_RenderIR, ir).buffers.items():
        if name in ir.param_buffers:
            continue
        if leaves := leaves_by_tensor.get(name, ()):
            scope = ir.tree.root if buf.location == "shared_hbm" else owners[name]
            if buf.location == "register":
                for leaf in leaves:
                    if name in ir.dependency.info(leaf).writes:
                        cast(_RenderIR, ir).unscoped_blocks.update(ancestors[leaf][len(ancestors[scope]) + 1 :])
            version_loop = version_loops.get(name)
            if version_loop is not None and scope not in ancestors[version_loop]:
                raise ValueError(f"buffer {name!r} must be placed outside pipeline loop {version_loop}")
            anchor = _anchor_child(scope, leaves, ancestors)
            out.setdefault(anchor, []).append(buf)
    return out


def _anchor_child(scope: int, leaves: list[int], ancestors: dict[int, tuple[int, ...]]) -> int:
    """Return the node to emit a buffer's declaration before.

    ``scope`` is the declaration's block and ``leaves`` is in execution order.
    Validate that every access is enclosed, then use the first access's path.
    """
    for leaf in leaves:
        if scope not in ancestors[leaf]:
            raise AssertionError(f"scope {scope} does not enclose touching leaf {leaf}")
    return (*ancestors[leaves[0]], leaves[0])[len(ancestors[scope]) + 1]


def _pipeline_loops(ir: KernelIR) -> dict[int, dict[str, Any]]:
    """Map ``loop_nid -> software_pipeline annotation`` for every annotated block.

    Scans every BlockNode; an absent ``software_pipeline`` annotation
    contributes nothing, so an un-annotated kernel yields an empty map and the
    rotation threading is a no-op.
    """
    out: dict[int, dict[str, Any]] = {}
    for block_nid in ir.tree.blocks():
        block = ir.tree.block(block_nid)
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


def _pipeline_iteration(offset: Expr | None, iteration: Expr) -> Expr:
    """Return one pipeline-local or program-global logical iteration."""
    return iteration if offset is None else Add(left=offset, right=iteration)


def _pipeline_rotations(ir: KernelIR, logical_iteration: Expr, versioned_buffers: tuple[str, ...]) -> dict[str, Expr]:
    """Return rotations for the buffers versioned by one pipeline."""
    out: dict[str, Expr] = {}
    for name in versioned_buffers:
        buf = ir.buffer(name)
        if buf.versions <= 1:
            raise AssertionError(f"pipeline marks single-version buffer {name!r} as versioned")
        if isinstance(logical_iteration, Const):
            mod: Expr = Const(value=logical_iteration.value % buf.versions)
        else:
            mod = Mod(left=logical_iteration, right=Const(value=buf.versions))
        tiles_per_list = buf.tiles_per_list()
        out[name] = mod if tiles_per_list == 1 else Mul(left=mod, right=Const(value=tiles_per_list))
    return out


def _emit_alloc(buf: Buffer) -> str:
    """Emit the buffer declaration for ``buf``.

    ``shared_hbm`` buffers emit a single bare ``nl.ndarray`` of
    :meth:`Buffer.physical_shape` (no tile axis). Every sbuf/psum buffer emits a
    Python list of :attr:`Buffer.list_len` per-tile ndarrays
    (:meth:`Buffer.per_tile_physical_shape`) — uniformly, including ``list_len == 1``
    (a list-of-one), so the call site always indexes with a leading ``[list_idx]``.
    """
    if buf.location == "register":
        if buf.shape != (1, 1) or buf.list_len != 1 or buf.versions != 1:
            raise ValueError(f"{buf.name}: scalar registers require one unversioned element")
        return f"{buf.name} = nisa.register_alloc()"
    if buf.location == "shared_hbm":
        return f"{buf.name} = nl.ndarray({buf.physical_shape()}, dtype=nl.{buf.physical_dtype()}, buffer=nl.shared_hbm)"
    return f"{buf.name} = [nl.ndarray({buf.per_tile_physical_shape()}, dtype=nl.{buf.physical_dtype()}, buffer=nl.{buf.location}) for _ in range({buf.list_len})]"


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
    scalar_copy = getattr(op_cls, "SCALAR_OFFSET_COPY", False)
    parts: list[str] = []
    for slot in op_cls.OPERAND_AXES:
        if scalar_copy and slot == "offset":
            continue
        if slot in node.operand_bindings:
            region = _substituted_region(node.operand_bindings[slot], substitutions)
            access_pattern = node.access_patterns.get(slot)
            if substitutions and access_pattern is not None:
                access_pattern = AccessPattern(
                    pattern=tuple(
                        (substitute(stride, substitutions), substitute(extent, substitutions))
                        for stride, extent in access_pattern.pattern
                    ),
                    offset=substitute(access_pattern.offset, substitutions),
                )
            buf = ir.buffer(region.tensor)
            rotation = rotations.get(region.tensor)
            if slot in getattr(op_cls, "STRIDED_COPY_INPUTS", ()):
                free = buf.shape[1]
                access_pattern = AccessPattern(
                    pattern=(
                        (Const(value=buf.logical_tile_count() * free), region.ranges[0][1]),
                        *((Const(value=s), Const(value=n)) for s, n in node.kwargs["pattern"]),
                    ),
                    offset=Add(
                        left=Mul(left=region.ranges[0][0], right=Const(value=free)),
                        right=Add(left=region.ranges[1][0], right=Const(value=node.kwargs["offset"])),
                    ),
                )
            if slice_specs := getattr(op_cls, "INPUT_SLICES", {}).get(slot, ()):
                for axis, start_key, width_key, *alignment in slice_specs:
                    start, width = (node.kwargs[start_key], int(node.kwargs[width_key]))
                    output = None
                    if alignment:
                        (output_slot,) = cast(tuple[str], tuple(alignment))
                        output = _substituted_region(node.operand_bindings[output_slot], substitutions)
                    region = region.with_partition_aligned_slice(axis, start, width, output)
            if access_pattern is None:
                rendered = render_buffer_region(region, buf, rotation)
            else:
                rendered = render_access_pattern(region.tensor, access_pattern, buf, rotation)
            if dtype := getattr(op_cls, "REINTERPRET_INPUT_DTYPES", {}).get(slot):
                rendered = f"{rendered}.view(dtype=nl.{dtype})"
            if scalar_copy and slot == scalar_copy:
                source = _substituted_region(
                    node.operand_bindings["dst" if scalar_copy == "src" else "src"], substitutions
                )
                offset = _substituted_region(node.operand_bindings["offset"], substitutions)
                if ir.buffer(source.tensor).physical_dtype() != buf.physical_dtype():
                    raise ValueError("dynamic slice copies require matching source and destination storage")
                partition, width = (_constant_width(source, axis) for axis in (0, 1))
                stride = math.prod(buf.per_tile_physical_shape()[1:])
                index = render_buffer_region(offset, ir.buffer(offset.tensor), rotations.get(offset.tensor))
                rendered = f"{rendered}.ap(pattern=[[{stride}, {partition}], [1, {width}]], scalar_offset={index}, indirect_dim=1)"
            native_slot = getattr(op_cls, "ISA_OPERAND_NAMES", {}).get(slot, slot)
            parts.append(f"{native_slot}={rendered}")
    kwargs = dict(node.kwargs)
    isa_name = op_cls.NAME
    if native_parameters := getattr(op_cls, "native_parameters", None):
        isa_name, kwargs = native_parameters(kwargs, frozenset(node.operand_bindings))
    if op_cls.NAME == "nc_matmul":
        kwargs.setdefault("name", "matmul")
        kwargs.setdefault("accumulate", True)
    for abstract, (offset_key, extent_key, multiplier_key) in op_cls.ITERATION_OFFSET_KWARGS.items():
        concrete = ir.tree.block(owning_block(ir, leaf_nid)).axis_map[abstract]
        base = int(kwargs.get(offset_key, 0))
        scale = int(kwargs[extent_key]) * int(kwargs.get(multiplier_key, 0))
        if scale:
            dynamic = Mul(left=operation_axis_value(ir, leaf_nid, concrete, {}), right=Const(value=scale))
            kwargs[offset_key] = dynamic if base == 0 else Add(left=Const(value=base), right=dynamic)
    internal_kwargs = getattr(op_cls, "CODEGEN_ONLY_KWARGS", frozenset()) | {"no_reorder", "program_ownership"}
    for k, v in kwargs.items():
        if k not in internal_kwargs:
            if k == "name" and isinstance(v, str) and v:
                block = ir.tree.block(owning_block(ir, leaf_nid))
                axis_index = tuple(block.axis_map).index("K")
                iteration = format_expr(substitute(block.iter_values[axis_index], substitutions))
                coordinates = "_".join(
                    (
                        f"{{{format_expr(substitutions.get(loop.loop_var, Var(name=loop.loop_var)))}}}"
                        for loop in map(ir.tree.data, ir.tree.ancestors(leaf_nid))
                        if isinstance(loop, ForNode)
                    )
                )
                rendered = f'f"{v}_{leaf_nid}_{coordinates}_{{{iteration}}}"'
            else:
                rendered = (
                    _render_first_write(ir, leaf_nid, cast(tuple[str, ...], v), substitutions)
                    if k == "accumulate" and isinstance(v, tuple)
                    else _render_kwarg(k, substitute(v, substitutions) if isinstance(v, Expr) else v)
                )
            parts.append(f"{k}={rendered}")
    call = f"nisa.{isa_name}({', '.join(parts)})"
    if getattr(op_cls, "SHARDED_SINGLE_PROGRAM_ZERO", False):
        destination = next((part for part in parts if part.startswith("dst=")))
        one_participant = node.kwargs.get("participating_programs", 2) == 1
        call = f"if {one_participant!r} or nl.num_programs(0) == 1:\n{_INDENT}nisa.memset({destination}, value=0.0)\nelse:\n{_INDENT}{call}"
    ownership = node.kwargs.get("program_ownership")
    if isinstance(ownership, tuple):
        axis, programs = cast(tuple[str, int], ownership)
        iteration = format_expr(operation_axis_value(ir, leaf_nid, axis, substitutions))
        indented = "\n".join((f"{_INDENT}{line}" for line in call.splitlines()))
        call = f"if nl.num_programs(0) == 1 or ({iteration}) % {programs} == nl.program_id(0):\n{indented}"
    if node.kwargs.get("no_reorder"):
        call = "with nl.no_reorder():\n" + "\n".join((f"{_INDENT}{line}" for line in call.splitlines()))
    return call


def _emit_indirect_dma(node: ISANode, ir: KernelIR, rotations: dict[str, Expr], substitutions: dict[str, Expr]) -> str:
    """Render one row gather or scatter with an SBUF offset."""
    regions = {slot: _substituted_region(region, substitutions) for slot, region in node.operand_bindings.items()}
    source, indices, destination = (regions[key] for key in ("src", "indices", "dst"))
    mode = node.op_cls.INDIRECT_DMA_MODE
    gather = mode in {"column_gather", "gather", "scalar_gather"}
    data_region, hbm_region = (destination, source) if gather else (source, destination)
    partition, free = (_constant_width(data_region, axis) for axis in (0, 1))
    scalar = mode in {"scalar_gather", "scalar_scatter"}
    if scalar:
        indices = indices.with_partition_aligned_slice(1, cast(int | Expr, node.kwargs.get("index", 0)), 1)
    free_lower, hbm_buffer = (hbm_region.ranges[1][0], ir.buffer(hbm_region.tensor))
    index_text = render_buffer_region(indices, ir.buffer(indices.tensor), rotations.get(indices.tensor))
    data_text = render_buffer_region(data_region, ir.buffer(data_region.tensor), rotations.get(data_region.tensor))
    row_stride = hbm_buffer.shape[1]
    if mode == "column_gather":
        free_lower, row_stride = (Const(value=0), 1)
    if scalar:
        row_lower = substitute(cast(Expr, node.kwargs.get("row_offset", data_region.ranges[0][0])), substitutions)
        row_lower = Mul(left=row_lower, right=Const(value=ir.buffer(data_region.tensor).partition_extent()))
        row_stride = int(node.kwargs.get("width", row_stride))
        column = substitute(cast(Expr, node.kwargs.get("column_offset", data_region.ranges[1][0])), substitutions)
        free_lower = Add(
            left=Mul(left=row_lower, right=Const(value=row_stride)), right=Add(left=free_lower, right=column)
        )
    offset_kind = "scalar_offset" if scalar else "vector_offset"
    indirect = f"{hbm_region.tensor}.ap(pattern=[[{row_stride}, {partition}], [1, {free}]], offset={format_expr(free_lower)}, {offset_kind}={index_text}, indirect_dim=0)"
    source, destination = (indirect, data_text) if gather else (data_text, indirect)
    descriptor_mode = "hwdge" if scalar else "swdge"
    return f"nisa.dma_copy(src={source}, dst={destination}, oob_mode=oob_mode.error, dge_mode=nisa.dge_mode.{descriptor_mode})"


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
"ISA kwargs whose string value names an ``nl`` math operator. ``nisa`` ALU\nops (``tensor_tensor``, ``tensor_scalar``, ``activation``, ``tensor_reduce``)\ntake the operator as an ``nl`` reference (e.g. ``op=nl.add``), not a bare\nstring — so these render as ``nl.<value>`` while every other kwarg renders\nvia ``repr`` (e.g. memset's ``value=0.0``)."


def _render_kwarg(key: str, value: Any) -> str:
    """Render one ISA kwarg value, mapping ALU-operator names to ``nl.<name>``."""
    value = "maximum" if key in {"op", "reduce_op"} and value == "max" else value
    if key == "reduce_cmd" or (key in {"send_to_rank", "recv_from_rank"} and value == "program_peer"):
        return f"nisa.reduce_cmd.{value}" if key == "reduce_cmd" else "1 - nl.program_id(0)"
    if key in _NL_OP_KWARGS | {"engine"} and isinstance(value, str):
        namespace = "nisa.engine" if key == "engine" else "nl"
        return f"{namespace}.{value}"
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return f"float('{value}')"
    return format_expr(value) if isinstance(value, Expr) else repr(value)


def _render_first_write(
    ir: KernelIR, leaf_nid: int, reduction_axes: tuple[str, ...], substitutions: dict[str, Expr]
) -> str:
    """Render accumulation from the final reduction-loop structure."""
    iterations = sum((operation_axis_iterations(ir, leaf_nid, axis, substitutions) for axis in reduction_axes), ())
    return " or ".join((f"{format_expr(iteration)} != 0" for iteration in iterations)) or "False"


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
    if buf.location == "register":
        if rotation is not None or region.ranges != ((Const(value=0), Const(value=1)),) * 2:
            raise ValueError(f"{buf.name}: register operands must select their single scalar")
        return buf.name
    list_subscript = ""
    parts: list[str] = []
    for axis_index, (lo, hi) in enumerate(region.ranges):
        if axis_index == 0 and buf.location != "shared_hbm":
            partition_extent = buf.partition_extent()
            if not isinstance(hi, Const) or not 0 < hi.value <= partition_extent:
                raise AssertionError(f"{buf.name}: partition tile {hi} exceeds {partition_extent}")
            a = buf.tiles_per_list()
            if buf.list_len == 1:
                list_subscript = "[0]"
                slot = _format_raw(lo)
            elif a == 1:
                list_subscript = f"[{_format_raw(lo)}]"
                slot = "0"
            else:
                tile = f"({_format_raw(lo)})"
                list_subscript = f"[{tile} // {a}]"
                slot = f"{tile} % {a}"
            if rotation is not None:
                slot = _format_raw(rotation) if slot == "0" else f"{slot} + {_format_raw(rotation)}"
            parts.extend((f"0:{hi.value}", slot))
        else:
            lo_str = _format_raw(lo)
            hi_str = _format_raw(hi)
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
        (f"[{format_expr(stride)}, {format_expr(extent)}]" for stride, extent in access_pattern.pattern)
    )
    offset = access_pattern.offset
    result = _format_raw(offset)
    if rotation is not None:
        free = buf.per_tile_physical_shape()[2]
        flattened = rotation if free == 1 else Mul(left=rotation, right=Const(value=free))
        rotation_text = _format_raw(flattened)
        result = rotation_text if isinstance(offset, Const) and offset.value == 0 else f"{result} + {rotation_text}"
    return f"{base}.ap(pattern=[{dimensions}], offset={result})"


__all__ = ["emit_body", "render_access_pattern", "render_buffer_region"]
