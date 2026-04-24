"""``render_ir``: mechanical lowering of :class:`KernelIR` to NKI source.

Follows the recipe in ``/home/ubuntu/nki-autotune/nkigym/src/nkigym/design.md``:

1. Header — function signature + param asserts + HBM output ndarray.
2. Loop skeleton — nested ``for i_block_<d> in range(num_blocks[d])``
   in ``dim_order`` (a given op only gets the loops for its own dim
   subset; dims not used by any op in the current scope are skipped).
3. Allocation emission — at each depth, emit ``allocate_buffers`` for
   every physical buffer whose ``emission_depth`` matches. Buffers with
   ``num_buffers = None`` on both axes are emitted at each op's
   tightest-enclosing-loop site.
4. Per-op emission — gadget dispatch via ``Op.kind`` and ``Op.attrs``.
5. Matmul-accumulator memset + HBM store at the derived depth.

This is a single-file backend: no extra modules under ``codegen/``
beyond ``gadgets.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nkigym.kernel_ir.ir import BufferScope, KernelIR, NumBuffers, Op, PhysicalBuffer
from nkigym.kernel_ir.types import DimRole

_GADGETS_IMPORT = (
    "from nkigym.codegen.gadgets import (\n"
    "    activation_block,\n"
    "    activation_reduce_block,\n"
    "    allocate_buffers,\n"
    "    load_block,\n"
    "    matmul_block,\n"
    "    memset_buffers,\n"
    "    store_block,\n"
    "    tensor_scalar_block,\n"
    "    transpose_block,\n"
    ")"
)


def render_ir(ir: KernelIR) -> str:
    """Lower ``ir`` to NKI source code using the gadgets module."""
    writer = _SourceWriter()
    writer.line("import nki")
    writer.line("import nki.language as nl")
    writer.line()
    writer.line(_GADGETS_IMPORT)
    writer.line()
    writer.line()
    writer.line("@nki.jit")
    params = ", ".join(ir.param_names)
    writer.line(f"def {ir.func_name}({params}):")
    writer.indent()
    _emit_header(writer, ir)
    _emit_body(writer, ir)
    writer.line(f"return {ir.return_name}_hbm" if _has_store_op(ir) else f"return {ir.return_name}")
    writer.dedent()
    return writer.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Source writer
# ─────────────────────────────────────────────────────────────────────────────


class _SourceWriter:
    """Tiny line-based writer that tracks indentation."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._depth = 0

    def indent(self) -> None:
        self._depth += 1

    def dedent(self) -> None:
        self._depth -= 1

    def line(self, text: str = "") -> None:
        if text == "":
            self._lines.append("")
        else:
            self._lines.append("    " * self._depth + text)

    def getvalue(self) -> str:
        return "\n".join(self._lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────


def _emit_header(writer: _SourceWriter, ir: KernelIR) -> None:
    """Emit parameter asserts and the HBM output tensor."""
    for p in ir.param_names:
        shape = ir.logical_tensors[p].shape
        writer.line(f"assert {p}.shape == {tuple(shape)}")
    ret_info = ir.logical_tensors[ir.return_name]
    ret_shape = tuple(ret_info.shape)
    ret_dtype = f"nl.{ret_info.dtype}"
    store_op = _find_store_op(ir)
    if store_op is not None:
        out_name = store_op.outputs[0]
        writer.line(f"{out_name} = nl.ndarray({ret_shape}, dtype={ret_dtype}, buffer=nl.shared_hbm)")
    else:
        writer.line(f"{ir.return_name} = nl.ndarray({ret_shape}, dtype={ret_dtype}, buffer=nl.shared_hbm)")
    writer.line()


def _has_store_op(ir: KernelIR) -> bool:
    return _find_store_op(ir) is not None


def _find_store_op(ir: KernelIR) -> Op | None:
    for op in ir.ops:
        if op.kind == "NKIStore":
            return op
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Body
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _BufferAllocInfo:
    """Resolved codegen view of one physical buffer."""

    name: str
    buf: PhysicalBuffer
    num_buffers: NumBuffers
    emission_depth: int
    p_tile: int
    num_p_tiles: int
    f_tile: int
    num_f_tiles: int


@dataclass
class _OpSchedule:
    """Where each op in ``ir.ops`` emits and what loops enclose it."""

    op: Op
    index: int
    loops: list[str] = field(default_factory=list)


def _emit_body(writer: _SourceWriter, ir: KernelIR) -> None:
    """Emit kernel-top allocs, loopnest, per-op calls, stores.

    Algorithm: walk ``ir.ops`` in topological order. For each op,
    adjust the loop stack so it exactly matches the op's required
    ``loops[]``, then emit the op. Matmul accumulator prologues and
    stores fire at the end of the owning loop body once all ops
    that write the accumulator have emitted.
    """
    buf_info = _resolve_buffer_info(ir)
    schedules = _schedule_ops(ir)
    store_op = _find_store_op(ir)

    """Depth 0 — kernel top allocations."""
    _emit_kernel_top_allocs(writer, ir, buf_info)
    writer.line()

    stack: list[str] = []
    ctx = _ScheduleContext(buf_info=buf_info, schedules=schedules, store_op=store_op)

    for sched in schedules:
        if sched.op is store_op:
            continue
        _align_stack_to(writer, ir, ctx, stack, sched.loops)
        _emit_op(writer, ir, buf_info, sched, stack)

    """Close every remaining loop; fire stores + prologues on close."""
    _align_stack_to(writer, ir, ctx, stack, [])


@dataclass
class _ScheduleContext:
    """Per-render-pass state passed through the stack-alignment walk."""

    buf_info: dict[str, _BufferAllocInfo]
    schedules: list[_OpSchedule] = field(default_factory=list)
    store_op: Op | None = None
    prologue_fired: set[int] = field(default_factory=set)


def _align_stack_to(
    writer: _SourceWriter, ir: KernelIR, ctx: _ScheduleContext, stack: list[str], target: list[str]
) -> None:
    """Close loops from ``stack`` until its prefix matches ``target``,
    then open loops so that ``stack == target``.

    Close fires store/accumulator-tail hooks at the outgoing depth.
    Open fires accumulator-prologue hooks at the new depth.
    """
    """Find common prefix length."""
    common = 0
    while common < len(stack) and common < len(target) and stack[common] == target[common]:
        common += 1

    """Close from innermost until stack length == common."""
    while len(stack) > common:
        _on_close_loop(writer, ir, ctx, stack)
        stack.pop()
        writer.dedent()

    """Open until stack == target."""
    while len(stack) < len(target):
        dim = target[len(stack)]
        writer.line(f"for i_block_{dim} in range({ir.num_blocks(dim)}):")
        writer.indent()
        stack.append(dim)
        _on_open_loop(writer, ir, ctx, stack)


def _on_open_loop(writer: _SourceWriter, ir: KernelIR, ctx: _ScheduleContext, stack: list[str]) -> None:
    """After opening a new loop, emit any allocs / prologues owned by this depth."""
    depth = len(stack)
    _emit_allocs_at_depth(writer, ir, ctx.buf_info, depth=depth)
    _emit_accumulator_prologue(writer, ir, ctx.buf_info, stack, ctx.schedules, ctx.prologue_fired)


def _on_close_loop(writer: _SourceWriter, ir: KernelIR, ctx: _ScheduleContext, stack: list[str]) -> None:
    """Before closing the innermost loop, emit store if its owner depth is here."""
    _emit_store_if_owned(writer, ir, ctx.buf_info, stack, ctx.store_op)


# ─────────────────────────────────────────────────────────────────────────────
# Buffer info resolution
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_buffer_info(ir: KernelIR) -> dict[str, _BufferAllocInfo]:
    """Compute per-buffer (p_tile, num_p_tiles, f_tile, num_f_tiles, emission_depth).

    Accumulator buffers (the outputs of matmul ops) have a *derived*
    scope, not one chosen by the sampler: they span full extent on
    every non-ACC output dim, and one block on every ACC-adjacent dim.
    Specifically, along the M axis (the matmul's p_axis=d1 output dim)
    the accumulator must survive the entire d1 iteration space under
    the outermost non-ACC loop, which = full d1 in the common matmul
    shape.
    """
    acc_outputs = _accumulator_output_names(ir)
    result: dict[str, _BufferAllocInfo] = {}
    for name, buf in ir.physical_buffers.items():
        if name in acc_outputs:
            p_tile, num_p_tiles, f_tile, num_f_tiles = _accumulator_axis_counts(ir, buf, acc_outputs[name])
        else:
            """Scopes not declared in the IR default to INNER — scratch buffers
            and single-block working sets."""
            scope = ir.buffer_scopes.get(name, BufferScope.INNER)
            p_tile, num_p_tiles, f_tile, num_f_tiles = _scope_to_axis_counts(ir, buf, scope)
        num_buffers = ir.num_buffers.get(name, NumBuffers())
        emission_depth = ir.emission_depth.get(name, 0)
        result[name] = _BufferAllocInfo(
            name=name,
            buf=buf,
            num_buffers=num_buffers,
            emission_depth=emission_depth,
            p_tile=p_tile,
            num_p_tiles=num_p_tiles,
            f_tile=f_tile,
            num_f_tiles=num_f_tiles,
        )
    return result


def _accumulator_output_names(ir: KernelIR) -> dict[str, Op]:
    """Map ``{sbuf_name: matmul_op}`` for every matmul's output."""
    result: dict[str, Op] = {}
    for op in ir.ops:
        if op.kind == "NKIMatmul":
            for out in op.outputs:
                result[out] = op
    return result


def _accumulator_axis_counts(ir: KernelIR, buf: PhysicalBuffer, matmul_op: Op) -> tuple[int, int, int, int]:
    """Compute (p_tile, num_p_tiles, f_tile, num_f_tiles) for a matmul
    accumulator.

    The derived rule for each output axis:

    * If the axis is **below** the matmul's ACC dim in ``dim_order``
      (iterates inside the reduction), the accumulator must hold every
      tile visited while the reduction is open → full-dim extent.
    * If the axis is **above** the ACC dim (iterates outside the
      reduction), each outer-iter gets a fresh accumulator → one-block
      extent (``ltiles_per_block`` tiles).
    """
    p_tile, num_p_tiles = _accumulator_extent(ir, matmul_op, buf.p_axis)
    if buf.f_axis is None:
        return p_tile, num_p_tiles, 1, 1
    f_tile, num_f_tiles = _accumulator_extent(ir, matmul_op, buf.f_axis)
    return p_tile, num_p_tiles, f_tile, num_f_tiles


def _accumulator_extent(ir: KernelIR, matmul_op: Op, axis: str) -> tuple[int, int]:
    """(tile, num_tiles) for one accumulator axis per the derived rule."""
    info = ir.dimensions[axis]
    ptile = info.physical_tile_size
    full_tiles = info.dim_size // ptile
    block_tiles = ir.ltiles_per_block[axis]
    acc_dims = [d for d in matmul_op.blocking_dims if ir.dimensions[d].role is DimRole.ACCUMULATION]
    if not acc_dims:
        return ptile, block_tiles
    order = ir.dim_order
    axis_pos = order.index(axis) if axis in order else -1
    first_acc_pos = min(order.index(d) for d in acc_dims if d in order)
    if axis_pos > first_acc_pos:
        """Axis iterates inside the ACC loop → need to hold everything
        the reduction visits → full-dim tiles."""
        return ptile, full_tiles
    """Axis iterates outside the ACC loop → per-iter fresh accumulator,
    holds one block's worth of tiles."""
    return ptile, block_tiles


def _scope_to_axis_counts(ir: KernelIR, buf: PhysicalBuffer, scope: BufferScope) -> tuple[int, int, int, int]:
    """Map ``buffer_scopes`` entry to ``(p_tile, num_p_tiles, f_tile, num_f_tiles)``.

    ``INNER`` — both axes tile-sized (one block on each dim).
    ``MIDDLE`` — outermost-in-``dim_order`` is per-block; the other is full.
    ``OUTER`` — both axes span their full extent.
    """
    p_axis = buf.p_axis
    f_axis = buf.f_axis
    p_info = ir.dimensions[p_axis]
    if f_axis is None:
        """1-D tensor lifted to 2D with f=1."""
        p_tile = _axis_tile_size(ir, p_axis, scope, is_outermost_in_order(ir, p_axis, [p_axis]))
        num_p_tiles = _axis_num_tiles(ir, p_axis, scope, is_outermost_in_order(ir, p_axis, [p_axis]))
        return p_tile, num_p_tiles, 1, 1

    f_info = ir.dimensions[f_axis]
    """For 2D buffers, MIDDLE means outermost-in-dim_order per-block, inner full."""
    axes = [a for a in ir.dim_order if a in (p_axis, f_axis)]
    p_is_outer = axes and axes[0] == p_axis
    f_is_outer = axes and axes[0] == f_axis

    p_tile = _axis_tile_size(ir, p_axis, scope, p_is_outer)
    num_p_tiles = _axis_num_tiles(ir, p_axis, scope, p_is_outer)
    f_tile = _axis_tile_size(ir, f_axis, scope, f_is_outer)
    num_f_tiles = _axis_num_tiles(ir, f_axis, scope, f_is_outer)
    return p_tile, num_p_tiles, f_tile, num_f_tiles


def is_outermost_in_order(ir: KernelIR, axis: str, buf_axes: list[str]) -> bool:
    """True iff ``axis`` is the outermost of ``buf_axes`` in ``dim_order``."""
    for d in ir.dim_order:
        if d in buf_axes:
            return d == axis
    return False


def _axis_tile_size(ir: KernelIR, dim: str, scope: BufferScope, is_outer_in_order: bool) -> int:
    """How wide one list-slot's tile is along this axis."""
    info = ir.dimensions[dim]
    if scope is BufferScope.INNER:
        return info.physical_tile_size
    if scope is BufferScope.OUTER:
        return info.physical_tile_size
    """MIDDLE: outermost-in-order axis stays tile-sized; the other stays ptile (full goes into num_tiles)."""
    return info.physical_tile_size


def _axis_num_tiles(ir: KernelIR, dim: str, scope: BufferScope, is_outer_in_order: bool) -> int:
    """How many list-slots along this axis."""
    info = ir.dimensions[dim]
    num_ltile = info.dim_size // info.physical_tile_size
    ltiles_per_block = ir.ltiles_per_block[dim]
    if scope is BufferScope.INNER:
        return ltiles_per_block
    if scope is BufferScope.OUTER:
        return num_ltile
    """MIDDLE: outermost-in-order axis is per-block, inner is full."""
    if is_outer_in_order:
        return ltiles_per_block
    return num_ltile


# ─────────────────────────────────────────────────────────────────────────────
# Schedule: which loops enclose each op
# ─────────────────────────────────────────────────────────────────────────────


def _schedule_ops(ir: KernelIR) -> list[_OpSchedule]:
    """Compute enclosing-loop list for each op, then reorder for emission.

    Each op's ``loops`` is the ``dim_order``-filtered subset of its
    required dims. Ops are re-sorted into emission order so that loops
    open and close monotonically: we use a stable DFS over the
    loop-prefix trie, emitting ops with shorter prefixes before ops
    with longer ones at the same branch. Data dependencies (``edges``)
    break ties within a single prefix group.
    """
    schedules: list[_OpSchedule] = []
    for i, op in enumerate(ir.ops):
        dims = _op_dim_set(ir, op)
        loops = [d for d in ir.dim_order if d in dims]
        schedules.append(_OpSchedule(op=op, index=i, loops=loops))
    return _reorder_by_loop_prefix(ir, schedules)


def _reorder_by_loop_prefix(ir: KernelIR, schedules: list[_OpSchedule]) -> list[_OpSchedule]:
    """Reorder so loops open/close monotonically, respecting data deps.

    Primary sort: topological rank (producer before consumer).
    Within one rank, we can't reorder, but consecutive same-rank ops
    naturally share loop prefixes because they originate from the same
    math-level sub-expression.

    To avoid unnecessary loop open/close between ops whose prefixes
    differ, we greedily extend a "current prefix" as long as the next
    op's loops start with it; if not, we fall back to the next topo-
    compatible op whose loops share the longest common prefix with
    what's currently open.
    """
    topo_rank = _topological_ranks(ir)
    remaining = sorted(schedules, key=lambda s: topo_rank.get(s.index, s.index))
    ready: list[_OpSchedule] = []
    ordered: list[_OpSchedule] = []
    """Track which producers have emitted (by tensor name)."""
    produced: set[str] = set()
    """Seed with ops whose inputs are kernel params / already produced."""
    dim_idx = {d: i for i, d in enumerate(ir.dim_order)}

    def deps_satisfied(s: _OpSchedule) -> bool:
        """True iff every tensor-valued input has been produced or is a kernel input."""
        for tname in s.op.inputs.values():
            if tname in ir.param_names:
                continue
            if tname not in produced and ir.has_tensor(tname):
                if ir.producer_of(tname) is not None and tname not in produced:
                    return False
        return True

    """Build the ordered list: at each step, among deps-satisfied ops,
    prefer the one with longest loop-prefix match to the previous op."""
    current_prefix: tuple[str, ...] = ()
    while remaining:
        """Partition into ready / not-ready."""
        ready_mask: list[bool] = [deps_satisfied(s) for s in remaining]
        ready_pool = [s for s, r in zip(remaining, ready_mask) if r]
        if not ready_pool:
            """Cycle or missing producer — fall through by taking the first
            in topo rank to make progress."""
            ready_pool = [remaining[0]]
        """Pick the ready op that best fits the current loop context:
        first by longest prefix match, then by shortest total depth
        (so that independent ops with fewer enclosing loops — e.g.
        a shallower d0-only load — emit before a d0×d1 inner op),
        then by topo rank for stability."""
        best = min(
            ready_pool,
            key=lambda s: (
                -_common_prefix_len(current_prefix, tuple(s.loops)),
                len(s.loops),
                topo_rank.get(s.index, s.index),
            ),
        )
        ordered.append(best)
        remaining.remove(best)
        current_prefix = tuple(best.loops)
        for out in best.op.outputs:
            produced.add(out)
    return ordered


def _common_prefix_len(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    """Length of the common prefix of two tuples."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _topological_ranks(ir: KernelIR) -> dict[int, int]:
    """Topological rank for each op index based on ``ir.edges``.

    Lower rank ⇒ earlier in dep order. Producers have lower rank
    than consumers.
    """
    rank: dict[int, int] = {}
    n = len(ir.ops)
    indeg: dict[int, int] = dict.fromkeys(range(n), 0)
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for producer_idx, consumer_idx, _tensor, _role in ir.edges:
        if 0 <= producer_idx < n and 0 <= consumer_idx < n and producer_idx != consumer_idx:
            adj[producer_idx].append(consumer_idx)
            indeg[consumer_idx] += 1
    queue = [i for i in range(n) if indeg[i] == 0]
    order = 0
    while queue:
        i = queue.pop(0)
        rank[i] = order
        order += 1
        for j in adj[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                queue.append(j)
    for i in range(n):
        rank.setdefault(i, order + i)
    return rank


def _op_dim_set(ir: KernelIR, op: Op) -> set[str]:
    """Dims the op's loops must iterate over.

    * Matmul / compute ops: union of input + output tensor dim_ids +
      any ``blocking_dims``.
    * Load: dims whose HBM slice varies (i.e. dim extent of the
      destination buffer < full dim size), plus enclosing loops of
      the destination buffer's ``emission_depth``.
    * Other compute ops on a wider-than-one-block buffer: same rule
      as Load — only include dims where the buffer doesn't span full
      extent.
    * Store: handled separately; never scheduled here.
    """
    if op.kind == "NKIStore":
        return set()
    if op.kind == "NKILoad":
        return _load_dim_set(ir, op)
    dims: set[str] = set(op.blocking_dims)
    for tname in list(op.inputs.values()) + list(op.outputs):
        if ir.has_tensor(tname):
            dims.update(_varying_dims_for_tensor(ir, tname))
    return dims


def _load_dim_set(ir: KernelIR, op: Op) -> set[str]:
    """For NKILoad: dims whose HBM slice varies + emission_depth loops."""
    dst = op.outputs[0]
    dims: set[str] = set()
    if dst in ir.physical_buffers:
        dims.update(_varying_dims_for_tensor(ir, dst))
    depth = ir.emission_depth.get(dst, 0)
    for d in ir.dim_order[:depth]:
        dims.add(d)
    return dims


def _varying_dims_for_tensor(ir: KernelIR, tname: str) -> set[str]:
    """Dims along which this tensor's slice varies per block-iter.

    For a logical tensor, all dim_ids are "varying" (one block at a
    time). For a physical buffer, a dim is varying only if the
    buffer's extent along that axis is smaller than the full dim size
    (i.e. scope is INNER or MIDDLE-on-this-axis rather than OUTER).
    """
    info = ir.tensor_info(tname)
    if tname not in ir.physical_buffers:
        return set(info.dim_ids)
    buf = ir.physical_buffers[tname]
    dims: set[str] = set()
    return _varying_dims_from_buffer(ir, buf)


def _varying_dims_from_buffer(ir: KernelIR, buf: PhysicalBuffer) -> set[str]:
    """For a physical buffer, dims whose extent < full dim size."""
    dims: set[str] = set()
    name = _buffer_name_of(ir, buf)
    is_acc = _is_accumulator(ir, name) if name else False
    scope = ir.buffer_scopes.get(name) if name else None
    if scope is None and not is_acc:
        """Scratch/scalar buffers default to INNER (one block per use)."""
        scope = BufferScope.INNER
    """Resolve leaf extents via the same rule as _scope_to_axis_counts."""
    for axis_name, axis_id in (("p", buf.p_axis), ("f", buf.f_axis)):
        if axis_id is None:
            continue
        info = ir.dimensions[axis_id]
        extent = _axis_extent(ir, buf, axis_id, scope)
        if extent < info.dim_size:
            dims.add(axis_id)
    return dims


def _is_accumulator(ir: KernelIR, name: str | None) -> bool:
    """True iff ``name`` is a matmul output (derived-scope accumulator)."""
    if name is None:
        return False
    for op in ir.ops:
        if op.kind == "NKIMatmul" and name in op.outputs:
            return True
    return False


def _buffer_name_of(ir: KernelIR, buf: PhysicalBuffer) -> str | None:
    """Reverse-lookup the buffer's key name in ``physical_buffers``."""
    for name, b in ir.physical_buffers.items():
        if b is buf:
            return name
    return None


def _axis_extent(ir: KernelIR, buf: PhysicalBuffer, axis_id: str, scope: BufferScope | None) -> int:
    """Size of the buffer along one of its axes.

    Accumulator buffers have derived extents: p-axis spans full dim;
    f-axis spans one-block. Everything else is driven by ``scope``.
    """
    if scope is None:
        """Derived (accumulator) extents."""
        if axis_id == buf.p_axis:
            return ir.dimensions[axis_id].dim_size
        return ir.block_extent(axis_id)
    info = ir.dimensions[axis_id]
    if scope is BufferScope.INNER:
        return ir.block_extent(axis_id)
    if scope is BufferScope.OUTER:
        return info.dim_size
    """MIDDLE: outermost-in-dim_order is per-block, inner is full."""
    buf_axes = [buf.p_axis, buf.f_axis] if buf.f_axis else [buf.p_axis]
    outer_axis = None
    for d in ir.dim_order:
        if d in buf_axes:
            outer_axis = d
            break
    if axis_id == outer_axis:
        return ir.block_extent(axis_id)
    return info.dim_size


# ─────────────────────────────────────────────────────────────────────────────
# Allocation emission
# ─────────────────────────────────────────────────────────────────────────────


def _emit_kernel_top_allocs(writer: _SourceWriter, ir: KernelIR, buf_info: dict[str, _BufferAllocInfo]) -> None:
    """Emit allocations for buffers with emission_depth == 0 (and multi-buffered).

    Compiler-offload buffers (both ``num_buffers`` axes ``None``) are
    emitted at each op's tightest-enclosing-loop site, not here.
    """
    _emit_allocs_at_depth(writer, ir, buf_info, depth=0)


def _emit_allocs_at_depth(
    writer: _SourceWriter, ir: KernelIR, buf_info: dict[str, _BufferAllocInfo], depth: int
) -> None:
    """Emit every multi-buffered allocation whose ``emission_depth == depth``."""
    for name, info in buf_info.items():
        if info.num_buffers.is_compiler_offload:
            continue
        if info.emission_depth != depth:
            continue
        writer.line(_alloc_call(name, info))


def _alloc_call(name: str, info: _BufferAllocInfo) -> str:
    """Render the ``allocate_buffers(...)`` call for one buffer."""
    nb = info.num_buffers
    p = "None" if nb.num_p_buffers is None else str(nb.num_p_buffers)
    f = "None" if nb.num_f_buffers is None else str(nb.num_f_buffers)
    return (
        f"{name} = allocate_buffers({info.p_tile}, {info.num_p_tiles}, "
        f"{info.f_tile}, {info.num_f_tiles}, nl.sbuf, nl.{info.buf.dtype}, "
        f"num_p_buffers={p}, num_f_buffers={f})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Accumulator handling
# ─────────────────────────────────────────────────────────────────────────────


def _emit_accumulator_prologue(
    writer: _SourceWriter,
    ir: KernelIR,
    buf_info: dict[str, _BufferAllocInfo],
    stack: list[str],
    schedules: list[_OpSchedule],
    already_fired: set[int] | None = None,
) -> None:
    """Emit ``cur_<name> = <name>[...]; memset_buffers(...)`` at the top of
    the outermost non-ACC loop that encloses the output.

    The accumulator lives outside every ACCUMULATION loop so its state
    persists across the reduction. It's freshly memset at the top of
    the outermost enclosing non-ACC loop (the widest scope where the
    accumulator's contents are ``output``-local).
    """
    fired = already_fired if already_fired is not None else set()
    for sched in schedules:
        op = sched.op
        if op.kind != "NKIMatmul":
            continue
        if id(op) in fired:
            continue
        if not _accumulator_prologue_stack_matches(ir, op, stack):
            continue
        out_name = op.outputs[0]
        sbuf_name = out_name if out_name.startswith("sbuf_") else f"sbuf_{out_name}"
        if sbuf_name not in buf_info:
            continue
        info = buf_info[sbuf_name]
        if info.num_buffers.is_compiler_offload:
            cur = f"cur_{sbuf_name}"
            writer.line(_alloc_call(cur, info))
            writer.line(f"memset_buffers({cur}, 0.0)")
        else:
            _emit_cur_slot_binding(writer, sbuf_name, info, stack)
            writer.line(f"memset_buffers(cur_{sbuf_name}, 0.0)")
        fired.add(id(op))


def _accumulator_prologue_depth(ir: KernelIR, matmul_op: Op) -> int:
    """Depth where the accumulator's ``cur_<name> + memset`` prologue fires.

    Computed relative to the matmul's own enclosing-loop list. The
    prologue fires at the innermost non-ACC loop that still encloses
    the ACC loop: i.e. the loop-list index of the first ACC dim among
    the matmul's loops.

    Example (matmul loops ``[d1, d2, d0]``, d0 = ACC):
    prologue depth = 2 (inside d1 and d2, above d0).
    """
    loops = [d for d in ir.dim_order if d in _op_dim_set(ir, matmul_op)]
    for i, d in enumerate(loops):
        if ir.dimensions[d].role is DimRole.ACCUMULATION:
            return i
    return len(loops)


def _accumulator_prologue_stack_matches(ir: KernelIR, matmul_op: Op, stack: list[str]) -> bool:
    """True iff the current stack is the matmul's non-ACC-loop prefix.

    The prologue fires exactly once, when the open loops are the
    matmul's loops up to (but not including) the first ACC loop.
    """
    loops = [d for d in ir.dim_order if d in _op_dim_set(ir, matmul_op)]
    depth = _accumulator_prologue_depth(ir, matmul_op)
    return tuple(stack) == tuple(loops[:depth])


def _emit_cur_slot_binding(writer: _SourceWriter, sbuf_name: str, info: _BufferAllocInfo, stack: list[str]) -> None:
    """Emit ``cur_<name> = <name>[...]`` rotating on active axes."""
    idx = _rotation_index(info, stack)
    writer.line(f"cur_{sbuf_name} = {sbuf_name}{idx}")


def _rotation_index(info: _BufferAllocInfo, stack: list[str]) -> str:
    """Build the bracketed rotation index for multi-buffered slot selection."""
    nb = info.num_buffers
    parts: list[str] = []
    if nb.num_p_buffers is not None:
        parts.append(f"[i_block_{info.buf.p_axis} % {nb.num_p_buffers}]")
    if nb.num_f_buffers is not None and info.buf.f_axis is not None:
        parts.append(f"[i_block_{info.buf.f_axis} % {nb.num_f_buffers}]")
    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Per-op emission
# ─────────────────────────────────────────────────────────────────────────────


def _emit_op(
    writer: _SourceWriter, ir: KernelIR, buf_info: dict[str, _BufferAllocInfo], sched: _OpSchedule, stack: list[str]
) -> None:
    """Dispatch to the right gadget based on ``op.kind``."""
    op = sched.op
    kind = op.kind
    if kind == "NKILoad":
        _emit_load(writer, ir, buf_info, op, stack)
    elif kind == "NKIStore":
        """Stores are handled by _emit_store_if_owned; skip here."""
    elif kind == "NKIActivationReduce":
        _emit_activation_reduce(writer, ir, buf_info, op, stack)
    elif kind == "NKITensorScalar":
        _emit_tensor_scalar(writer, ir, buf_info, op, stack)
    elif kind == "NKIActivation":
        _emit_activation(writer, ir, buf_info, op, stack)
    elif kind == "NKITranspose":
        _emit_transpose(writer, ir, buf_info, op, stack)
    elif kind == "NKIMatmul":
        _emit_matmul(writer, ir, buf_info, op, stack)
    else:
        raise NotImplementedError(f"render: op kind {kind!r} not supported")


# ---------- NKILoad ----------


def _emit_load(
    writer: _SourceWriter, ir: KernelIR, buf_info: dict[str, _BufferAllocInfo], op: Op, stack: list[str]
) -> None:
    """Emit ``load_block(sbuf, mem_slice, transpose=...)`` for an NKILoad op.

    Also emits the alloc if the buffer is compiler-offload
    (``num_buffers = None`` on both axes).
    """
    src_name = op.inputs["data"]
    dst_name = op.outputs[0]
    info = buf_info[dst_name]
    transpose = bool(op.attrs.get("transpose", False))

    if info.num_buffers.is_compiler_offload:
        writer.line(_alloc_call(dst_name, info))
        dst_expr = dst_name
    else:
        _emit_cur_slot_binding(writer, dst_name, info, stack)
        dst_expr = f"cur_{dst_name}"

    mem_slice = _hbm_slice(ir, info, src_name, transpose)
    writer.line(f"load_block({dst_expr}, {mem_slice}, transpose={transpose})")


def _hbm_slice(ir: KernelIR, info: _BufferAllocInfo, src: str, transpose: bool) -> str:
    """Build the ``src[p0:p1, f0:f1]`` slice expression for the HBM side of load.

    For ``transpose=True``, the HBM slice is indexed in
    ``(free_axis, partition_axis)`` order — the dma_transpose contract.
    """
    buf = info.buf
    p_axis = buf.p_axis
    f_axis = buf.f_axis
    p_extent = info.p_tile * info.num_p_tiles
    f_extent = info.f_tile * info.num_f_tiles
    p_expr = _axis_offset_expr(ir, p_axis, p_extent)
    f_expr = _axis_offset_expr(ir, f_axis, f_extent) if f_axis else None

    """Order slices according to the src logical-tensor's dim order."""
    src_dims = ir.tensor_info(src).dim_ids
    axes = {p_axis: p_expr}
    if f_axis:
        axes[f_axis] = f_expr
    if transpose:
        """After load+transpose, mem_slice is indexed (f_axis, p_axis); but
        load_block(transpose=True) expects mem_slice.shape == (f_tile, p_tile * num_p_tiles)."""
        slices = [axes.get(d, f"0:{ir.dimensions[d].dim_size}") for d in (f_axis, p_axis) if d is not None]
    else:
        slices = [axes.get(d, f"0:{ir.dimensions[d].dim_size}") for d in src_dims]
    return f"{src}[{', '.join(slices)}]"


def _axis_offset_expr(ir: KernelIR, dim: str, extent: int) -> str:
    """Slice ``i_block_<dim> * extent : ... + extent``.

    Degenerate cases:
    * If the axis isn't a block loop in ``dim_order`` → full-dim slice.
    * If ``extent`` equals the full dim size → full-dim slice (the
      buffer spans the entire dim; no block indexing needed).
    """
    full = ir.dimensions[dim].dim_size
    if extent >= full:
        return f"0:{full}"
    if dim not in ir.dim_order:
        return f"0:{full}"
    return f"i_block_{dim} * {extent} : i_block_{dim} * {extent} + {extent}"


# ---------- NKIActivationReduce ----------


def _emit_activation_reduce(
    writer: _SourceWriter, ir: KernelIR, buf_info: dict[str, _BufferAllocInfo], op: Op, stack: list[str]
) -> None:
    """Emit sum_sq allocator + ``activation_reduce_block(...)``."""
    src = op.inputs["data"]
    src_sbuf = _sbuf_of(ir, src)
    out = op.outputs[0]
    out_sbuf = _sbuf_of(ir, out)
    out_info = buf_info[out_sbuf]
    if out_info.num_buffers.is_compiler_offload:
        writer.line(_alloc_call(out_sbuf, out_info))
        writer.line(f"memset_buffers({out_sbuf}, 0.0)")
    activation = op.kwargs.get("op", "square")
    reduce_op = op.kwargs.get("reduce_op", "add")
    writer.line(f"activation_reduce_block({out_sbuf}, {src_sbuf}, op=nl.{activation}, reduce_op=nl.{reduce_op})")


# ---------- NKITensorScalar ----------


def _emit_tensor_scalar(
    writer: _SourceWriter, ir: KernelIR, buf_info: dict[str, _BufferAllocInfo], op: Op, stack: list[str]
) -> None:
    """Emit ``tensor_scalar_block(dst, src, op0=..., operand0=..., ...)``.

    ``operand0`` may be either a scalar literal (Python float) or the
    name of another tensor (e.g. ``rsqrt_val``); we pass the SBUF
    version of the latter into the gadget (which handles per-partition
    vector broadcast).
    """
    src = op.inputs["data"]
    src_sbuf = _sbuf_of(ir, src)
    out = op.outputs[0]
    out_sbuf = _sbuf_of(ir, out)
    out_info = buf_info[out_sbuf]
    if out_info.num_buffers.is_compiler_offload:
        writer.line(_alloc_call(out_sbuf, out_info))
    op0 = op.kwargs.get("op0", "multiply")
    operand0 = _operand_expr(ir, op.kwargs.get("operand0"))
    parts = [f"tensor_scalar_block({out_sbuf}, {src_sbuf}", f"op0=nl.{op0}", f"operand0={operand0}"]
    if "op1" in op.kwargs:
        op1 = op.kwargs["op1"]
        operand1 = _operand_expr(ir, op.kwargs.get("operand1"))
        parts.extend([f"op1=nl.{op1}", f"operand1={operand1}"])
    writer.line(", ".join(parts) + ")")


def _operand_expr(ir: KernelIR, operand: Any) -> str:
    """If ``operand`` is a tensor name, return its SBUF alias; else Python-repr the literal."""
    if isinstance(operand, str) and ir.has_tensor(operand):
        return _sbuf_of(ir, operand)
    return repr(operand)


# ---------- NKIActivation ----------


def _emit_activation(
    writer: _SourceWriter, ir: KernelIR, buf_info: dict[str, _BufferAllocInfo], op: Op, stack: list[str]
) -> None:
    """Emit ``activation_block(dst, src, op=...)``."""
    src = op.inputs["data"]
    src_sbuf = _sbuf_of(ir, src)
    out = op.outputs[0]
    out_sbuf = _sbuf_of(ir, out)
    out_info = buf_info[out_sbuf]
    if out_info.num_buffers.is_compiler_offload and out_sbuf != src_sbuf:
        writer.line(_alloc_call(out_sbuf, out_info))
    activation = op.kwargs.get("op", "rsqrt")
    writer.line(f"activation_block({out_sbuf}, {src_sbuf}, op=nl.{activation})")


# ---------- NKITranspose ----------


def _emit_transpose(
    writer: _SourceWriter, ir: KernelIR, buf_info: dict[str, _BufferAllocInfo], op: Op, stack: list[str]
) -> None:
    """Emit ``transpose_block(dst, src)`` for ``mode="dma_transpose"``."""
    mode = op.attrs.get("mode", "dma_transpose")
    if mode != "dma_transpose":
        raise NotImplementedError(f"NKITranspose.mode={mode!r} not supported (only dma_transpose)")
    src = op.inputs["data"]
    src_sbuf = _sbuf_of(ir, src)
    out = op.outputs[0]
    out_sbuf = _sbuf_of(ir, out)
    out_info = buf_info[out_sbuf]
    if out_info.num_buffers.is_compiler_offload:
        writer.line(_alloc_call(out_sbuf, out_info))
    writer.line(f"transpose_block({out_sbuf}, {src_sbuf})")


# ---------- NKIMatmul ----------


def _emit_matmul(
    writer: _SourceWriter, ir: KernelIR, buf_info: dict[str, _BufferAllocInfo], op: Op, stack: list[str]
) -> None:
    """Emit ``matmul_block(sbuf_out_slice, sbuf_lhs_T, sbuf_rhs)``.

    - ``sbuf_lhs_T``: the op's ``stationary`` input. If MIDDLE-scope
      (full K), slice by current d0-block; otherwise pass directly.
    - ``sbuf_rhs``: the ``moving`` input; same slicing.
    - ``sbuf_out``: the output accumulator. The caller has already
      bound ``cur_<name>`` and memset'd it at the d2-iter top.
    """
    axis_map = op.axis_map
    k_dim = axis_map.get("K")
    m_dim = axis_map.get("M")
    n_dim = axis_map.get("N")

    lhs = op.inputs["stationary"]
    rhs = op.inputs["moving"]
    out = op.outputs[0]
    lhs_sbuf = _sbuf_of(ir, lhs)
    rhs_sbuf = _sbuf_of(ir, rhs)
    out_sbuf = _sbuf_of(ir, out)

    lhs_expr = _matmul_lhs_expr(ir, buf_info, lhs_sbuf, k_dim)
    rhs_expr = _matmul_rhs_expr(ir, buf_info, rhs_sbuf, k_dim)
    out_expr = _matmul_out_expr(ir, buf_info, out_sbuf, m_dim)

    writer.line(f"matmul_block({out_expr}, {lhs_expr}, {rhs_expr})")


def _matmul_operand_expr(ir: KernelIR, buf_info: dict[str, _BufferAllocInfo], sbuf_name: str, k_dim: str) -> str:
    """Build a matmul input argument expression (lhs or rhs).

    If the buffer is multi-buffered, use its ``cur_`` binding (emitted
    earlier when the load fired). Otherwise use the bare buffer name.
    If the buffer spans the full K axis (MIDDLE / OUTER on K), slice
    by the current d0-block.
    """
    info = buf_info[sbuf_name]
    base = sbuf_name if info.num_buffers.is_compiler_offload else f"cur_{sbuf_name}"
    if info.buf.p_axis == k_dim and _axis_spans_full(ir, sbuf_name, k_dim):
        lt_k = ir.ltiles_per_block[k_dim]
        return f"{base}[i_block_{k_dim} * {lt_k} : i_block_{k_dim} * {lt_k} + {lt_k}]"
    if info.buf.f_axis == k_dim and _axis_spans_full(ir, sbuf_name, k_dim):
        lt_k = ir.ltiles_per_block[k_dim]
        return f"{base}[i_block_{k_dim} * {lt_k} : i_block_{k_dim} * {lt_k} + {lt_k}]"
    return base


def _matmul_lhs_expr(ir: KernelIR, buf_info: dict[str, _BufferAllocInfo], sbuf_name: str, k_dim: str) -> str:
    """Stationary argument."""
    return _matmul_operand_expr(ir, buf_info, sbuf_name, k_dim)


def _matmul_rhs_expr(ir: KernelIR, buf_info: dict[str, _BufferAllocInfo], sbuf_name: str, k_dim: str) -> str:
    """Moving argument."""
    return _matmul_operand_expr(ir, buf_info, sbuf_name, k_dim)


def _matmul_out_expr(ir: KernelIR, buf_info: dict[str, _BufferAllocInfo], sbuf_name: str, m_dim: str) -> str:
    """Output accumulator slice — the M-tiles this iteration writes.

    The matmul writes one M-block (``ltiles_per_block[M]`` M-tiles) per
    d1-iter. If the buffer holds more M-tiles than that, slice to the
    current d1-block.
    """
    info = buf_info[sbuf_name]
    base = sbuf_name if info.num_buffers.is_compiler_offload else f"cur_{sbuf_name}"
    lt_m = ir.ltiles_per_block[m_dim]
    if info.buf.p_axis == m_dim and info.num_p_tiles > lt_m:
        return f"{base}[i_block_{m_dim} * {lt_m} : i_block_{m_dim} * {lt_m} + {lt_m}]"
    return base


def _axis_is_middle_full(ir: KernelIR, sbuf_name: str) -> bool:
    """True iff this buffer's scope is MIDDLE — i.e. one axis spans full extent."""
    scope = ir.buffer_scopes.get(sbuf_name)
    return scope is BufferScope.MIDDLE or scope is BufferScope.OUTER


def _axis_spans_full(ir: KernelIR, sbuf_name: str, axis: str) -> bool:
    """True iff the given axis in this buffer's scope spans full dim extent."""
    scope = ir.buffer_scopes.get(sbuf_name, BufferScope.INNER)
    buf = ir.physical_buffers[sbuf_name]
    if scope is BufferScope.OUTER:
        return True
    if scope is BufferScope.INNER:
        return False
    """MIDDLE: outermost-in-dim_order axis is per-block; the other is full."""
    buf_axes = [buf.p_axis, buf.f_axis] if buf.f_axis else [buf.p_axis]
    for d in ir.dim_order:
        if d in buf_axes:
            return d != axis
    return False


# ---------- NKIStore ----------


def _emit_store_if_owned(
    writer: _SourceWriter, ir: KernelIR, buf_info: dict[str, _BufferAllocInfo], stack: list[str], store_op: Op | None
) -> None:
    """Emit store at the tail of the loop that owns the output strip.

    Store depth = 1 + max(depth of any output dim that's above all
    ACCUMULATION dims in ``dim_order``).
    """
    if store_op is None:
        return
    src_name = store_op.inputs["data"]
    sbuf_src = _sbuf_of(ir, src_name)
    if sbuf_src not in buf_info:
        return
    if not _store_stack_matches(ir, sbuf_src, stack):
        return
    out_dims = ir.tensor_info(sbuf_src).dim_ids
    """Only emit once — track on store_op."""
    if store_op.attrs.get("_emitted"):
        return
    store_op.attrs["_emitted"] = True

    info = buf_info[sbuf_src]
    mem_expr = _store_mem_slice(ir, info, out_dims)
    producer = ir.producer_of(sbuf_src)
    is_matmul_output = producer is not None and ir.ops[producer].kind == "NKIMatmul"
    if is_matmul_output:
        src_expr = f"cur_{sbuf_src}"
    elif info.num_buffers.is_compiler_offload:
        src_expr = sbuf_src
    else:
        src_expr = f"cur_{sbuf_src}"
    hbm_name = store_op.outputs[0]
    writer.line(f"store_block({hbm_name}{mem_expr}, {src_expr})")


def _store_stack_matches(ir: KernelIR, sbuf_src: str, stack: list[str]) -> bool:
    """True iff the current stack matches the store's owning-loop prefix.

    If the source is a matmul accumulator, the store must match the
    matmul's own non-ACC-loop prefix (same as the prologue condition).
    Otherwise, store at the tightest enclosing loop of the source's
    dim set.
    """
    producer = ir.producer_of(sbuf_src)
    if producer is not None:
        prod_op = ir.ops[producer]
        if prod_op.kind == "NKIMatmul":
            return _accumulator_prologue_stack_matches(ir, prod_op, stack)
    src_dims = ir.tensor_info(sbuf_src).dim_ids
    target = [d for d in ir.dim_order if d in src_dims]
    return tuple(stack) == tuple(target)


def _store_mem_slice(ir: KernelIR, info: _BufferAllocInfo, out_dims: tuple[str, ...]) -> str:
    """Slice expression into the HBM output tensor."""
    buf = info.buf
    p_axis = buf.p_axis
    f_axis = buf.f_axis
    p_extent = info.p_tile * info.num_p_tiles
    f_extent = info.f_tile * info.num_f_tiles
    slices: list[str] = []
    for d in out_dims:
        if d == p_axis:
            slices.append(_axis_slice(ir, d, p_extent))
        elif d == f_axis:
            slices.append(_axis_slice(ir, d, f_extent))
        else:
            slices.append(f"0:{ir.dimensions[d].dim_size}")
    return f"[{', '.join(slices)}]"


def _axis_slice(ir: KernelIR, dim: str, extent: int) -> str:
    """Slice along one axis — block-indexed if the dim is a block loop,
    else full extent."""
    full = ir.dimensions[dim].dim_size
    if extent >= full:
        return f"0:{full}"
    if dim in ir.dim_order:
        return f"i_block_{dim} * {extent} : i_block_{dim} * {extent} + {extent}"
    return f"0:{full}"


# ─────────────────────────────────────────────────────────────────────────────
# Tensor ↔ SBUF alias resolution
# ─────────────────────────────────────────────────────────────────────────────


def _sbuf_of(ir: KernelIR, tensor_name: str) -> str:
    """Return the physical SBUF buffer name for a logical tensor.

    Convention: every logical tensor has a matching ``sbuf_<name>``
    entry in ``physical_buffers``. Kernel params map to ``<p>_sbuf``
    if DMA-inserted; return tensor maps to ``<r>_hbm``.
    """
    if tensor_name in ir.physical_buffers:
        return tensor_name
    """Try sbuf_<name>."""
    alias = f"sbuf_{tensor_name}"
    if alias in ir.physical_buffers:
        return alias
    """Fall back: the name itself (kernel param name, etc.)."""
    return tensor_name
