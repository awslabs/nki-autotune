"""``render_ir``: lower :class:`KernelIR` to NKI source code.

Follows the per-operator walkthrough in
``examples/matmul_lhsT_rhs.md``. The kernel shape is:

1. Header — ``@nki.jit`` function signature, parameter asserts,
   the HBM output ``nl.ndarray``.
2. Kernel-top allocations for every buffer with ``emission_depth == 0``.
3. Nested ``for i_block_<d> in range(num_blocks[d])`` in ``dim_order``.
   At each depth we emit (a) allocations for buffers whose
   ``emission_depth == current_depth``, and (b) the accumulator
   ``memset`` at the innermost non-ACCUMULATION depth.
4. Per-op body: ``NKILoad`` → ``load_block(...)``, ``NKIMatmul`` →
   ``matmul_block(...)``, ``NKIStore`` → ``store_block(...)`` after
   every ACCUMULATION loop closes.
"""

from dataclasses import dataclass

from nkigym.kernel_ir.ir import BufferScope, KernelIR, NumBuffers, Op, PhysicalBuffer
from nkigym.kernel_ir.types import DimRole

_GADGETS_IMPORT = (
    "from nkigym.codegen.gadgets import (\n"
    "    allocate_buffers,\n"
    "    dma_transpose_block,\n"
    "    load_block,\n"
    "    matmul_block,\n"
    "    memset_buffers,\n"
    "    store_block,\n"
    "    transpose_block,\n"
    ")"
)


def render_ir(ir: KernelIR) -> str:
    """Lower ``ir`` to NKI source code using the gadgets module."""
    w = _Writer()
    w.line("import nki")
    w.line("import nki.language as nl")
    w.line()
    w.line(_GADGETS_IMPORT)
    w.line()
    w.line()
    w.line("@nki.jit")
    params = ", ".join(ir.param_names)
    w.line(f"def {ir.func_name}({params}):")
    w.indent()
    _emit_header(w, ir)
    _emit_body(w, ir)
    w.line(f"return {_store_op(ir).outputs[0]}")
    w.dedent()
    return w.getvalue()


class _Writer:
    """Tiny line-based writer with indentation."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._depth = 0

    def indent(self) -> None:
        self._depth += 1

    def dedent(self) -> None:
        self._depth -= 1

    def line(self, text: str = "") -> None:
        self._lines.append(("    " * self._depth + text) if text else "")

    def getvalue(self) -> str:
        return "\n".join(self._lines) + "\n"


"""────────────────────────────────────────────────────────────────
Header
────────────────────────────────────────────────────────────────"""


def _emit_header(w: _Writer, ir: KernelIR) -> None:
    """Parameter asserts + HBM output ``nl.ndarray``."""
    for p in ir.param_names:
        shape = ir.logical_tensors[p].shape
        w.line(f"assert {p}.shape == {tuple(shape)}")
    store_op = _store_op(ir)
    hbm_name = store_op.outputs[0]
    hbm = ir.physical_buffers[hbm_name]
    shape = tuple(ir.dimensions[d].dim_size for d in hbm.dim_ids)
    w.line(f"{hbm_name} = nl.ndarray({shape}, dtype=nl.{hbm.dtype}, buffer=nl.shared_hbm)")
    w.line()


def _store_op(ir: KernelIR) -> Op:
    """The single ``NKIStore`` at the tail of the ops list."""
    for op in ir.ops:
        if op.kind == "NKIStore":
            return op
    raise ValueError("KernelIR has no NKIStore op")


"""────────────────────────────────────────────────────────────────
Body: loop nest + per-op emission
────────────────────────────────────────────────────────────────"""


@dataclass
class _BufAlloc:
    """Resolved codegen view of one physical buffer."""

    name: str
    buf: PhysicalBuffer
    num_buffers: NumBuffers
    emission_depth: int
    p_tile: int
    num_p_tiles: int
    f_tile: int
    num_f_tiles: int


def _emit_body(w: _Writer, ir: KernelIR) -> None:
    """Kernel-top allocs → nested loop skeleton → per-op emission.

    The loop skeleton matches ``dim_order`` verbatim. Buffers fire
    their allocation at their declared ``emission_depth``. Compute /
    store ops attach to the loop whose open set matches their
    required dims.
    """
    allocs = _resolve_allocations(ir)
    matmul_op = _find_matmul_op(ir)
    store_op = _store_op(ir)

    _emit_allocs_at_depth(w, allocs, depth=0)
    if allocs and any(info.emission_depth == 0 for info in allocs.values()):
        w.line()
    _emit_loads_at_depth(w, ir, allocs, depth=0)
    _maybe_emit_accumulator_prologue(w, ir, allocs, matmul_op, depth=0)

    _emit_loop_nest(w, ir, allocs, matmul_op, store_op, dim_idx=0)


def _emit_loop_nest(
    w: _Writer, ir: KernelIR, allocs: dict[str, _BufAlloc], matmul_op: Op, store_op: Op, dim_idx: int
) -> None:
    """Recursively emit ``for i_block_<d> in range(...)`` in ``dim_order``.

    At each depth:
    * Emit allocations whose ``emission_depth == dim_idx``.
    * Bind + memset the accumulator slot at the accumulator's prologue depth.
    * Emit loads at their required depth.
    * Recurse.
    * Fire the store at the depth immediately outside all ACCUMULATION
      loops (``= _store_emission_depth``). Output dims that live outside
      the ACC loops contribute per-block slices; those nested inside
      (and thus already closed) contribute full-extent slices.
    """
    if dim_idx == len(ir.dim_order):
        _emit_compute_ops(w, ir, allocs)
        return

    dim = ir.dim_order[dim_idx]
    w.line(f"for i_block_{dim} in range({ir.num_blocks(dim)}):")
    w.indent()

    _emit_allocs_at_depth(w, allocs, depth=dim_idx + 1)
    _maybe_emit_accumulator_prologue(w, ir, allocs, matmul_op, dim_idx + 1)
    _emit_loads_at_depth(w, ir, allocs, dim_idx + 1)
    _emit_loop_nest(w, ir, allocs, matmul_op, store_op, dim_idx + 1)

    w.dedent()

    """Store placement: fire when we close the loop immediately outside
    the first ACC dim (i.e. right after the outermost ACC loop finishes
    and we're still inside any enclosing non-ACC loops whose indices the
    store slice needs).
    """
    store_depth = _store_emission_depth(ir)
    if dim_idx == store_depth:
        open_loops = ir.dim_order[:store_depth]
        _emit_store(w, ir, allocs, store_op, matmul_op, open_loops)


def _store_emission_depth(ir: KernelIR) -> int:
    """Depth at which the store fires.

    The store must run outside every ACCUMULATION loop (so the
    accumulator is complete) and inside every non-ACC loop that's
    outside any ACC loop (so the store slice can reference those
    block indices). That's exactly the position of the first ACC dim
    in ``dim_order`` — the store fires just as we dedent back out of
    that loop.
    """
    for i, d in enumerate(ir.dim_order):
        if ir.dimensions[d].role is DimRole.ACCUMULATION:
            return i
    return len(ir.dim_order)


def _find_matmul_op(ir: KernelIR) -> Op:
    """The single ``NKIMatmul`` op."""
    for op in ir.ops:
        if op.kind == "NKIMatmul":
            return op
    raise ValueError("KernelIR has no NKIMatmul op")


_TRANSPOSE_GADGETS: dict[str, str] = {"NKITranspose": "transpose_block", "NKIDMATranspose": "dma_transpose_block"}


def _emit_compute_ops(w: _Writer, ir: KernelIR, allocs: dict[str, _BufAlloc]) -> None:
    """Emit every non-Load/non-Store op at the innermost loop body.

    Walks ``ir.ops`` in list order (producer before consumer thanks to
    topological construction in ``build_ir``). Dispatches by kind.
    Fused HBM-sourced ``NKIDMATranspose`` ops are emitted by
    ``_emit_loads_at_depth`` instead — skipped here.
    """
    for op in ir.ops:
        if op.kind in _TRANSPOSE_GADGETS:
            if op.kind == "NKIDMATranspose" and op.inputs["data"] in ir.param_names:
                continue
            _emit_transpose(w, ir, allocs, op, _TRANSPOSE_GADGETS[op.kind])
        elif op.kind == "NKIMatmul":
            _emit_matmul(w, ir, allocs, op)


def _emit_transpose(w: _Writer, ir: KernelIR, allocs: dict[str, _BufAlloc], op: Op, gadget: str) -> None:
    """Emit ``cur_<dst> = <dst>[rotation]; <gadget>(cur_dst, cur_src)``.

    Both operands live at the tightest enclosing loop (same rotation
    discipline as loads). The source's ``cur_<name>`` was bound earlier
    when its load fired. ``gadget`` is the gadget name to call —
    ``transpose_block`` for TE-engine, ``dma_transpose_block`` for DMA.
    """
    src = op.inputs["data"]
    dst = op.outputs[0]
    dst_info = allocs[dst]
    cur_dst = _cur_name(dst)
    w.line(f"{cur_dst} = {dst}{_rotation_index(dst_info)}")
    w.line(f"{gadget}({cur_dst}, {_cur_name(src)})")


"""────────────────────────────────────────────────────────────────
Buffer sizing
────────────────────────────────────────────────────────────────"""


def _resolve_allocations(ir: KernelIR) -> dict[str, _BufAlloc]:
    """Compute per-buffer ``(p_tile, num_p_tiles, f_tile, num_f_tiles, emission_depth)``.

    * Load-destination / logical-tensor buffers use ``buffer_scopes`` to
      pick the axis extents.
    * The matmul accumulator (matmul output sbuf) is derived: each output
      axis spans full-dim extent if the axis is *below* the ACC dim in
      ``dim_order``, one-block otherwise.
    * The HBM destination is sized to the full tensor shape and is
      declared via the header, never allocated with ``allocate_buffers``.
    """
    matmul_op = _find_matmul_op(ir)
    acc_out_names = set(matmul_op.outputs)

    result: dict[str, _BufAlloc] = {}
    for name, buf in ir.physical_buffers.items():
        if name.startswith("hbm_"):
            continue
        if name in ir.buffer_scopes:
            p_tile, num_p, f_tile, num_f = _scope_extents(ir, buf, ir.buffer_scopes[name])
        elif name in acc_out_names:
            p_tile, num_p, f_tile, num_f = _accumulator_extents(ir, buf, matmul_op)
        else:
            p_tile, num_p, f_tile, num_f = _scope_extents(ir, buf, BufferScope.INNER)
        num_buffers = ir.num_buffers.get(name, NumBuffers())
        emission_depth = ir.emission_depth.get(name, 0)
        result[name] = _BufAlloc(
            name=name,
            buf=buf,
            num_buffers=num_buffers,
            emission_depth=emission_depth,
            p_tile=p_tile,
            num_p_tiles=num_p,
            f_tile=f_tile,
            num_f_tiles=num_f,
        )
    return result


def _accumulator_extents(ir: KernelIR, buf: PhysicalBuffer, matmul_op: Op) -> tuple[int, int, int, int]:
    """Per-axis extents for a matmul accumulator buffer.

    Each output axis is compared against the matmul's ACC dim position
    in ``dim_order``:

    * Axis is *below* ACC (iterates inside the reduction) → full-dim tiles.
    * Axis is *above* ACC (iterates outside the reduction) → one-block tiles.
    """
    p_tile, num_p = _acc_axis_extent(ir, buf.p_axis, matmul_op)
    if buf.f_axis is None:
        return p_tile, num_p, 1, 1
    f_tile, num_f = _acc_axis_extent(ir, buf.f_axis, matmul_op)
    return p_tile, num_p, f_tile, num_f


def _acc_axis_extent(ir: KernelIR, axis: str, matmul_op: Op) -> tuple[int, int]:
    """``(tile_size, num_tiles)`` for one accumulator axis."""
    info = ir.dimensions[axis]
    ptile = info.physical_tile_size
    full_tiles = info.dim_size // ptile
    block_tiles = ir.ltiles_per_block[axis]
    acc_dims = {d for d in matmul_op.blocking_dims if ir.dimensions[d].role is DimRole.ACCUMULATION}
    if not acc_dims or axis not in ir.dim_order:
        return ptile, block_tiles
    axis_pos = ir.dim_order.index(axis)
    first_acc_pos = min(ir.dim_order.index(d) for d in acc_dims if d in ir.dim_order)
    if axis_pos > first_acc_pos:
        return ptile, full_tiles
    return ptile, block_tiles


def _scope_extents(ir: KernelIR, buf: PhysicalBuffer, scope: BufferScope) -> tuple[int, int, int, int]:
    """Map a ``buffer_scope`` to ``(p_tile, num_p_tiles, f_tile, num_f_tiles)``.

    ``INNER`` — both axes tile-sized (one block per dim).
    ``MIDDLE`` — outermost-in-``dim_order`` axis per-block, the other full.
    ``OUTER`` — both axes full.
    """
    p_info = ir.dimensions[buf.p_axis]
    p_ltiles = ir.ltiles_per_block[buf.p_axis]
    p_full_tiles = p_info.dim_size // p_info.physical_tile_size
    if buf.f_axis is None:
        num_p = {BufferScope.INNER: p_ltiles, BufferScope.OUTER: p_full_tiles, BufferScope.MIDDLE: p_ltiles}[scope]
        return p_info.physical_tile_size, num_p, 1, 1

    f_info = ir.dimensions[buf.f_axis]
    f_ltiles = ir.ltiles_per_block[buf.f_axis]
    f_full_tiles = f_info.dim_size // f_info.physical_tile_size

    outer_axis = _outer_axis_in_order(ir, buf)
    p_is_outer = outer_axis == buf.p_axis

    def _num(axis_is_outer: bool, ltiles: int, full_tiles: int) -> int:
        if scope is BufferScope.INNER:
            return ltiles
        if scope is BufferScope.OUTER:
            return full_tiles
        return ltiles if axis_is_outer else full_tiles

    return (
        p_info.physical_tile_size,
        _num(p_is_outer, p_ltiles, p_full_tiles),
        f_info.physical_tile_size,
        _num(not p_is_outer, f_ltiles, f_full_tiles),
    )


def _outer_axis_in_order(ir: KernelIR, buf: PhysicalBuffer) -> str:
    """The outermost of this buffer's axes in ``dim_order``."""
    buf_axes = [buf.p_axis, buf.f_axis] if buf.f_axis else [buf.p_axis]
    for d in ir.dim_order:
        if d in buf_axes:
            return d
    raise ValueError(f"Buffer axes {buf_axes} not in dim_order {ir.dim_order}")


"""────────────────────────────────────────────────────────────────
Allocation emission
────────────────────────────────────────────────────────────────"""


def _emit_allocs_at_depth(w: _Writer, allocs: dict[str, _BufAlloc], depth: int) -> None:
    """Emit ``<name> = allocate_buffers(...)`` for every buffer whose
    ``emission_depth == depth``."""
    for info in allocs.values():
        if info.emission_depth == depth:
            w.line(_alloc_call(info))


def _alloc_call(info: _BufAlloc) -> str:
    """Render one ``allocate_buffers(...)`` call."""
    nb = info.num_buffers
    p = "None" if nb.num_p_buffers is None else str(nb.num_p_buffers)
    f = "None" if nb.num_f_buffers is None else str(nb.num_f_buffers)
    return (
        f"{info.name} = allocate_buffers(p_tile_size={info.p_tile}, num_p_tiles={info.num_p_tiles}, "
        f"f_tile_size={info.f_tile}, num_f_tiles={info.num_f_tiles}, "
        f"loc=nl.sbuf, dtype=nl.{info.buf.dtype}, "
        f"num_p_buffers={p}, num_f_buffers={f})"
    )


"""────────────────────────────────────────────────────────────────
Accumulator prologue
────────────────────────────────────────────────────────────────"""


def _maybe_emit_accumulator_prologue(
    w: _Writer, ir: KernelIR, allocs: dict[str, _BufAlloc], matmul_op: Op, depth: int
) -> None:
    """Emit ``cur_<acc> = <acc>[...]; memset_buffers(cur_<acc>, 0.0)``
    at the depth where the accumulator's rotation slot first resolves.

    Fires just inside the deepest non-ACC loop whose index appears in
    the accumulator's rotation (i.e. the slot is stable from here down
    through every ACC iteration). If the accumulator has no non-ACC
    rotation and no ACC-rotation either, it fires at depth 0.
    """
    out_name = matmul_op.outputs[0]
    info = allocs[out_name]
    if depth != _accumulator_prologue_depth(ir, matmul_op, info):
        return
    cur = _cur_name(out_name)
    w.line(f"{cur} = {out_name}{_rotation_index(info)}")
    w.line(f"memset_buffers({cur}, 0.0)")


def _accumulator_prologue_depth(ir: KernelIR, matmul_op: Op, info: "_BufAlloc") -> int:
    """Depth where the accumulator's ``cur_<name>`` slot binding fires.

    Equal to ``1 + max(dim_order.index(d))`` over non-ACC rotation axes
    of the accumulator. If the accumulator doesn't rotate on any
    non-ACC dim, falls back to the depth just above the first ACC loop.
    """
    rot_dims: list[str] = []
    nb = info.num_buffers
    if nb.num_p_buffers is not None:
        rot_dims.append(info.buf.p_axis)
    if nb.num_f_buffers is not None and info.buf.f_axis is not None:
        rot_dims.append(info.buf.f_axis)
    non_acc_rot = [d for d in rot_dims if ir.dimensions[d].role is not DimRole.ACCUMULATION]
    if non_acc_rot:
        return 1 + max(ir.dim_order.index(d) for d in non_acc_rot if d in ir.dim_order)
    for i, d in enumerate(ir.dim_order):
        if ir.dimensions[d].role is DimRole.ACCUMULATION:
            return i
    _ = matmul_op
    return 0


def _rotation_index(info: _BufAlloc) -> str:
    """``[i_block_<p> % N][i_block_<f> % M]`` — only active axes appear."""
    nb = info.num_buffers
    parts: list[str] = []
    if nb.num_p_buffers is not None:
        parts.append(f"[i_block_{info.buf.p_axis} % {nb.num_p_buffers}]")
    if nb.num_f_buffers is not None and info.buf.f_axis is not None:
        parts.append(f"[i_block_{info.buf.f_axis} % {nb.num_f_buffers}]")
    return "".join(parts)


def _cur_name(sbuf_name: str) -> str:
    """Per-iteration rotation slot binding name."""
    return f"cur_{sbuf_name}"


"""────────────────────────────────────────────────────────────────
Load emission
────────────────────────────────────────────────────────────────"""


def _emit_loads_at_depth(w: _Writer, ir: KernelIR, allocs: dict[str, _BufAlloc], depth: int) -> None:
    """Emit every HBM-sourced op whose destination buffer wants to appear here.

    That's ``NKILoad`` always, plus ``NKIDMATranspose`` ops whose
    ``data`` input is a kernel parameter (HBM tensor) — i.e. fused
    load-transposes. Each fires at the depth equal to ``1 +`` the
    deepest ``dim_order`` position of any of its destination buffer's
    rotating block axes.
    """
    for op in ir.ops:
        is_plain_load = op.kind == "NKILoad"
        is_fused_dma_transpose = op.kind == "NKIDMATranspose" and op.inputs.get("data") in ir.param_names
        if not (is_plain_load or is_fused_dma_transpose):
            continue
        dst_name = op.outputs[0]
        info = allocs[dst_name]
        if _load_emission_depth(ir, info) != depth:
            continue
        if is_plain_load:
            _emit_load(w, ir, op, info)
        else:
            _emit_hbm_dma_transpose(w, ir, op, info)


def _load_emission_depth(ir: KernelIR, info: _BufAlloc) -> int:
    """Depth at which the load fires.

    A load must sit inside every block loop whose index appears in the
    buffer's slice expression. For a buffer whose p/f axes are both
    block-indexed (``INNER`` scope), this is the depth of whichever of
    those axes lies deepest in ``dim_order``.
    """
    buf_axes = _block_indexed_axes(ir, info)
    if not buf_axes:
        return 0
    positions = [ir.dim_order.index(a) for a in buf_axes if a in ir.dim_order]
    if not positions:
        return 0
    return max(positions) + 1


def _block_indexed_axes(ir: KernelIR, info: _BufAlloc) -> list[str]:
    """Axes along which the buffer does NOT span the full dim.

    Those are the axes whose ``i_block_<d>`` appears in the buffer's
    HBM slice expression.
    """
    axes: list[str] = []
    for axis_id, tile_size, num_tiles in (
        (info.buf.p_axis, info.p_tile, info.num_p_tiles),
        (info.buf.f_axis, info.f_tile, info.num_f_tiles),
    ):
        if axis_id is None:
            continue
        if tile_size * num_tiles < ir.dimensions[axis_id].dim_size:
            axes.append(axis_id)
    return axes


def _emit_load(w: _Writer, ir: KernelIR, op: Op, info: _BufAlloc) -> None:
    """Emit ``cur_<dst> = <dst>[...]`` + ``load_block(cur_<dst>, src[...])``."""
    dst = op.outputs[0]
    src = op.inputs["data"]
    cur = _cur_name(dst)
    w.line(f"{cur} = {dst}{_rotation_index(info)}")
    w.line(f"load_block({cur}, {_hbm_slice_expr(ir, src, info)})")


def _emit_hbm_dma_transpose(w: _Writer, ir: KernelIR, op: Op, info: _BufAlloc) -> None:
    """Emit ``cur_<dst> = <dst>[...]; dma_transpose_block(cur_<dst>, src[...])``.

    Fires for ``NKIDMATranspose`` ops whose ``data`` input is an HBM
    parameter (installed by the ``LoadTranspose`` rewrite). The HBM
    slice is ordered ``(f_axis_of_dst, p_axis_of_dst)`` — i.e. swapped
    vs the src tensor's declared dims — to match the DMA-transpose
    shape contract.
    """
    dst = op.outputs[0]
    src = op.inputs["data"]
    cur = _cur_name(dst)
    w.line(f"{cur} = {dst}{_rotation_index(info)}")
    w.line(f"dma_transpose_block({cur}, {_hbm_slice_expr(ir, src, info, transpose=True)})")


def _hbm_slice_expr(ir: KernelIR, src: str, info: _BufAlloc, transpose: bool = False) -> str:
    """Build ``src[<slices>]`` matching the receiving gadget's shape contract.

    For plain loads, slices follow the src tensor's own dim order.
    For ``transpose=True`` (DMA transpose from HBM), the destination
    sbuf's ``(p_axis, f_axis)`` are swapped relative to ``src`` — the
    slice must be ordered ``(f_axis_of_dst, p_axis_of_dst)`` to yield
    a ``(f_tile*num_f_tiles, num_p_tiles*p_tile)`` region.
    """
    src_dims = ir.logical_tensors[src].dim_ids
    if transpose:
        axis_order = [info.buf.f_axis, info.buf.p_axis]
    else:
        axis_order = list(src_dims)
    slices = [_axis_slice(ir, d, info) for d in axis_order if d is not None]
    return f"{src}[{', '.join(slices)}]"


def _axis_slice(ir: KernelIR, axis: str, info: _BufAlloc) -> str:
    """Slice string for one axis of the load's HBM source."""
    full = ir.dimensions[axis].dim_size
    if axis == info.buf.p_axis:
        extent = info.p_tile * info.num_p_tiles
    elif axis == info.buf.f_axis:
        extent = info.f_tile * info.num_f_tiles
    else:
        extent = full
    if extent >= full or axis not in ir.dim_order:
        return f"0:{full}"
    return f"i_block_{axis} * {extent} : i_block_{axis} * {extent} + {extent}"


"""────────────────────────────────────────────────────────────────
Matmul emission
────────────────────────────────────────────────────────────────"""


def _emit_matmul(w: _Writer, ir: KernelIR, allocs: dict[str, _BufAlloc], op: Op) -> None:
    """Emit ``matmul_block(out_slice, lhs_T, rhs)`` at the innermost loop body.

    The output accumulator is sliced by the current M-block
    (``ltiles_per_block[M]`` tiles wide) when the accumulator holds more
    than one block along M. Inputs pass through their ``cur_<name>`` slot.
    """
    k_dim = op.axis_map["K"]
    m_dim = op.axis_map["M"]
    n_dim = op.axis_map["N"]
    lhs = op.inputs["stationary"]
    rhs = op.inputs["moving"]
    out = op.outputs[0]
    out_info = allocs[out]

    lt_m = ir.ltiles_per_block[m_dim]
    out_base = _cur_name(out)
    if out_info.buf.p_axis == m_dim and out_info.num_p_tiles > lt_m:
        out_expr = f"{out_base}[i_block_{m_dim} * {lt_m} : i_block_{m_dim} * {lt_m} + {lt_m}]"
    else:
        out_expr = out_base

    lhs_expr = _matmul_input_expr(ir, allocs, lhs, k_dim, m_dim)
    rhs_expr = _matmul_input_expr(ir, allocs, rhs, k_dim, n_dim)
    w.line(f"matmul_block({out_expr}, {lhs_expr}, {rhs_expr})")


def _matmul_input_expr(ir: KernelIR, allocs: dict[str, _BufAlloc], name: str, k_dim: str, other_dim: str) -> str:
    """Matmul operand expression.

    Input buffers may hold more than one block along some axis — if so,
    slice to the active block so the gadget sees exactly the tiles it
    needs:

    * ``k_dim`` on the P-axis → slice the P-slot list by K-block.
    * ``k_dim`` on the packed free axis → slice each leaf's free axis
      by K-block.
    * ``other_dim`` (M for stationary, N for moving) on the free axis →
      slice each leaf's free axis by that block.
    * ``other_dim`` on the P-axis (when K is on free) → slice P-slots
      by that block.
    """
    info = allocs[name]
    base = (
        _cur_name(name)
        if info.num_buffers.num_p_buffers is not None or info.num_buffers.num_f_buffers is not None
        else name
    )
    buf = info.buf
    lt_k = ir.ltiles_per_block[k_dim]
    lt_other = ir.ltiles_per_block[other_dim]

    """K-axis slicing."""
    expr = base
    if buf.p_axis == k_dim and info.num_p_tiles > lt_k:
        expr = f"{expr}[i_block_{k_dim} * {lt_k} : i_block_{k_dim} * {lt_k} + {lt_k}]"
    elif buf.f_axis == k_dim and info.num_f_tiles > lt_k:
        f_tile = info.f_tile
        expr = (
            f"[leaf[:, i_block_{k_dim} * {lt_k * f_tile} : "
            f"i_block_{k_dim} * {lt_k * f_tile} + {lt_k * f_tile}] for leaf in {expr}]"
        )

    """Output-dim (M for stationary, N for moving) slicing."""
    if buf.f_axis == other_dim and info.num_f_tiles > lt_other:
        f_tile = info.f_tile
        expr = (
            f"[leaf[:, i_block_{other_dim} * {lt_other * f_tile} : "
            f"i_block_{other_dim} * {lt_other * f_tile} + {lt_other * f_tile}] for leaf in {expr}]"
        )
    elif buf.p_axis == other_dim and info.num_p_tiles > lt_other and buf.p_axis != k_dim:
        expr = f"{expr}[i_block_{other_dim} * {lt_other} : i_block_{other_dim} * {lt_other} + {lt_other}]"
    return expr


"""────────────────────────────────────────────────────────────────
Store emission
────────────────────────────────────────────────────────────────"""


def _emit_store(
    w: _Writer, ir: KernelIR, allocs: dict[str, _BufAlloc], store_op: Op, matmul_op: Op, open_loops: list[str]
) -> None:
    """Emit ``store_block(hbm[<slice>], <sbuf_expr>)``.

    Store placement rule: fire after all SEQUENTIAL/ACCUMULATION loops
    of the producer's sbuf data are done. For matmul-output sbuf that
    means emitting the store when the innermost ACC loop closes.

    Only ``open_loops`` (the loops still in scope at the store point)
    contribute block-indexed slices. For HBM dims that are block-indexed
    we use ``block_extent(d)``; for everything else we emit the full span.

    Same indexing applies to the sbuf source: if the accumulator buffer
    is wider than one block along an open-loop dim, slice it per block.
    """
    sbuf_name = store_op.inputs["data"]
    hbm_name = store_op.outputs[0]
    info = allocs[sbuf_name]
    hbm = ir.physical_buffers[hbm_name]
    _ = matmul_op

    hbm_slices: list[str] = []
    for d in hbm.dim_ids:
        full = ir.dimensions[d].dim_size
        if d in open_loops:
            extent = ir.block_extent(d)
            hbm_slices.append(f"i_block_{d} * {extent} : i_block_{d} * {extent} + {extent}")
        else:
            hbm_slices.append(f"0:{full}")

    sbuf_expr = _cur_name(sbuf_name)
    p_axis = info.buf.p_axis
    f_axis = info.buf.f_axis
    if p_axis in open_loops and info.num_p_tiles > ir.ltiles_per_block[p_axis]:
        lt = ir.ltiles_per_block[p_axis]
        sbuf_expr = f"{sbuf_expr}[i_block_{p_axis} * {lt} : i_block_{p_axis} * {lt} + {lt}]"
    elif f_axis is not None and f_axis in open_loops and info.num_f_tiles > ir.ltiles_per_block[f_axis]:
        """Free-axis per-block slicing — rare; included for completeness."""
        lt = ir.ltiles_per_block[f_axis]
        sbuf_expr = f"{sbuf_expr}[:, i_block_{f_axis} * {lt} : i_block_{f_axis} * {lt} + {lt}]"

    w.line(f"store_block({hbm_name}[{', '.join(hbm_slices)}], {sbuf_expr})")
