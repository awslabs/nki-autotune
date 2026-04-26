"""``render_ir``: mechanical lowering of :class:`KernelIR` to NKI source.

Follows the codegen contract in ``examples/matmul_lhsT_rhs.md``:

1. Every loop in ``loop_order`` is emitted verbatim (``for
   i_block_<d>``/``for i_tile_<d>`` in order).
2. Every non-HBM physical buffer becomes a single ``nl.ndarray(P_tile,
   num_p_tiles, num_f_tiles * f_tile)`` allocation, placed at the
   tightest depth that still covers every access (derived from the
   buffer's per-dim ``buffer_scopes`` entry + the reducing-dim
   accumulator rule).
3. Every op fires at ``max(operand-availability, op-intrinsic tile
   requirement)``. For matmul-style reducers, the PSUM →  SBUF drain
   fires at the PSUM's emission depth (right after the reducing-dim
   tile loop closes).
4. Store ops emit the main-nest position outside every reducing block
   loop of their sbuf producer, plus a local ``{p_axis}.block`` /
   ``{p_axis}.tile`` sub-nest inside so the DMA's P-axis ``.tile`` is
   open.

Illegal IR raises — no silent rewriting.
"""

from dataclasses import dataclass

from nkigym.kernel_ir.ir import DimScope, KernelIR, Op, PhysicalBuffer

_MATMUL_KINDS = frozenset({"NKIMatmul"})
_LOAD_KINDS = frozenset({"NKILoad"})
_STORE_KINDS = frozenset({"NKIStore"})


def render_ir(ir: KernelIR) -> str:
    """Mechanically lower ``ir`` to NKI source code."""
    w = _Writer()
    w.line("import nki")
    w.line("import nki.isa as nisa")
    w.line("import nki.language as nl")
    w.line()
    w.line()
    w.line("@nki.jit")
    params = ", ".join(ir.param_names)
    w.line(f"def {ir.func_name}({params}):")
    w.indent()
    _emit_header(w, ir)

    plan = _build_plan(ir)
    _emit_nest(w, ir, plan, depth=0, open_loops=[])

    hbm_name = _store_op(ir).outputs[0]
    w.line(f"return {hbm_name}")
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
        if op.kind in _STORE_KINDS:
            return op
    raise ValueError("KernelIR has no NKIStore op")


"""────────────────────────────────────────────────────────────────
Plan — derive emission depths / fire depths / drain depths up front
────────────────────────────────────────────────────────────────"""


@dataclass
class _BufferInfo:
    """Resolved codegen view of one physical buffer."""

    name: str
    buf: PhysicalBuffer
    emission_depth: int
    """Depth at which to emit ``<name> = nl.ndarray(...)``.
    ``0`` = kernel-top (before the outermost loop)."""
    prologue_depth: int | None
    """Depth at which to emit ``nisa.memset(<name>[...], identity)``.
    ``None`` = no memset (not an accumulator)."""
    shape: tuple[int, int, int]
    """``(p_tile, num_p_tiles, num_f_tiles * f_tile)``."""
    per_axis_num_tiles: dict[str, int]
    """Per-axis tile count (``num_p_tiles`` for p_axis, ``num_f_tiles``
    for f_axis). Reducing-dim axes not stored here (they don't
    contribute to storage shape)."""


@dataclass
class _Plan:
    """Per-kernel derived placement metadata."""

    buffers: dict[str, _BufferInfo]
    """Non-HBM buffers with allocation metadata."""
    op_fire_depth: dict[int, int]
    """Keyed by ``id(op)``. Depth in the main nest at which the op fires."""
    matmul_drain_depth: dict[int, int]
    """Keyed by ``id(op)`` for matmul ops — PSUM → SBUF drain depth."""


def _build_plan(ir: KernelIR) -> _Plan:
    """Compute all derived placement info for the IR.

    Assumes ``ir`` has already passed
    :func:`nkigym.kernel_ir.validate.is_valid`. Will raise
    ``ValueError`` on structural contradictions (missing scopes,
    infeasible emission depth), but doesn't re-check the
    semantic invariants the validator enforces (accumulator
    coverage, SBUF-outlives-drain).
    """
    buffers: dict[str, _BufferInfo] = {}
    for name, buf in ir.physical_buffers.items():
        if buf.loc == "hbm":
            continue
        buffers[name] = _build_buffer_info(ir, name, buf)

    op_fire_depth: dict[int, int] = {}
    drain_depth: dict[int, int] = {}
    for op in ir.ops:
        op_fire_depth[id(op)] = _compute_op_fire_depth(ir, op, buffers)
        if op.kind in _MATMUL_KINDS and op.outputs:
            psum_name = _psum_name_of(op)
            if psum_name in buffers:
                drain_depth[id(op)] = buffers[psum_name].emission_depth
    return _Plan(buffers=buffers, op_fire_depth=op_fire_depth, matmul_drain_depth=drain_depth)


def _is_full_k_drain(ir: KernelIR, op: Op) -> bool:
    """True iff every reducing dim has PSUM scope ``FULL`` (Option A).

    Selects the drain op (``tensor_copy`` vs ``tensor_tensor``).
    Drain *depth* is always ``psum.emission_depth`` regardless of
    variant — emission depth already encodes "immediately enclosing
    the outermost loop re-entering PSUM's lifetime", which is
    exactly the depth at which the drain needs to fire:

    * Option A (K=FULL): emission ≤ ``pos(K.block)-1`` → drain fires
      on K.block close.
    * Option B (K=PER_BLOCK): emission ≥ ``pos(K.block)`` and ≤
      ``pos(K.tile)-1`` → drain fires on K.tile close.
    """
    psum_name = _psum_name_of(op)
    scope_map = ir.buffer_scopes.get(psum_name, {})
    for d in op.blocking_dims:
        if scope_map.get(d) is not DimScope.FULL:
            return False
    return bool(op.blocking_dims)


def _psum_name_of(op: Op) -> str:
    """``psum_<stem>`` name for a matmul op's hoisted PSUM buffer."""
    out = op.outputs[0]
    if not out.startswith("sbuf_"):
        raise ValueError(f"matmul op {op.kind} expected sbuf_-prefixed output, got {out!r}")
    return "psum_" + out[len("sbuf_") :]


def _reducing_dims_of_buffer(ir: KernelIR, name: str) -> set[str]:
    """Return the reducing dims for ``name`` if it's a matmul accumulator.

    For matmul SBUF outputs + PSUM siblings: reducing dims come from
    the producing matmul op's ``blocking_dims``. Other buffers have no
    reducing dims (empty set).
    """
    for op in ir.ops:
        if op.kind not in _MATMUL_KINDS:
            continue
        if name == op.outputs[0]:
            return set(op.blocking_dims)
        if name == _psum_name_of(op):
            return set(op.blocking_dims)
    return set()


def _build_buffer_info(ir: KernelIR, name: str, buf: PhysicalBuffer) -> _BufferInfo:
    """Compute emission depth, prologue depth, and 3D shape for ``buf``.

    ``buffer_scopes[name]`` may list dims beyond ``buf.dim_ids`` —
    specifically, matmul PSUM buffers carry a reducing-dim entry even
    though the reducing axis does not contribute to the storage shape.
    Extra dims only affect emission depth (they clamp it).
    """
    if name not in ir.buffer_scopes:
        raise ValueError(
            f"buffer {name!r} has no buffer_scopes entry — the renderer mechanically "
            f"lowers IR knobs and does not invent defaults"
        )
    scope_map = ir.buffer_scopes[name]
    reducing = _reducing_dims_of_buffer(ir, name)

    """For matmul SBUF outputs, the reducing dim is implicitly FULL
    (codegen rule — the accumulator must survive every K.block).
    Fill it in so emission-depth logic sees every buffer dim."""
    if buf.loc == "sbuf" and reducing:
        scope_map = dict(scope_map)
        for d in reducing:
            if d in buf.dim_ids:
                scope_map.setdefault(d, DimScope.FULL)

    emission_depth = _buffer_emission_depth(ir, name, buf, scope_map, reducing)

    p_axis = buf.p_axis
    f_axis = buf.f_axis
    p_tile, f_tile = buf.tile

    num_p_tiles = _num_tiles_for_scope(ir, p_axis, scope_map[p_axis]) if p_axis in scope_map else 1
    if f_axis is not None and f_axis in scope_map:
        num_f_tiles = _num_tiles_for_scope(ir, f_axis, scope_map[f_axis])
    else:
        num_f_tiles = 1
    shape = (p_tile, num_p_tiles, num_f_tiles * f_tile)

    per_axis = {p_axis: num_p_tiles}
    if f_axis is not None:
        per_axis[f_axis] = num_f_tiles

    prologue_depth: int | None = None
    if _needs_memset_prologue(buf, reducing):
        prologue_depth = emission_depth

    return _BufferInfo(
        name=name,
        buf=buf,
        emission_depth=emission_depth,
        prologue_depth=prologue_depth,
        shape=shape,
        per_axis_num_tiles=per_axis,
    )


def _needs_memset_prologue(buf: PhysicalBuffer, reducing: set[str]) -> bool:
    """Accumulators need a memset prologue.

    * PSUM buffers (matmul accumulators) — zero before K accumulation.
    * SBUF matmul outputs (post-PSUM drain sums across K.blocks) —
      zero before any block contributes.
    """
    if buf.loc == "psum":
        return True
    if buf.loc == "sbuf" and reducing:
        return True
    return False


def _num_tiles_for_scope(ir: KernelIR, axis: str, scope: DimScope) -> int:
    """Map a ``DimScope`` on ``axis`` to a tile count.

    * ``PER_TILE`` → 1.
    * ``PER_BLOCK`` → ``ltiles_per_block[axis]``.
    * ``FULL`` → ``num_ltile[axis]``.
    """
    if scope is DimScope.PER_TILE:
        return 1
    if scope is DimScope.PER_BLOCK:
        return ir.ltiles_per_block[axis]
    return ir.num_ltile(axis)


def _buffer_emission_depth(
    ir: KernelIR, name: str, buf: PhysicalBuffer, scope_map: dict[str, DimScope], reducing: set[str]
) -> int:
    """Mechanical per-dim emission depth.

    Per dim ``d`` in ``scope_map`` (which may include reducing axes
    that aren't in ``buf.dim_ids``):

    * ``FULL`` → depth ≤ ``pos({d}.block) - 1`` (must sit OUTSIDE the
      block loop so it spans every block).
    * ``PER_BLOCK`` → depth ≥ ``pos({d}.block)`` (inside block loop).
    * ``PER_TILE`` → depth ≥ ``pos({d}.tile)`` (inside tile loop).

    Plus: for PSUM accumulators, the reducing-dim's ``.tile`` loop is
    the HW accumulation loop; the PSUM must be allocated OUTSIDE that
    loop (upper bound ``pos({d_reducing}.tile) - 1``). Otherwise
    re-entry zero-init wipes the partial sum.

    Returns the computed depth; raises ``ValueError`` when upper bound
    < lower bound (illegal IR).
    """
    lower = 0
    upper = len(ir.loop_order)
    for d, scope in scope_map.items():
        block_pos = _loop_pos(ir, f"{d}.block")
        tile_pos = _loop_pos(ir, f"{d}.tile")
        if scope is DimScope.FULL:
            upper = min(upper, block_pos - 1)
        elif scope is DimScope.PER_BLOCK:
            lower = max(lower, block_pos)
        elif scope is DimScope.PER_TILE:
            lower = max(lower, tile_pos)
    if buf.loc == "psum":
        for d in reducing:
            upper = min(upper, _loop_pos(ir, f"{d}.tile") - 1)
    if upper < lower:
        raise ValueError(
            f"buffer {name!r}: no feasible emission depth (lower={lower}, upper={upper}). "
            f"scope_map={ {k: v.value for k, v in scope_map.items()} } "
            f"loop_order={ir.loop_order}"
        )
    return lower


def _loop_pos(ir: KernelIR, entry: str) -> int:
    """1-indexed position of ``entry`` in ``loop_order`` — matches
    the loop-nest depth at which ``entry``'s body runs."""
    return ir.loop_order.index(entry) + 1


def _compute_op_fire_depth(ir: KernelIR, op: Op, buffers: dict[str, _BufferInfo]) -> int:
    """Main-nest fire depth for ``op``.

    ``fire_depth = max(operand-availability, op-intrinsic tile floor)``.

    * Operand availability = ``max(buf.emission_depth for buf in operands)``.
      Kernel params (HBM inputs) are available at depth 0.
    * Op-intrinsic tile floor:
      - ``nc_matmul`` touches ``(K, M, N)`` — needs all three ``.tile``
        loops open → depth ≥ max of their positions.
      - ``dma_copy`` (load) — P-axis must be inside its ``.tile``
        loop. The free axis, if block-indexed in the sbuf destination
        (scope PER_BLOCK or FULL), also needs its ``.block`` (and
        ``.tile`` when FULL with ltile>1) loop open so each DMA lands
        in the correct sbuf slot.
      - ``dma_copy`` (store) — fires AFTER every reducing block loop
        of its sbuf producer has closed, i.e. at
        ``min(pos({d_reducing}.block)) - 1``.
    """
    operand_names = list(op.inputs.values()) + list(op.outputs)
    operand_depth = 0
    for name in operand_names:
        if name in buffers:
            operand_depth = max(operand_depth, buffers[name].emission_depth)

    tile_floor = 0
    if op.kind in _MATMUL_KINDS:
        for axis in ("K", "M", "N"):
            if axis in op.axis_map:
                tile_floor = max(tile_floor, _loop_pos(ir, f"{op.axis_map[axis]}.tile"))
    elif op.kind in _LOAD_KINDS:
        tile_floor = _load_tile_floor(ir, op, buffers)
    elif op.kind in _STORE_KINDS:
        """Store fires at the depth where the SBUF accumulator holds
        its complete reduction result.

        * Option A (PSUM K=FULL) — drain writes the final SBUF once
          at PSUM emission depth; store fires at that same depth.
        * Option B (PSUM K=PER_BLOCK) — drain runs per K.block iter
          and folds into SBUF; SBUF is complete after K.block closes.
          Store fires at ``pos(K.block) - 1``.

        Unified rule: ``min(drain_depth, pos(K.block) - 1)``.
        For Option A, drain is already at ``pos(K.block) - 1`` (the
        PSUM emission depth is clamped there by the ``K=FULL``
        constraint), so both yield the same depth. For Option B,
        drain is deeper (inside K.block), and the outer close wins.
        """
        producer_drain_depth = _matmul_producer_drain_depth(ir, op, buffers)
        reducing_block_positions = _store_reducing_block_positions(ir, op)
        outer_close = min(reducing_block_positions) - 1 if reducing_block_positions else 0
        if producer_drain_depth is not None:
            store_depth = min(producer_drain_depth, outer_close)
            return max(operand_depth, store_depth)
        return max(operand_depth, outer_close)
    return max(operand_depth, tile_floor)


def _matmul_producer_drain_depth(ir: KernelIR, store_op: Op, buffers: dict[str, _BufferInfo]) -> int | None:
    """Drain depth of the matmul that produces the store's sbuf input.

    The store can only read valid data after the drain has written to
    the sbuf. Returns the drain depth (= PSUM emission depth) or
    ``None`` when the producer is not a matmul.
    """
    sbuf_name = store_op.inputs["data"]
    for op in ir.ops:
        if op.kind in _MATMUL_KINDS and sbuf_name in op.outputs:
            psum_name = _psum_name_of(op)
            if psum_name in buffers:
                return buffers[psum_name].emission_depth
    return None


def _store_reducing_block_positions(ir: KernelIR, store_op: Op) -> list[int]:
    """``pos({d}.block)`` for every reducing dim of the store's sbuf producer.

    The store fires outside every reducing ``.block`` loop AND every
    reducing ``.tile`` loop. Since ``.block`` precedes ``.tile`` for
    each dim, the ``.block`` positions dominate the constraint.
    """
    sbuf_name = store_op.inputs["data"]
    producer: Op | None = None
    for op in ir.ops:
        if sbuf_name in op.outputs:
            producer = op
            break
    if producer is None:
        return []
    return [_loop_pos(ir, f"{d}.block") for d in producer.blocking_dims if f"{d}.block" in ir.loop_order]


def _load_partition_axis(ir: KernelIR, op: Op) -> str:
    """Partition axis (``P``) of a load op — the destination sbuf's
    first ``dim_ids`` entry."""
    dst = op.outputs[0]
    return ir.physical_buffers[dst].p_axis


def _load_tile_floor(ir: KernelIR, op: Op, buffers: dict[str, _BufferInfo]) -> int:
    """Minimum depth at which a load can fire.

    * P-axis: ``pos({p_axis}.tile)`` — one tile per DMA call.
    * F-axis: depending on scope the sbuf holds, the load needs the
      relevant loop open so its destination slot and source offset
      address the right tile:
        - FULL scope (multiple tiles): needs ``.block`` open. When
          ``ltile > 1``, needs ``.tile`` open too.
        - PER_BLOCK scope: if ``ltile > 1`` needs ``.tile`` open;
          otherwise no extra constraint (block loop is already open
          via operand-availability).
        - PER_TILE: no extra — emission depth already inside .tile.
    """
    dst = op.outputs[0]
    info = buffers[dst]
    buf = info.buf
    p_axis = buf.p_axis
    f_axis = buf.f_axis
    floor = _loop_pos(ir, f"{p_axis}.tile")
    if f_axis is not None:
        f_tiles = info.per_axis_num_tiles.get(f_axis, 1)
        ltile = ir.ltiles_per_block[f_axis]
        num_ltile = ir.num_ltile(f_axis)
        if f_tiles == num_ltile and f_tiles > 1:
            """FULL on F — need block open to pick the right block;
            when ltile>1 also need tile open."""
            floor = max(floor, _loop_pos(ir, f"{f_axis}.block"))
            if ltile > 1:
                floor = max(floor, _loop_pos(ir, f"{f_axis}.tile"))
        elif f_tiles == ltile and ltile > 1:
            """PER_BLOCK on F with ltile>1 — need tile open to
            pick which tile within the block."""
            floor = max(floor, _loop_pos(ir, f"{f_axis}.tile"))
    """P-axis analogue: if sbuf's P-axis is FULL or PER_BLOCK with
    ltile>1 we need extra block/tile open too."""
    p_tiles = info.per_axis_num_tiles.get(p_axis, 1)
    p_ltile = ir.ltiles_per_block[p_axis]
    p_num_ltile = ir.num_ltile(p_axis)
    if p_tiles == p_num_ltile and p_tiles > 1:
        floor = max(floor, _loop_pos(ir, f"{p_axis}.block"))
    return floor


"""────────────────────────────────────────────────────────────────
Nest emission
────────────────────────────────────────────────────────────────"""


def _emit_nest(w: _Writer, ir: KernelIR, plan: _Plan, depth: int, open_loops: list[str]) -> None:
    """Recursively emit loop_order entries at ``depth`` with body content.

    Each depth's body:

    1. PRE — allocations, memsets, compute ops whose fire_depth is
       ``depth`` AND fire before any deeper loop opens (loads, matmul).
    2. Open the next ``loop_order[depth]`` loop (if any) and recurse.
    3. POST — drains, stores. These fire AFTER the nested block closes
       (their drain/store depth equals the current depth).
    """
    _emit_allocs_at_depth(w, plan, depth)
    _emit_prologues_at_depth(w, plan, depth)
    _emit_pre_compute_ops(w, ir, plan, depth, open_loops)

    if depth < len(ir.loop_order):
        entry = ir.loop_order[depth]
        dim_id, kind = entry.split(".")
        trip = ir.num_blocks(dim_id) if kind == "block" else ir.ltiles_per_block[dim_id]
        var = f"i_{kind}_{dim_id}"
        w.line(f"for {var} in range({trip}):")
        w.indent()
        _emit_nest(w, ir, plan, depth + 1, open_loops + [entry])
        w.dedent()

    _emit_post_compute_ops(w, ir, plan, depth, open_loops)


def _emit_allocs_at_depth(w: _Writer, plan: _Plan, depth: int) -> None:
    """Emit every buffer whose ``emission_depth == depth``."""
    for info in plan.buffers.values():
        if info.emission_depth != depth:
            continue
        w.line(_alloc_call(info))


def _alloc_call(info: _BufferInfo) -> str:
    """Render one ``<name> = nl.ndarray(...)`` allocation."""
    buf = info.buf
    return f"{info.name} = nl.ndarray({info.shape}, dtype=nl.{buf.dtype}, buffer=nl.{buf.loc})"


def _emit_prologues_at_depth(w: _Writer, plan: _Plan, depth: int) -> None:
    """Emit ``nisa.memset(<name>[0:P,0:Np,0:Nf*f], 0.0)`` for every
    accumulator whose prologue fires here."""
    for info in plan.buffers.values():
        if info.prologue_depth != depth:
            continue
        p_tile, num_p, packed_f = info.shape
        w.line(f"nisa.memset({info.name}[0:{p_tile}, 0:{num_p}, 0:{packed_f}], value=0.0)")


def _emit_pre_compute_ops(w: _Writer, ir: KernelIR, plan: _Plan, depth: int, open_loops: list[str]) -> None:
    """Compute ops that fire in the PRE section at ``depth``.

    Loads and matmul bodies fire "pre" — before any deeper loop opens.
    Drains/stores are emitted by ``_emit_post_compute_ops``.
    """
    for op in ir.ops:
        if plan.op_fire_depth.get(id(op)) != depth:
            continue
        if op.kind in _LOAD_KINDS:
            _emit_load(w, ir, plan, op, open_loops)
        elif op.kind in _MATMUL_KINDS:
            _emit_matmul_call(w, ir, plan, op, open_loops)


def _emit_post_compute_ops(w: _Writer, ir: KernelIR, plan: _Plan, depth: int, open_loops: list[str]) -> None:
    """Drains + stores whose emission fires AFTER the deeper loop closes.

    * Matmul drain: fires at ``drain_depth == psum.emission_depth``.
    * Store: fires at ``operand_depth`` (in the main nest), with a
      local P-axis ``.block`` / ``.tile`` sub-nest inside to satisfy
      the DMA's P-axis tile floor.
    """
    for op in ir.ops:
        if op.kind in _MATMUL_KINDS:
            if plan.matmul_drain_depth.get(id(op)) == depth:
                _emit_matmul_drain(w, ir, plan, op, open_loops)
    for op in ir.ops:
        if op.kind not in _STORE_KINDS:
            continue
        if plan.op_fire_depth[id(op)] != depth:
            continue
        _emit_store(w, ir, plan, op, open_loops)


"""────────────────────────────────────────────────────────────────
Load emission
────────────────────────────────────────────────────────────────"""


def _emit_load(w: _Writer, ir: KernelIR, plan: _Plan, op: Op, open_loops: list[str]) -> None:
    """Emit ``nisa.dma_copy(dst=<sbuf>[...], src=<hbm>[...])``.

    The load's partition axis is sliced one tile per call (DMA
    contract). The free axis spans the largest contiguous region of
    the sbuf destination that isn't yet iterated by an open tile
    loop — a single dma_copy can span multiple free-axis tiles.
    """
    src_hbm = op.inputs["data"]
    dst = op.outputs[0]
    dst_info = plan.buffers[dst]
    dst_expr = _load_sbuf_slice(ir, dst_info, open_loops)
    src_expr = _load_hbm_slice(ir, src_hbm, dst_info, open_loops)
    w.line("nisa.dma_copy(")
    w.indent()
    w.line(f"dst={dst_expr},")
    w.line(f"src={src_expr},")
    w.dedent()
    w.line(")")


def _load_sbuf_slice(ir: KernelIR, info: _BufferInfo, open_loops: list[str]) -> str:
    """SBUF dst slice for a load call.

    Partition axis: one tile per call — slot from the current
    ``i_tile_<p>`` (required to be open).
    Free axis: spans every tile the sbuf holds that isn't indexed by
    a currently-open ``i_tile_<f>`` / ``i_block_<f>`` loop.
    """
    name = info.name
    buf = info.buf
    p_tile, f_tile = buf.tile
    p_axis = buf.p_axis
    f_axis = buf.f_axis

    p_expr = _axis_slot_expr(ir, p_axis, info.per_axis_num_tiles[p_axis], open_loops)
    if f_axis is None:
        return f"{name}[0:{p_tile}, {p_expr}, 0:{f_tile}]"
    f_range = _load_free_sbuf_range(ir, f_axis, info.per_axis_num_tiles.get(f_axis, 1), f_tile, open_loops)
    return f"{name}[0:{p_tile}, {p_expr}, {f_range}]"


def _load_free_sbuf_range(ir: KernelIR, axis: str, scope_tiles: int, f_tile: int, open_loops: list[str]) -> str:
    """F-axis range for the sbuf dst of a load.

    Width is one ``f_tile`` — the load writes exactly one tile on the
    free axis per call (covering multiple tiles is handled by looping
    over ``.tile`` / ``.block``). The slot offset is driven by the
    currently open block/tile loops.

    * ``scope_tiles == 1`` (PER_TILE or degenerate) → ``0:f_tile``.
    * ``scope_tiles == ltile`` (PER_BLOCK):
        - ``ltile == 1`` → ``0:f_tile``.
        - ``ltile > 1`` with ``.tile`` open → ``i_tile * f_tile : ...``.
    * ``scope_tiles == num_ltile`` (FULL):
        - slot index = ``i_block * ltile + i_tile``, using whichever
          loops are open. Range is one ``f_tile`` at that slot.
    """
    tile_open = f"{axis}.tile" in open_loops
    block_open = f"{axis}.block" in open_loops
    ltile = ir.ltiles_per_block[axis]
    if scope_tiles <= 1:
        return f"0:{f_tile}"
    if scope_tiles == ltile:
        if ltile == 1 or not tile_open:
            return f"0:{f_tile}"
        return f"i_tile_{axis} * {f_tile} : i_tile_{axis} * {f_tile} + {f_tile}"
    parts: list[str] = []
    if block_open:
        parts.append(f"i_block_{axis} * {ltile}")
    if tile_open:
        parts.append(f"i_tile_{axis}")
    if not parts:
        return f"0:{f_tile}"
    slot = " + ".join(parts)
    slot_paren = f"({slot})" if "+" in slot else slot
    return f"{slot_paren} * {f_tile} : {slot_paren} * {f_tile} + {f_tile}"


def _load_hbm_slice(ir: KernelIR, src: str, dst_info: _BufferInfo, open_loops: list[str]) -> str:
    """HBM src slice matching the SBUF dst extent for a load.

    Partition axis: one tile per call, indexed by the currently open
    ``i_block_<p>`` + ``i_tile_<p>``.
    Free axis: span matches the sbuf destination — multi-tile if the
    sbuf covers multiple tiles and their loops aren't open.
    Other axes: full extent.
    """
    src_dims = ir.logical_tensors[src].dim_ids
    slices: list[str] = []
    for d in src_dims:
        if d == dst_info.buf.p_axis:
            slices.append(_hbm_partition_axis_slice(ir, d, open_loops))
        elif d == dst_info.buf.f_axis:
            slices.append(_hbm_free_axis_slice(ir, d, dst_info.per_axis_num_tiles.get(d, 1), open_loops))
        else:
            full = ir.dimensions[d].dim_size
            slices.append(f"0:{full}")
    return f"{src}[{', '.join(slices)}]"


def _hbm_partition_axis_slice(ir: KernelIR, axis: str, open_loops: list[str]) -> str:
    """HBM slice for the partition axis — exactly one ``ptile``."""
    ptile = ir.dimensions[axis].physical_tile_size
    block_elems = ir.block_extent(axis)
    tile_open = f"{axis}.tile" in open_loops
    block_open = f"{axis}.block" in open_loops
    parts: list[str] = []
    if block_open:
        parts.append(f"i_block_{axis} * {block_elems}")
    if tile_open:
        parts.append(f"i_tile_{axis} * {ptile}")
    if not parts:
        return f"0:{ptile}"
    base = " + ".join(parts)
    return f"{base} : {base} + {ptile}"


def _hbm_free_axis_slice(ir: KernelIR, axis: str, scope_tiles: int, open_loops: list[str]) -> str:
    """HBM slice for the free axis — one ``ptile`` per DMA call.

    Matches the SBUF destination's per-call extent (always one
    physical tile). Offset uses the currently open ``.block`` /
    ``.tile`` loops:

    * ``scope_tiles == 1`` → whole axis covered by a single tile; if
      no loop is open, slice is ``0:ptile``; else ``i_block_ *
      block_extent + i_tile_ * ptile``.
    * ``scope_tiles == ltile`` (PER_BLOCK): same offset form —
      ``i_block_ * block_extent + i_tile_ * ptile``.
    * ``scope_tiles == num_ltile`` (FULL): same offset form.
    """
    ptile = ir.dimensions[axis].physical_tile_size
    block_elems = ir.block_extent(axis)
    tile_open = f"{axis}.tile" in open_loops
    block_open = f"{axis}.block" in open_loops
    parts: list[str] = []
    if block_open:
        parts.append(f"i_block_{axis} * {block_elems}")
    if tile_open:
        parts.append(f"i_tile_{axis} * {ptile}")
    if not parts:
        return f"0:{ptile}"
    base = " + ".join(parts)
    _ = scope_tiles
    return f"{base} : {base} + {ptile}"


def _buffer_tile_slice(ir: KernelIR, info: _BufferInfo, open_loops: list[str]) -> str:
    """Slice expression for a per-tile access to ``info``'s 3D buffer.

    Returns ``<name>[0:P_tile, <p_index>, <f_range>]`` where:

    * ``<p_index>`` is the P-axis tile slot — 0 / ``i_tile_p`` /
      ``i_block_p * ltile + i_tile_p`` depending on the P-axis scope.
    * ``<f_range>`` is the F-axis column slice at ``f_tile`` width —
      analogous to ``<p_index>`` but multiplied by ``f_tile``.
    """
    name = info.name
    buf = info.buf
    p_tile, f_tile = buf.tile
    p_axis = buf.p_axis
    f_axis = buf.f_axis

    p_expr = _axis_slot_expr(ir, p_axis, info.per_axis_num_tiles[p_axis], open_loops)
    if f_axis is None:
        return f"{name}[0:{p_tile}, {p_expr}, 0:{f_tile}]"
    f_slots = info.per_axis_num_tiles.get(f_axis, 1)
    f_slot = _axis_slot_expr(ir, f_axis, f_slots, open_loops)
    if f_slot == "0":
        f_range = f"0:{f_tile}"
    else:
        f_slot_paren = f"({f_slot})" if "+" in f_slot else f_slot
        f_range = f"{f_slot_paren} * {f_tile} : {f_slot_paren} * {f_tile} + {f_tile}"
    return f"{name}[0:{p_tile}, {p_expr}, {f_range}]"


def _axis_slot_expr(ir: KernelIR, axis: str, scope_tiles: int, open_loops: list[str]) -> str:
    """Index expression for the current tile on ``axis``.

    * ``scope_tiles == 1`` → ``0``.
    * ``scope_tiles == ltiles_per_block[axis]`` → ``i_tile_<axis>`` when
      the tile loop is open; else ``0`` (the axis folds into a single
      slot at the current depth).
    * ``scope_tiles == num_ltile[axis]`` → ``i_block_<axis> * ltile +
      i_tile_<axis>`` using whichever of those loops are open.
    """
    if scope_tiles <= 1:
        return "0"
    ltile = ir.ltiles_per_block[axis]
    tile_open = f"{axis}.tile" in open_loops
    block_open = f"{axis}.block" in open_loops
    if scope_tiles == ltile:
        return f"i_tile_{axis}" if tile_open else "0"
    """FULL on this axis — combine block + tile indices."""
    parts: list[str] = []
    if block_open:
        parts.append(f"i_block_{axis} * {ltile}")
    if tile_open:
        parts.append(f"i_tile_{axis}")
    if not parts:
        return "0"
    return " + ".join(parts)


"""────────────────────────────────────────────────────────────────
Matmul emission
────────────────────────────────────────────────────────────────"""


def _emit_matmul_call(w: _Writer, ir: KernelIR, plan: _Plan, op: Op, open_loops: list[str]) -> None:
    """Emit the innermost ``nisa.nc_matmul(...)`` call.

    At fire-depth every ``.tile`` loop the matmul touches is open, so
    the operand slice expressions resolve correctly.
    """
    lhs = op.inputs["stationary"]
    rhs = op.inputs["moving"]
    psum_name = _psum_name_of(op)
    psum_info = plan.buffers[psum_name]
    lhs_info = plan.buffers[lhs]
    rhs_info = plan.buffers[rhs]

    psum_expr = _buffer_tile_slice(ir, psum_info, open_loops)
    lhs_expr = _buffer_tile_slice(ir, lhs_info, open_loops)
    rhs_expr = _buffer_tile_slice(ir, rhs_info, open_loops)
    w.line("nisa.nc_matmul(")
    w.indent()
    w.line(f"{psum_expr},")
    w.line(f"stationary={lhs_expr},")
    w.line(f"moving={rhs_expr},")
    w.dedent()
    w.line(")")


def _emit_matmul_drain(w: _Writer, ir: KernelIR, plan: _Plan, op: Op, open_loops: list[str]) -> None:
    """Emit PSUM → SBUF drain. Two variants depending on PSUM scope
    on the reducing dim:

    * **Option B** (PSUM K=PER_BLOCK) — per-K.block fold:
      ``nisa.tensor_tensor(dst=sbuf, data1=sbuf, data2=psum, op=nl.add)``.
    * **Option A** (PSUM K=FULL) — single full-K copy (no
      accumulation): ``nisa.dma_copy(dst=sbuf, src=psum)``.

    When PSUM spans multiple tiles on an axis whose loop is not yet
    open at the drain depth, the drain opens a local loop over that
    axis to iterate one tile per call.
    """
    sbuf_name = op.outputs[0]
    psum_name = _psum_name_of(op)
    sbuf_info = plan.buffers[sbuf_name]
    psum_info = plan.buffers[psum_name]
    is_full_k = _is_full_k_drain(ir, op)

    local_opens: list[str] = []
    for axis in (psum_info.buf.p_axis, psum_info.buf.f_axis):
        if axis is None:
            continue
        tiles = psum_info.per_axis_num_tiles.get(axis, 1)
        if tiles <= 1:
            continue
        tile_entry = f"{axis}.tile"
        block_entry = f"{axis}.block"
        if tile_entry in open_loops and (tiles == ir.ltiles_per_block[axis] or block_entry in open_loops):
            continue
        if tiles == ir.ltiles_per_block[axis] and tile_entry not in open_loops:
            tile_trip = ir.ltiles_per_block[axis]
            w.line(f"for i_tile_{axis} in range({tile_trip}):")
            w.indent()
            local_opens.append(tile_entry)
        elif tiles == ir.num_ltile(axis):
            if block_entry not in open_loops:
                block_trip = ir.num_blocks(axis)
                w.line(f"for i_block_{axis} in range({block_trip}):")
                w.indent()
                local_opens.append(block_entry)
            if tile_entry not in open_loops:
                tile_trip = ir.ltiles_per_block[axis]
                w.line(f"for i_tile_{axis} in range({tile_trip}):")
                w.indent()
                local_opens.append(tile_entry)

    full_opens = open_loops + local_opens
    sbuf_expr = _buffer_tile_slice(ir, sbuf_info, full_opens)
    psum_expr = _buffer_tile_slice(ir, psum_info, full_opens)
    if is_full_k:
        """Option A — PSUM → SBUF dtype-narrowing copy. ``dma_copy``
        rejects PSUM memory regions, so use ``nisa.tensor_copy``."""
        w.line("nisa.tensor_copy(")
        w.indent()
        w.line(f"dst={sbuf_expr},")
        w.line(f"src={psum_expr},")
        w.dedent()
        w.line(")")
    else:
        w.line("nisa.tensor_tensor(")
        w.indent()
        w.line(f"dst={sbuf_expr},")
        w.line(f"data1={sbuf_expr},")
        w.line(f"data2={psum_expr},")
        w.line("op=nl.add,")
        w.dedent()
        w.line(")")

    for _ in local_opens:
        w.dedent()


"""────────────────────────────────────────────────────────────────
Store emission
────────────────────────────────────────────────────────────────"""


def _emit_store(w: _Writer, ir: KernelIR, plan: _Plan, op: Op, open_loops: list[str]) -> None:
    """Emit ``nisa.dma_copy(dst=<hbm>, src=<sbuf>)`` per sbuf tile.

    The store writes exactly one physical tile per call (DMA
    partition-axis contract). Opens local ``.block`` and ``.tile``
    loops on every buffer axis that's held by the SBUF but NOT
    already iterated by an open outer loop — this covers the entire
    valid SBUF region that the drain populated.
    """
    sbuf_name = op.inputs["data"]
    hbm_name = op.outputs[0]
    sbuf_info = plan.buffers[sbuf_name]
    hbm_buf = ir.physical_buffers[hbm_name]

    local_opens: list[str] = []
    for axis in (sbuf_info.buf.p_axis, sbuf_info.buf.f_axis):
        if axis is None:
            continue
        tiles = sbuf_info.per_axis_num_tiles.get(axis, 1)
        block_entry = f"{axis}.block"
        tile_entry = f"{axis}.tile"
        if tiles == ir.num_ltile(axis) and tiles > 1:
            if block_entry not in open_loops and block_entry not in local_opens:
                block_trip = ir.num_blocks(axis)
                w.line(f"for i_block_{axis} in range({block_trip}):")
                w.indent()
                local_opens.append(block_entry)
            if tile_entry not in open_loops and tile_entry not in local_opens:
                tile_trip = ir.ltiles_per_block[axis]
                w.line(f"for i_tile_{axis} in range({tile_trip}):")
                w.indent()
                local_opens.append(tile_entry)
        elif tiles == ir.ltiles_per_block[axis] and tiles > 1:
            if tile_entry not in open_loops and tile_entry not in local_opens:
                tile_trip = ir.ltiles_per_block[axis]
                w.line(f"for i_tile_{axis} in range({tile_trip}):")
                w.indent()
                local_opens.append(tile_entry)
        if axis == sbuf_info.buf.p_axis:
            """P axis always needs block+tile open (DMA partition
            one-per-call), even when tiles == 1."""
            if block_entry not in open_loops and block_entry not in local_opens:
                block_trip = ir.num_blocks(axis)
                w.line(f"for i_block_{axis} in range({block_trip}):")
                w.indent()
                local_opens.append(block_entry)
            if tile_entry not in open_loops and tile_entry not in local_opens:
                tile_trip = ir.ltiles_per_block[axis]
                w.line(f"for i_tile_{axis} in range({tile_trip}):")
                w.indent()
                local_opens.append(tile_entry)

    full_opens = open_loops + local_opens
    sbuf_expr = _buffer_tile_slice(ir, sbuf_info, full_opens)
    hbm_expr = _store_hbm_per_tile_slice(ir, hbm_buf, hbm_name, sbuf_info, full_opens)
    w.line("nisa.dma_copy(")
    w.indent()
    w.line(f"dst={hbm_expr},")
    w.line(f"src={sbuf_expr},")
    w.dedent()
    w.line(")")

    for _ in local_opens:
        w.dedent()


def _store_hbm_per_tile_slice(
    ir: KernelIR, hbm: PhysicalBuffer, hbm_name: str, sbuf_info: _BufferInfo, open_loops: list[str]
) -> str:
    """HBM slice for store — always one ``ptile`` per axis per call,
    offset by the currently open block/tile loops. Matches
    ``_buffer_tile_slice`` on the SBUF side (one tile per dim)."""
    slices: list[str] = []
    for d in hbm.dim_ids:
        if d in (sbuf_info.buf.p_axis, sbuf_info.buf.f_axis):
            slices.append(_hbm_partition_axis_slice(ir, d, open_loops))
        else:
            full = ir.dimensions[d].dim_size
            slices.append(f"0:{full}")
    return f"{hbm_name}[{', '.join(slices)}]"


def _store_sbuf_slice(ir: KernelIR, info: _BufferInfo, open_loops: list[str]) -> str:
    """SBUF slice for the store.

    Store writes one P-tile on the partition axis per DMA, but spans
    the ENTIRE F-axis width the sbuf holds in one call. That's the
    ``num_f_tiles * f_tile`` extent — no f-axis tile slot indexing.
    """
    name = info.name
    buf = info.buf
    p_tile, f_tile = buf.tile
    p_axis = buf.p_axis
    f_axis = buf.f_axis
    p_expr = _axis_slot_expr(ir, p_axis, info.per_axis_num_tiles[p_axis], open_loops)
    if f_axis is None:
        return f"{name}[0:{p_tile}, {p_expr}, 0:{f_tile}]"
    f_width = info.per_axis_num_tiles.get(f_axis, 1) * f_tile
    return f"{name}[0:{p_tile}, {p_expr}, 0:{f_width}]"


def _store_hbm_slice(
    ir: KernelIR, hbm: PhysicalBuffer, hbm_name: str, sbuf_info: _BufferInfo, open_loops: list[str]
) -> str:
    """HBM slice for the store.

    The store's partition axis is sliced one tile per call
    (partition-axis contract). The free axis mirrors the sbuf
    destination's free-axis coverage — a FULL or PER_BLOCK sbuf
    contributes its full block extent in one DMA.
    """
    slices: list[str] = []
    for d in hbm.dim_ids:
        if d == sbuf_info.buf.p_axis:
            slices.append(_hbm_partition_axis_slice(ir, d, open_loops))
        elif d == sbuf_info.buf.f_axis:
            slices.append(_store_free_axis_slice(ir, d, sbuf_info.per_axis_num_tiles[d], open_loops))
        else:
            full = ir.dimensions[d].dim_size
            slices.append(f"0:{full}")
    return f"{hbm_name}[{', '.join(slices)}]"


def _store_free_axis_slice(ir: KernelIR, axis: str, sbuf_tiles: int, open_loops: list[str]) -> str:
    """Free-axis slice for the store.

    The store's free axis emits whatever span the sbuf destination
    covers — one ptile, one block, or the full dim — offset by the
    currently open block/tile loops.
    """
    ptile = ir.dimensions[axis].physical_tile_size
    block_elems = ir.block_extent(axis)
    full = ir.dimensions[axis].dim_size
    tile_open = f"{axis}.tile" in open_loops
    block_open = f"{axis}.block" in open_loops
    if sbuf_tiles == ir.num_ltile(axis):
        return f"0:{full}"
    if sbuf_tiles == ir.ltiles_per_block[axis]:
        if block_open:
            base = f"i_block_{axis} * {block_elems}"
            return f"{base} : {base} + {block_elems}"
        return f"0:{block_elems}"
    parts: list[str] = []
    if block_open:
        parts.append(f"i_block_{axis} * {block_elems}")
    if tile_open:
        parts.append(f"i_tile_{axis} * {ptile}")
    if not parts:
        return f"0:{ptile}"
    base = " + ".join(parts)
    return f"{base} : {base} + {ptile}"
