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

_GADGETS_IMPORT = (
    "from nkigym.codegen.gadgets import (\n"
    "    activation_block,\n"
    "    activation_reduce_block,\n"
    "    allocate_buffers,\n"
    "    dma_transpose_block,\n"
    "    load_block,\n"
    "    matmul_block,\n"
    "    matmul_drain_block,\n"
    "    memset_buffers,\n"
    "    online_matmul_block,\n"
    "    store_block,\n"
    "    tensor_scalar_block,\n"
    "    transpose_block,\n"
    ")"
)

"""Per-op taxonomy for render dispatch:

* ``_LOAD_KINDS``: HBM→SBUF loads, emitted by ``_emit_loads_at_depth``.
* ``_TRANSPOSE_GADGETS``: map NKIOp kind → gadget name for SBUF→SBUF transposes.
* ``_REDUCING_KINDS``: ops whose output accumulates across one of their
  blocking dims — need a memset prologue and a fresh accumulator binding
  at the prologue depth. Matmul and activation_reduce are both reducers.
"""
_LOAD_KINDS = frozenset({"NKILoad"})
_TRANSPOSE_GADGETS: dict[str, str] = {"NKITranspose": "transpose_block", "NKIDMATranspose": "dma_transpose_block"}
_REDUCING_KINDS = frozenset({"NKIMatmul", "NKIActivationReduce", "NKIOnlineMatmul"})
_MATMUL_KINDS = frozenset({"NKIMatmul"})
"""Matmul-style reducers whose PSUM accumulator is hoisted out of the
gadget — ``psum_<output-stem>`` is a physical buffer in the IR, and
the renderer emits a separate ``matmul_drain_block`` once the K loop
closes. ``NKIOnlineMatmul`` is excluded because its per-K rescale
(``O_k = s_k · O_{k-1} + A_k``) is not associative across K: the
drain must fire inside each K iteration, so the gadget keeps its
PSUM scratch private."""


def _accumulator_buf_name(op: Op) -> str:
    """Buffer the reducer ``op`` accumulates into.

    ``NKIMatmul`` accumulates into ``psum_<stem>``; everything else
    (``NKIActivationReduce``, ``NKIOnlineMatmul`` — which manages its
    own PSUM scratch) writes its SBUF output in place.
    """
    out = op.outputs[0]
    if op.kind in _MATMUL_KINDS:
        if not out.startswith("sbuf_"):
            raise ValueError(f"matmul op {op.kind} expected sbuf_-prefixed output, got {out!r}")
        return "psum_" + out[len("sbuf_") :]
    return out


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
    store_op = _store_op(ir)
    phases = _assign_op_phases(ir)

    _emit_allocs_at_depth(w, allocs, depth=0)
    if allocs and any(info.emission_depth == 0 for info in allocs.values()):
        w.line()
    _emit_loads_at_depth(w, ir, allocs, phase=0, depth=0)
    _maybe_emit_reducer_prologue(w, ir, allocs, phases, depth=0)

    _emit_loop_nest(w, ir, allocs, store_op, phases, dim_idx=0)


def _emit_loop_nest(
    w: _Writer, ir: KernelIR, allocs: dict[str, _BufAlloc], store_op: Op, phases: "_PhaseMap", dim_idx: int
) -> None:
    """Recursively emit ``for i_block_<d> in range(...)`` in ``dim_order``.

    Innermost-ACC-dim sibling loops: when the current dim hosts ACC
    reducers AND the compute graph spans multiple phases, emit one loop
    per phase with inter-phase closures (post_op) between them.
    Non-innermost depths keep the single-loop structure.

    At each depth:
    * Emit allocations whose ``emission_depth == dim_idx``.
    * Bind + memset reducer-output slots at the prologue depth.
    * Emit loads at their required depth.
    * Emit compute ops whose natural depth matches this depth
      (i.e. ops whose input/output buffers don't reference any deeper
      dim in ``dim_order``). Keeps reducers whose output doesn't depend
      on inner loop dims from re-firing once per inner-dim iteration.
    * Recurse.
    * Fire the store at the depth immediately outside all ACCUMULATION
      loops (``= _store_emission_depth``).
    """
    here_ops = [op for op in _ops_in_phase(ir, phases, phase=0) if _op_natural_depth(ir, allocs, op) == dim_idx]
    pre_ops, post_ops = _split_here_ops_by_matmul_drain(ir, here_ops, dim_idx)
    _emit_compute_ops(w, ir, allocs, pre_ops)
    if dim_idx == len(ir.dim_order):
        return

    dim = ir.dim_order[dim_idx]
    if not _loop_has_body(ir, allocs, phases, dim_idx):
        _emit_loop_nest(w, ir, allocs, store_op, phases, dim_idx + 1)
        _emit_matmul_drains_at_depth(w, ir, allocs, depth=dim_idx, open_loops=ir.dim_order[:dim_idx])
        _emit_compute_ops(w, ir, allocs, post_ops)
        store_depth = _store_emission_depth(ir)
        if dim_idx == store_depth:
            open_loops = ir.dim_order[:store_depth]
            _emit_store(w, ir, allocs, store_op, open_loops)
        return

    max_phase = max(phases.op_phase.values(), default=0)
    is_phase_split_depth = max_phase > 0 and dim == _phase_split_dim(ir, phases)

    if is_phase_split_depth:
        _emit_phases_at_inner_depth(w, ir, allocs, store_op, phases, dim_idx)
    else:
        w.line(f"for i_block_{dim} in range({ir.num_blocks(dim)}):")
        w.indent()
        _emit_allocs_at_depth(w, allocs, depth=dim_idx + 1)
        _maybe_emit_reducer_prologue(w, ir, allocs, phases, dim_idx + 1)
        _emit_loads_at_depth(w, ir, allocs, phase=0, depth=dim_idx + 1)
        _emit_loop_nest(w, ir, allocs, store_op, phases, dim_idx + 1)
        w.dedent()

    _emit_matmul_drains_at_depth(w, ir, allocs, depth=dim_idx, open_loops=ir.dim_order[:dim_idx])
    _emit_compute_ops(w, ir, allocs, post_ops)
    store_depth = _store_emission_depth(ir)
    if dim_idx == store_depth:
        open_loops = ir.dim_order[:store_depth]
        _emit_store(w, ir, allocs, store_op, open_loops)


def _split_here_ops_by_matmul_drain(ir: KernelIR, here_ops: list[Op], dim_idx: int) -> tuple[list[Op], list[Op]]:
    """Split ``here_ops`` into pre- and post-drain halves.

    An op lands in the post-drain half if any of its inputs is the
    SBUF output of a matmul-style op whose own drain fires at
    ``dim_idx``. Those inputs only become readable once the drain
    has run, so the consumer must be emitted after drains (still at
    the same dim_idx, inside the enclosing ``dim_order[:dim_idx]`` body).

    Ops that don't depend on a drain at this depth stay in the pre-drain
    half and retain their original emission point (before the loop for
    ``dim_order[dim_idx]`` opens), preserving producer-before-consumer
    semantics for all non-drain relationships.
    """
    drain_outputs: set[str] = set()
    for op in ir.ops:
        if op.kind not in _MATMUL_KINDS or not op.outputs:
            continue
        if _matmul_drain_depth(ir, op) == dim_idx:
            drain_outputs.add(op.outputs[0])
    pre: list[Op] = []
    post: list[Op] = []
    for op in here_ops:
        if any(inp in drain_outputs for inp in op.inputs.values()):
            post.append(op)
        else:
            pre.append(op)
    return pre, post


def _emit_phases_at_inner_depth(
    w: _Writer, ir: KernelIR, allocs: dict[str, _BufAlloc], store_op: Op, phases: "_PhaseMap", dim_idx: int
) -> None:
    """Emit one sibling ``for i_block_<dim>`` loop per phase at the phase-split depth.

    Between phase N and phase N+1, emit any reducer-closure ops (post_op
    activation) whose reducer lives in phase N — those run after the
    phase-N loop closes and before the phase-N+1 loop opens.

    Allocations whose ``emission_depth == dim_idx+1`` are emitted once
    before the first sibling loop — they're shared across all phases.
    Reducer prologues (bind + memset) also emit once, before the first
    sibling loop.

    When the phase-split dim is not innermost, each sibling loop must
    still recurse through the remaining ``dim_order`` dims — otherwise
    deeper block indices referenced by op slicing go undefined.
    """
    dim = ir.dim_order[dim_idx]
    _emit_allocs_at_depth(w, allocs, depth=dim_idx + 1)
    _maybe_emit_reducer_prologue(w, ir, allocs, phases, dim_idx + 1)

    max_phase = max(phases.op_phase.values(), default=0)
    for phase in range(max_phase + 1):
        phase_ops = _ops_in_phase(ir, phases, phase)
        if not phase_ops and not _has_phase_loads(ir, phases, phase):
            continue
        w.line(f"for i_block_{dim} in range({ir.num_blocks(dim)}):")
        w.indent()
        _emit_loads_at_depth(w, ir, allocs, phase=phase, depth=dim_idx + 1)
        _emit_phase_tail(w, ir, allocs, store_op, phases, phase, phase_ops, dim_idx + 1)
        w.dedent()
        _emit_phase_closures(w, ir, allocs, phases, phase)


def _emit_phase_tail(
    w: _Writer,
    ir: KernelIR,
    allocs: dict[str, _BufAlloc],
    store_op: Op,
    phases: "_PhaseMap",
    phase: int,
    phase_ops: list[Op],
    dim_idx: int,
) -> None:
    """Open remaining ``dim_order`` loops inside one phase's sibling loop.

    Each deeper dim opens a normal ``for i_block_<d>`` — phase-split
    cannot re-fire (it was consumed at the outer sibling loop). At the
    innermost depth, emit this phase's compute ops.

    Fires the store once, inside the last phase (the one whose ops
    include the store's sbuf producer) at the dedent of the store's
    producer's outermost blocking dim — same rule as the non-phase-
    split path in ``_emit_loop_nest``.
    """
    here_ops = [op for op in phase_ops if _op_natural_depth(ir, allocs, op) == dim_idx]
    _emit_compute_ops(w, ir, allocs, here_ops)
    if dim_idx == len(ir.dim_order):
        return
    dim = ir.dim_order[dim_idx]
    if not _phase_tail_has_body(ir, allocs, phases, phase, phase_ops, dim_idx):
        _emit_phase_tail(w, ir, allocs, store_op, phases, phase, phase_ops, dim_idx + 1)
    else:
        w.line(f"for i_block_{dim} in range({ir.num_blocks(dim)}):")
        w.indent()
        _emit_allocs_at_depth(w, allocs, depth=dim_idx + 1)
        _maybe_emit_reducer_prologue(w, ir, allocs, phases, dim_idx + 1)
        _emit_loads_at_depth(w, ir, allocs, phase=phase, depth=dim_idx + 1)
        _emit_phase_tail(w, ir, allocs, store_op, phases, phase, phase_ops, dim_idx + 1)
        w.dedent()
    _emit_matmul_drains_at_depth(w, ir, allocs, depth=dim_idx, open_loops=ir.dim_order[:dim_idx])
    store_depth = _store_emission_depth(ir)
    if dim_idx == store_depth and _is_store_producer_phase(ir, phases, phase):
        open_loops = ir.dim_order[:store_depth]
        _emit_store(w, ir, allocs, store_op, open_loops)


def _is_store_producer_phase(ir: KernelIR, phases: "_PhaseMap", phase: int) -> bool:
    """True iff the op producing the store's sbuf input lives in ``phase``."""
    store = _store_op(ir)
    sbuf_name = store.inputs["data"]
    for op in ir.ops:
        if sbuf_name in op.outputs:
            return phases.op_phase.get(id(op), 0) == phase
    return phase == 0


def _store_emission_depth(ir: KernelIR) -> int:
    """Depth at which the store fires — literally read from the IR.

    The store's sbuf producer advertises its reducing axes via
    ``Op.blocking_dims``. The store is emitted outside the outermost
    such axis (i.e. at its ``dim_order`` position), so the accumulator
    is complete while outer non-reducing loops keep their block indices
    in scope. No ``DimRole`` fallback — if the producer has no blocking
    dims, the store fires at the bottom of the loop nest exactly as the
    IR implies.
    """
    store_op = _store_op(ir)
    sbuf_name = store_op.inputs["data"]
    producer: Op | None = None
    for op in ir.ops:
        if sbuf_name in op.outputs:
            producer = op
            break
    if producer is None or not producer.blocking_dims:
        return len(ir.dim_order)
    positions = [ir.dim_order.index(d) for d in producer.blocking_dims if d in ir.dim_order]
    if not positions:
        return len(ir.dim_order)
    return min(positions)


def _phase_tail_has_body(
    ir: KernelIR, allocs: dict[str, _BufAlloc], phases: "_PhaseMap", phase: int, phase_ops: list[Op], dim_idx: int
) -> bool:
    """Whether a phase-tail ``for i_block_<d>`` body contains any emitted lines.

    Mirrors :func:`_loop_has_body` for the phase-tail path — limits
    compute-op scanning to the phase's op subset and loads to that phase.
    """
    for depth in range(dim_idx + 1, len(ir.dim_order) + 1):
        if any(info.emission_depth == depth for info in allocs.values()):
            return True
        for op in ir.ops:
            is_load = op.kind in _LOAD_KINDS or (
                op.kind == "NKIDMATranspose" and op.inputs.get("data") in ir.param_names
            )
            if is_load:
                dst = op.outputs[0]
                if dst in allocs and _load_emission_depth(ir, allocs[dst]) == depth:
                    return True
            if op.kind in _REDUCING_KINDS and op.outputs:
                acc_name = _accumulator_buf_name(op)
                if acc_name in allocs and _accumulator_prologue_depth(ir, op, allocs[acc_name]) == depth:
                    return True
        for op in phase_ops:
            if _op_natural_depth(ir, allocs, op) == depth:
                return True
    _ = phase, phases
    return False


def _loop_has_body(ir: KernelIR, allocs: dict[str, _BufAlloc], phases: "_PhaseMap", dim_idx: int) -> bool:
    """Return True iff opening ``for i_block_<dim_order[dim_idx]>`` would
    enclose at least one emitted statement.

    When every deeper loop body is empty (no allocs, no loads, no
    compute ops, no reducer prologues, no store), opening the loop
    produces an ``IndentationError``. This happens when a dim has
    ``num_blocks == 1`` AND every downstream op's natural depth is at
    or outside the current depth.
    """
    for depth in range(dim_idx + 1, len(ir.dim_order) + 1):
        if any(info.emission_depth == depth for info in allocs.values()):
            return True
        for op in ir.ops:
            is_load = op.kind in _LOAD_KINDS or (
                op.kind == "NKIDMATranspose" and op.inputs.get("data") in ir.param_names
            )
            if is_load:
                dst = op.outputs[0]
                if dst in allocs and _load_emission_depth(ir, allocs[dst]) == depth:
                    return True
            if op.kind in _REDUCING_KINDS and op.outputs:
                acc_name = _accumulator_buf_name(op)
                if acc_name in allocs and _accumulator_prologue_depth(ir, op, allocs[acc_name]) == depth:
                    return True
        for op in _ops_in_phase(ir, phases, phase=0):
            if _op_natural_depth(ir, allocs, op) == depth:
                return True
    return False


def _op_natural_depth(ir: KernelIR, allocs: dict[str, _BufAlloc], op: Op) -> int:
    """Depth at which ``op`` belongs in the mechanical loop nest.

    ``= 1 + max(position(d))`` over every dim ``d`` the op references:

    * Dims in ``op.blocking_dims`` (the op iterates on those).
    * Dims appearing on any input/output buffer whose block loop has
      ``num_blocks(d) > 1`` — those dims' ``i_block_<d>`` indices appear
      in the slice expressions the op emits, so the op must be nested
      inside them.

    A dim with ``num_blocks == 1`` is never live at render time — the
    mechanical emitter still opens a ``for ... in range(1)`` for it, but
    the op does not depend on its index and can sit outside.
    """
    used: set[str] = set(op.blocking_dims)
    for name in list(op.inputs.values()) + list(op.outputs):
        if name not in allocs:
            continue
        info = allocs[name]
        for axis in (info.buf.p_axis, info.buf.f_axis):
            if axis is None:
                continue
            if ir.num_blocks(axis) > 1:
                used.add(axis)
    positions = [ir.dim_order.index(d) for d in used if d in ir.dim_order]
    if not positions:
        return 0
    return 1 + max(positions)


def _find_reducing_ops(ir: KernelIR) -> list[Op]:
    """Ops whose output accumulates across a blocking dim — need memset + prologue.

    Matmul reduces along K; ``NKIActivationReduce`` reduces along F. Both
    produce an accumulator SBUF that must be zeroed before the first
    innermost iteration and is live across every iteration of the
    reducing dim's loop.
    """
    return [op for op in ir.ops if op.kind in _REDUCING_KINDS]


"""────────────────────────────────────────────────────────────────
Phase analysis — split compute ops into reduce-closure phases.
────────────────────────────────────────────────────────────────"""


@dataclass
class _PhaseMap:
    """Per-op phase labels for rendering.

    A phase is a maximal set of ops that can share a single sibling
    ``for i_block_<inner_dim>`` loop at the innermost depth. Transitions
    between phases are caused by reducers whose closed output is
    consumed by a downstream op that does *not* block on the same
    dim — classic ``NKIActivationReduce(F→SEQUENTIAL) → NKITensorScalar``
    pattern.

    Attributes:
        op_phase: Map from Op id → integer phase label.
        phase_closers: Map from phase → list of reducer ops whose
            ``post_op`` closure must fire between that phase's loop
            and the next phase's loop.
    """

    op_phase: dict[int, int]
    phase_closers: dict[int, list[Op]]


def _phase_split_dim(ir: KernelIR, phases: _PhaseMap) -> str:
    """The dim whose innermost loop hosts the phase split.

    For sibling-loop rendering we open one loop per phase at this dim's
    depth. The dim is the one blocked by the phase-closing reducer —
    e.g. ``NKIActivationReduce`` with ``F=d1`` blocks on d1, so we split
    at d1's depth.

    Assumption (enforced): every phase-closing reducer in the graph
    shares the same single blocking dim. Multi-dim reducers aren't
    supported yet; ``NotImplementedError`` surfaces the constraint.
    """
    split_dims: set[str] = set()
    for ops in phases.phase_closers.values():
        for op in ops:
            split_dims.update(op.blocking_dims)
    if len(split_dims) != 1:
        raise NotImplementedError(
            f"phase-split requires exactly one shared blocking dim across closers; got {split_dims}"
        )
    return next(iter(split_dims))


def _assign_op_phases(ir: KernelIR) -> _PhaseMap:
    """Topologically assign each op a phase number for sibling-loop emission.

    Rule: an op's phase = max(phase of its producer ops) + bump, where
    ``bump = 1`` iff any producer is a reducer (in ``_REDUCING_KINDS``)
    whose blocking dim is not also blocked by this consumer. That is:
    "consumer needs reducer's closed result" ⇒ consumer must run after
    the reducer's loop closes ⇒ new phase.

    Matmul is a reducer, but store (its direct consumer) is an HBM op
    that already fires outside the innermost ACC loop via
    ``_store_emission_depth`` — it never triggers a phase bump.
    """
    op_phase: dict[int, int] = {}
    producer: dict[str, Op] = {}
    for op in ir.ops:
        for out in op.outputs:
            producer[out] = op

    phase_closers: dict[int, list[Op]] = {}
    for op in ir.ops:
        if op.kind == "NKIStore":
            op_phase[id(op)] = 0
            continue
        own_phase = 0
        for tname in op.inputs.values():
            prod = producer.get(tname)
            if prod is None:
                continue
            prod_phase = op_phase.get(id(prod), 0)
            bump = 0
            """Only ``NKIActivationReduce`` fires a phase split — its
            ``post_op`` closure (e.g. ``rsqrt``) runs once the F-reduce
            loop closes, separating pre-reduce and post-reduce ops into
            sibling loops. Matmul-style reducers drain through a PSUM
            gadget at ``_matmul_drain_depth``; their SBUF output is a
            plain forward edge to downstream ops and does not split
            phases."""
            if prod.kind == "NKIActivationReduce" and not (prod.blocking_dims & op.blocking_dims):
                bump = 1
                phase_closers.setdefault(prod_phase, [])
                if prod not in phase_closers[prod_phase]:
                    phase_closers[prod_phase].append(prod)
            own_phase = max(own_phase, prod_phase + bump)
        op_phase[id(op)] = own_phase
    return _PhaseMap(op_phase=op_phase, phase_closers=phase_closers)


def _ops_in_phase(ir: KernelIR, phases: _PhaseMap, phase: int) -> list[Op]:
    """Non-load, non-store ops labelled with ``phase``."""
    result: list[Op] = []
    for op in ir.ops:
        if op.kind in _LOAD_KINDS or op.kind == "NKIStore":
            continue
        if op.kind == "NKIDMATranspose" and op.inputs.get("data") in ir.param_names:
            continue
        if phases.op_phase.get(id(op), 0) == phase:
            result.append(op)
    return result


def _has_phase_loads(ir: KernelIR, phases: _PhaseMap, phase: int) -> bool:
    """True iff any HBM-load op should fire inside the ``phase`` sibling loop.

    Loads without a direct reducer-fed consumer belong in phase 0; loads
    feeding only phase-N ops (N > 0) would belong in phase N. In
    practice for rmsnorm+matmul, every load is consumed in phase 0
    (NKIActivationReduce consumes lhs, matmul consumes lhs_T which is
    phase-1 but the raw load still sits in phase 0 per how we bind
    cur_sbuf_lhs within both sibling loops). This predicate is used to
    decide whether to open an otherwise-empty phase loop.
    """
    _ = phases, phase
    return False


def _emit_phase_operand_rebindings(
    w: _Writer, ir: KernelIR, allocs: dict[str, _BufAlloc], phases: _PhaseMap, phase_ops: list[Op], loads_emitted: bool
) -> None:
    """Re-bind ``cur_<name>`` for every input buffer the phase's ops read.

    Buffers whose rotation-slot or block-index depends on the phase-
    split dim would otherwise keep a stale binding from the prior
    phase's final iteration. When ``loads_emitted`` is True, the loads
    themselves already bound ``cur_<name>`` for their destinations —
    skip those to avoid duplicates.
    """
    _ = phases
    already: set[str] = set()
    if loads_emitted:
        for op in ir.ops:
            if op.kind == "NKILoad":
                already.add(op.outputs[0])
            if op.kind == "NKIDMATranspose" and op.inputs.get("data") in ir.param_names:
                already.add(op.outputs[0])
    seen: set[str] = set()
    for op in phase_ops:
        for tname in op.inputs.values():
            if tname in already or tname in seen or tname not in allocs:
                continue
            seen.add(tname)
            info = allocs[tname]
            w.line(f"{_cur_name(tname)} = {tname}{_rotation_index(info)}")


def _emit_phase_closures(w: _Writer, ir: KernelIR, allocs: dict[str, _BufAlloc], phases: _PhaseMap, phase: int) -> None:
    """Emit ``activation_block(post_op)`` for every reducer closing phase ``phase``.

    Fires between the phase-N sibling loop's dedent and the phase-(N+1)
    loop's open. Only ``NKIActivationReduce`` with a ``post_op`` kwarg
    emits code here; other reducers leave their accumulator unchanged.

    The closure reads and writes the same per-block slice the reduce
    accumulated into, so both operands use
    :func:`_tensor_scalar_operand_slice` to narrow to the current
    block. OUTER-scoped accumulators otherwise apply ``rsqrt`` over
    uninitialized or foreign-block partitions.
    """
    for op in phases.phase_closers.get(phase, []):
        if op.kind != "NKIActivationReduce":
            continue
        post_op = op.kwargs.get("post_op")
        if post_op is None:
            continue
        out_name = op.outputs[0]
        info = allocs[out_name]
        cur = _cur_name(out_name)
        scale = op.kwargs.get("scale", 1.0)
        bias = op.kwargs.get("bias", 0.0)
        scale_repr = _scale_bias_repr(scale)
        bias_repr = _scale_bias_repr(bias)
        dst_expr = _tensor_scalar_operand_slice(ir, info, base=cur)
        w.line(f"activation_block({dst_expr}, {dst_expr}, op=nl.{post_op}, scale={scale_repr}, bias={bias_repr})")


def _scale_bias_repr(v: float) -> str:
    """Render a float literal for the emitted source."""
    return repr(v)


def _emit_compute_ops(w: _Writer, ir: KernelIR, allocs: dict[str, _BufAlloc], ops: list[Op]) -> None:
    """Emit the compute ops in ``ops`` at the current loop depth.

    Loads are handled by ``_emit_loads_at_depth`` (depth-aware, emit once
    per enclosing-dim binding). ``NKIStore`` is emitted by ``_emit_store``
    after the reducing loops close. Every other op dispatches by kind.
    """
    for op in ops:
        if op.kind in _TRANSPOSE_GADGETS:
            if op.kind == "NKIDMATranspose" and op.inputs["data"] in ir.param_names:
                continue
            _emit_transpose(w, ir, allocs, op, _TRANSPOSE_GADGETS[op.kind])
        elif op.kind == "NKIMatmul":
            _emit_matmul(w, ir, allocs, op)
        elif op.kind == "NKIOnlineMatmul":
            _emit_online_matmul(w, ir, allocs, op)
        elif op.kind == "NKIActivationReduce":
            _emit_activation_reduce(w, ir, allocs, op)
        elif op.kind == "NKIActivation":
            _emit_activation(w, ir, allocs, op)
        elif op.kind == "NKITensorScalar":
            _emit_tensor_scalar(w, ir, allocs, op)


def _emit_transpose(w: _Writer, ir: KernelIR, allocs: dict[str, _BufAlloc], op: Op, gadget: str) -> None:
    """Emit ``cur_<dst> = <dst>[rotation]; <gadget>(dst_expr, src_expr)``.

    Both operands live at the tightest enclosing loop (same rotation
    discipline as loads). When src/dst buffers hold more tiles than
    one block along their axes (OUTER/MIDDLE scope), the gadget input
    is sliced down so the gadget's inner ``num_k × num_m`` product
    equals ``lt[P] × lt[F]`` regardless of scope.
    """
    src = op.inputs["data"]
    dst = op.outputs[0]
    dst_info = allocs[dst]
    src_info = allocs[src]
    cur_dst = _cur_name(dst)
    w.line(f"{cur_dst} = {dst}{_rotation_index(dst_info)}")
    dst_expr = _transpose_operand_slice(ir, dst_info, base=cur_dst)
    src_expr = _transpose_operand_slice(ir, src_info, base=_cur_name(src))
    w.line(f"{gadget}({dst_expr}, {src_expr})")


def _transpose_operand_slice(ir: KernelIR, info: _BufAlloc, base: str) -> str:
    """Slice an operand down to one block on each of its axes.

    P-axis slice → picks ``ltiles_per_block[p_axis]`` contiguous
    P-slots. F-axis slice → per-leaf list-comprehension narrows each
    leaf's free-axis width to one block.
    """
    buf = info.buf
    expr = base
    p_lt = ir.ltiles_per_block.get(buf.p_axis, 1)
    if info.num_p_tiles > p_lt:
        expr = f"{expr}[i_block_{buf.p_axis} * {p_lt} : i_block_{buf.p_axis} * {p_lt} + {p_lt}]"
    if buf.f_axis is not None:
        f_lt = ir.ltiles_per_block.get(buf.f_axis, 1)
        if info.num_f_tiles > f_lt:
            f_width = info.f_tile * f_lt
            expr = (
                f"[leaf[:, i_block_{buf.f_axis} * {f_width} : i_block_{buf.f_axis} * {f_width} + {f_width}] "
                f"for leaf in {expr}]"
            )
    return expr


"""────────────────────────────────────────────────────────────────
Buffer sizing
────────────────────────────────────────────────────────────────"""


def _resolve_allocations(ir: KernelIR) -> dict[str, _BufAlloc]:
    """Compute per-buffer ``(p_tile, num_p_tiles, f_tile, num_f_tiles, emission_depth)``.

    Every non-HBM physical buffer must carry a ``buffer_scopes`` entry —
    the renderer lowers scope to extents via :func:`_scope_extents` and
    nothing more. No special-casing per op kind, per dim role, or per
    buffer name: if the IR omits a scope, that is a validator concern,
    not a render-side heuristic.

    The HBM destination is sized to the full tensor shape and is
    declared via the header, never allocated with ``allocate_buffers``.
    """
    result: dict[str, _BufAlloc] = {}
    for name, buf in ir.physical_buffers.items():
        if buf.loc == "hbm":
            continue
        if name not in ir.buffer_scopes:
            raise ValueError(
                f"buffer {name!r} has no buffer_scopes entry — the renderer "
                f"mechanically lowers IR knobs and does not invent defaults"
            )
        p_tile, num_p, f_tile, num_f = _scope_extents(ir, buf, ir.buffer_scopes[name])
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
        f"loc=nl.{info.buf.loc}, dtype=nl.{info.buf.dtype}, "
        f"num_p_buffers={p}, num_f_buffers={f})"
    )


"""────────────────────────────────────────────────────────────────
Accumulator prologue
────────────────────────────────────────────────────────────────"""


def _maybe_emit_reducer_prologue(
    w: _Writer, ir: KernelIR, allocs: dict[str, _BufAlloc], phases: _PhaseMap, depth: int
) -> None:
    """Emit ``cur_<acc> = <acc>[...]; memset_buffers(cur_<acc>, 0.0)``
    for every reducing op whose accumulator's slot first resolves at
    ``depth``.

    Matmul-style ops accumulate into ``psum_<stem>`` (the K-accumulator
    hoisted out of the gadget); ``NKIActivationReduce`` accumulates into
    its own SBUF output. Both fire the same binding+memset pattern at
    the same rule — the accumulator buffer's block-indexed first-use
    depth.
    """
    _ = phases
    for op in _find_reducing_ops(ir):
        acc_name = _accumulator_buf_name(op)
        info = allocs[acc_name]
        if depth != _accumulator_prologue_depth(ir, op, info):
            continue
        cur = _cur_name(acc_name)
        w.line(f"{cur} = {acc_name}{_rotation_index(info)}")
        w.line(f"memset_buffers({cur}, 0.0)")


def _accumulator_prologue_depth(ir: KernelIR, reducing_op: Op, info: "_BufAlloc") -> int:
    """Depth where the accumulator's ``cur_<name>`` slot binding fires.

    Same rule as ``_load_emission_depth``: ``1 + max(dim_order.index(d))``
    over the accumulator's block-indexed axes (those the buffer's scope
    leaves per-block rather than full-extent). Block-indexed axes are
    the ones whose ``i_block_<d>`` index appears in the slot
    expression, so the prologue must sit inside every such loop.

    When no axes are block-indexed (OUTER scope), the prologue fires at
    depth 0 (unconditionally, before any loop opens).
    """
    _ = reducing_op
    block_axes = _block_indexed_axes(ir, info)
    if not block_axes:
        return 0
    positions = [ir.dim_order.index(d) for d in block_axes if d in ir.dim_order]
    if not positions:
        return 0
    return 1 + max(positions)


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


def _emit_loads_at_depth(w: _Writer, ir: KernelIR, allocs: dict[str, _BufAlloc], phase: int, depth: int) -> None:
    """Emit every HBM-sourced op whose destination buffer wants to appear here.

    That's ``NKILoad`` always, plus ``NKIDMATranspose`` ops whose
    ``data`` input is a kernel parameter (HBM tensor) — i.e. fused
    load-transposes. Each fires at the depth equal to ``1 +`` the
    deepest ``dim_order`` position of any of its destination buffer's
    rotating block axes.

    The renderer is a mechanical lowering — loads fire at their natural
    depth regardless of phase membership. Phases that don't consume the
    loaded buffer simply re-execute the load; dead-load elimination is
    a later rotation-aware pass.
    """
    _ = phase
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
Activation-reduce / tensor-scalar emission.
────────────────────────────────────────────────────────────────"""


def _emit_activation_reduce(w: _Writer, ir: KernelIR, allocs: dict[str, _BufAlloc], op: Op) -> None:
    """Emit ``activation_reduce_block(cur_red[slice], cur_data[slice], op=..., reduce_op=...)``.

    ``post_op`` / ``scale`` / ``bias`` kwargs apply once the reducing
    loop closes — emitted by :func:`_emit_phase_closures`. Here we just
    do one accumulation step per iteration.

    Both dst and src get per-block P/F slicing via
    :func:`_tensor_scalar_operand_slice`. The gadget iterates over
    P-slots 0..len(src)-1 writing into dst[0..len(src)-1], so an
    OUTER-scoped dst holding full-dim P-slots must be narrowed to
    the current block's range or every d0 block's reduction
    overwrites the same prefix of dst partitions.
    """
    src = op.inputs["data"]
    dst = op.outputs[0]
    dst_info = allocs[dst]
    src_info = allocs[src]
    """``cur_<dst>`` was bound by the reducer prologue outside the
    innermost loop — we index into that binding rather than rebinding."""
    cur_dst = _cur_name(dst)
    cur_src = _cur_name(src)
    act = op.kwargs.get("op", "copy")
    red = op.kwargs.get("reduce_op", "add")
    dst_expr = _tensor_scalar_operand_slice(ir, dst_info, base=cur_dst)
    src_expr = _tensor_scalar_operand_slice(ir, src_info, base=cur_src)
    w.line(f"activation_reduce_block({dst_expr}, {src_expr}, op=nl.{act}, reduce_op=nl.{red})")


def _emit_activation(w: _Writer, ir: KernelIR, allocs: dict[str, _BufAlloc], op: Op) -> None:
    """Emit ``cur_<dst> = <dst>[...]; activation_block(cur_dst, cur_src, op=..., scale=..., bias=...)``.

    ``NKIActivation`` is a standalone Scalar-Engine pass — no F-reduce,
    no post_op phase. Both operands are sliced per-block on their P
    axis via :func:`_broadcast_operand_slice` (the F axis is always 1
    for ``(P,)`` tensors, so no per-leaf free-slice is needed).
    """
    src = op.inputs["data"]
    dst = op.outputs[0]
    dst_info = allocs[dst]
    src_info = allocs[src]
    cur_dst = _cur_name(dst)
    cur_src = _cur_name(src)
    w.line(f"{cur_dst} = {dst}{_rotation_index(dst_info)}")
    dst_expr = _broadcast_operand_slice(ir, dst_info, base=cur_dst)
    src_expr = _broadcast_operand_slice(ir, src_info, base=cur_src)
    act = op.kwargs.get("op", "copy")
    scale = op.kwargs.get("scale", 1.0)
    bias = op.kwargs.get("bias", 0.0)
    w.line(
        f"activation_block({dst_expr}, {src_expr}, op=nl.{act}, "
        f"scale={_scale_bias_repr(scale)}, bias={_scale_bias_repr(bias)})"
    )


def _emit_tensor_scalar(w: _Writer, ir: KernelIR, allocs: dict[str, _BufAlloc], op: Op) -> None:
    """Emit ``tensor_scalar_block(cur_dst, cur_data, cur_operand0, op=...)``.

    ``operand0`` is a ``(P,)`` per-row vector. Data + output are
    ``(P, F)`` tiles sliced down to one block on each of their axes
    (matches the transpose-operand slicing rule).
    """
    src = op.inputs["data"]
    operand = op.inputs["operand0"]
    dst = op.outputs[0]
    dst_info = allocs[dst]
    src_info = allocs[src]
    op_info = allocs[operand]
    cur_dst = _cur_name(dst)
    cur_src = _cur_name(src)
    cur_op = _cur_name(operand)
    w.line(f"{cur_dst} = {dst}{_rotation_index(dst_info)}")
    dst_expr = _tensor_scalar_operand_slice(ir, dst_info, base=cur_dst)
    src_expr = _tensor_scalar_operand_slice(ir, src_info, base=cur_src)
    operand_expr = _broadcast_operand_slice(ir, op_info, base=cur_op)
    math_op = op.kwargs.get("op", "multiply")
    w.line(f"tensor_scalar_block({dst_expr}, {src_expr}, {operand_expr}, op=nl.{math_op})")


def _tensor_scalar_operand_slice(ir: KernelIR, info: _BufAlloc, base: str) -> str:
    """Narrow a (P, F) operand down to one block on each axis, by-leaf on F."""
    buf = info.buf
    expr = base
    p_lt = ir.ltiles_per_block.get(buf.p_axis, 1)
    if info.num_p_tiles > p_lt:
        expr = f"{expr}[i_block_{buf.p_axis} * {p_lt} : i_block_{buf.p_axis} * {p_lt} + {p_lt}]"
    if buf.f_axis is not None:
        f_lt = ir.ltiles_per_block.get(buf.f_axis, 1)
        if info.num_f_tiles > f_lt:
            f_width = info.f_tile * f_lt
            expr = (
                f"[leaf[:, i_block_{buf.f_axis} * {f_width} : i_block_{buf.f_axis} * {f_width} + {f_width}] "
                f"for leaf in {expr}]"
            )
    return expr


def _broadcast_operand_slice(ir: KernelIR, info: _BufAlloc, base: str) -> str:
    """``operand0`` for ``tensor_scalar_block`` — a (P,) per-row vector.

    Only the P-axis may need per-block slicing; there's no F axis.
    """
    buf = info.buf
    expr = base
    p_lt = ir.ltiles_per_block.get(buf.p_axis, 1)
    if info.num_p_tiles > p_lt:
        expr = f"{expr}[i_block_{buf.p_axis} * {p_lt} : i_block_{buf.p_axis} * {p_lt} + {p_lt}]"
    return expr


"""────────────────────────────────────────────────────────────────
Matmul emission
────────────────────────────────────────────────────────────────"""


def _emit_matmul(w: _Writer, ir: KernelIR, allocs: dict[str, _BufAlloc], op: Op) -> None:
    """Emit ``matmul_block(psum_slice, lhs_T, rhs)`` at the innermost loop body.

    The accumulator lives in PSUM under name ``psum_<stem>`` — fed by
    the prologue at the matmul's K-enclosing depth. The gadget does one
    ``nc_matmul`` per K block, accumulating additively into the PSUM
    slot.

    Drain placement depends on K's iteration count:

    * All blocking dims with ``num_blocks == 1`` — drain inline right
      after the call (PSUM is complete after a single ``nc_matmul``).
      Needed when a downstream consumer of the SBUF output is nested
      inside a sibling non-K loop deeper than the producer's natural
      depth; post-recursion drain would land after the consumer reads
      a stale slot.
    * Multi-iter K — drain at :func:`_matmul_drain_depth` post-recursion
      via :func:`_emit_matmul_drains_at_depth`.
    """
    k_dim = op.axis_map["K"]
    m_dim = op.axis_map["M"]
    n_dim = op.axis_map["N"]
    lhs = op.inputs["stationary"]
    rhs = op.inputs["moving"]
    psum_name = _accumulator_buf_name(op)
    psum_info = allocs[psum_name]

    psum_expr = _matmul_output_expr(ir, psum_info, m_dim, n_dim)
    lhs_expr = _matmul_input_expr(ir, allocs, lhs, k_dim, m_dim)
    rhs_expr = _matmul_input_expr(ir, allocs, rhs, k_dim, n_dim)
    w.line(f"matmul_block({psum_expr}, {lhs_expr}, {rhs_expr})")

    if not _has_multi_iter_blocking(ir, op):
        open_loops = ir.dim_order[: _op_natural_depth(ir, allocs, op)]
        _emit_matmul_drain(w, ir, allocs, op, open_loops)


def _emit_online_matmul(w: _Writer, ir: KernelIR, allocs: dict[str, _BufAlloc], op: Op) -> None:
    """Emit ``online_matmul_block(out_slice, lhs_T, rhs, scale)``.

    Same (K, M, N) operand shaping as :func:`_emit_matmul`, plus a
    ``(P=M,)`` scale operand narrowed to one M-block when its buffer
    holds more. Unlike :func:`_emit_matmul`, the online gadget keeps
    its PSUM scratch internal — the ``s_k`` rescale fires inside each
    K iteration, so no external drain is emitted.
    """
    k_dim = op.axis_map["K"]
    m_dim = op.axis_map["M"]
    n_dim = op.axis_map["N"]
    lhs = op.inputs["stationary"]
    rhs = op.inputs["moving"]
    scale = op.inputs["scale"]
    out = op.outputs[0]
    out_info = allocs[out]
    scale_info = allocs[scale]

    out_expr = _matmul_output_expr(ir, out_info, m_dim, n_dim)
    lhs_expr = _matmul_input_expr(ir, allocs, lhs, k_dim, m_dim)
    rhs_expr = _matmul_input_expr(ir, allocs, rhs, k_dim, n_dim)
    scale_expr = _broadcast_operand_slice(ir, scale_info, base=_cur_name(scale))
    w.line(f"online_matmul_block({out_expr}, {lhs_expr}, {rhs_expr}, {scale_expr})")


def _emit_matmul_drains_at_depth(
    w: _Writer, ir: KernelIR, allocs: dict[str, _BufAlloc], depth: int, open_loops: list[str]
) -> None:
    """Emit PSUM→SBUF drain gadgets for matmul-style ops whose drain
    fires at ``depth``.

    Each ``NKIMatmul`` accumulates into PSUM across its own
    ``blocking_dims``. Drain placement depends on whether the K loop
    iterates:

    * Multi-iter K (any blocking dim with ``num_blocks > 1``) — drain
      at ``min(dim_order.index(d))`` over those multi-iter dims, i.e.
      right after the outermost multi-iter K loop closes.
    * All-1-iter K — drain inline at the producer's natural depth
      right after the single ``matmul_block`` call. Post-recursion
      placement would land below any consumer nested inside a sibling
      non-K loop at the same depth, causing the consumer to read a
      stale SBUF slot.

    The inline case is emitted by :func:`_emit_inline_matmul_drain`
    (called immediately after each matmul call in
    :func:`_emit_compute_ops`); this function handles only the
    post-recursion case.
    """
    for op in ir.ops:
        if op.kind not in _MATMUL_KINDS or not op.outputs:
            continue
        if not _has_multi_iter_blocking(ir, op):
            continue
        if _matmul_drain_depth(ir, op) != depth:
            continue
        _emit_matmul_drain(w, ir, allocs, op, open_loops)


def _has_multi_iter_blocking(ir: KernelIR, op: Op) -> bool:
    """True iff any of ``op``'s blocking dims has ``num_blocks > 1``."""
    return any(ir.num_blocks(d) > 1 for d in op.blocking_dims if d in ir.dim_order)


def _matmul_drain_depth(ir: KernelIR, op: Op) -> int:
    """Post-recursion depth where ``op``'s PSUM is drained to SBUF.

    Defined only for multi-iter K: returns the ``dim_order`` position
    of the outermost multi-iter blocking dim — the drain fires right
    after that loop closes. All-1-iter K matmuls drain inline at their
    natural depth (see :func:`_emit_inline_matmul_drain`); this helper
    does not apply to them.
    """
    positions = [ir.dim_order.index(d) for d in op.blocking_dims if d in ir.dim_order and ir.num_blocks(d) > 1]
    if not positions:
        return len(ir.dim_order)
    return min(positions)


def _emit_matmul_drain(w: _Writer, ir: KernelIR, allocs: dict[str, _BufAlloc], op: Op, open_loops: list[str]) -> None:
    """Emit the PSUM → SBUF drain once the matmul's K loop closes.

    Runs at the same depth as the store (right after all blocking loops
    of the matmul producer close). Binds ``cur_<sbuf_out>`` so the
    store can reference it, then calls the drain gadget. Slicing mirrors
    :func:`_emit_store` — the SBUF accumulator may be OUTER/MIDDLE and
    thus wider than one (m, n) block; the drain sees only the current
    block's slice.
    """
    m_dim = op.axis_map["M"]
    n_dim = op.axis_map["N"]
    sbuf_name = op.outputs[0]
    psum_name = _accumulator_buf_name(op)
    sbuf_info = allocs[sbuf_name]
    psum_info = allocs[psum_name]

    cur_sbuf = _cur_name(sbuf_name)
    w.line(f"{cur_sbuf} = {sbuf_name}{_rotation_index(sbuf_info)}")
    sbuf_expr = _drain_sbuf_slice(ir, sbuf_info, m_dim, n_dim, open_loops, base=cur_sbuf)
    psum_expr = _matmul_output_expr(ir, psum_info, m_dim, n_dim)
    w.line(f"matmul_drain_block({sbuf_expr}, {psum_expr})")


def _drain_sbuf_slice(ir: KernelIR, info: _BufAlloc, m_dim: str, n_dim: str, open_loops: list[str], base: str) -> str:
    """Slice an SBUF accumulator buffer to the current (m, n) block.

    The buffer may be OUTER/MIDDLE and hold more than one block on
    either axis. Only slices axes whose ``i_block_<d>`` is still open
    at drain time; axes already closed are fully covered by the
    accumulator (enforced by ``_check_accumulator_coverage``).
    """
    buf = info.buf
    expr = base
    lt_m = ir.ltiles_per_block[m_dim]
    lt_n = ir.ltiles_per_block[n_dim]
    if buf.p_axis == m_dim and m_dim in open_loops and info.num_p_tiles > lt_m:
        expr = f"{expr}[i_block_{m_dim} * {lt_m} : i_block_{m_dim} * {lt_m} + {lt_m}]"
    if buf.f_axis == n_dim and n_dim in open_loops and info.num_f_tiles > lt_n:
        f_width = info.f_tile * lt_n
        expr = f"[leaf[:, i_block_{n_dim} * {f_width} : i_block_{n_dim} * {f_width} + {f_width}] for leaf in {expr}]"
    return expr


def _matmul_output_expr(ir: KernelIR, info: _BufAlloc, m_dim: str, n_dim: str) -> str:
    """Accumulator slice for ``matmul_block``.

    * M on P-axis, accumulator wider than one M-block → slice P-slot list.
    * N on free-axis, accumulator wider than one N-block → per-leaf slice
      the free axis via a list comprehension.
    """
    expr = _cur_name(info.name)
    buf = info.buf
    lt_m = ir.ltiles_per_block[m_dim]
    lt_n = ir.ltiles_per_block[n_dim]
    if buf.p_axis == m_dim and info.num_p_tiles > lt_m:
        expr = f"{expr}[i_block_{m_dim} * {lt_m} : i_block_{m_dim} * {lt_m} + {lt_m}]"
    if buf.f_axis == n_dim and info.num_f_tiles > lt_n:
        f_width = info.f_tile * lt_n
        expr = f"[leaf[:, i_block_{n_dim} * {f_width} : i_block_{n_dim} * {f_width} + {f_width}] for leaf in {expr}]"
    return expr


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


def _emit_store(w: _Writer, ir: KernelIR, allocs: dict[str, _BufAlloc], store_op: Op, open_loops: list[str]) -> None:
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
    if f_axis is not None and f_axis in open_loops and info.num_f_tiles > ir.ltiles_per_block[f_axis]:
        """Free-axis per-block slicing: the sbuf is a list of P-slot
        leaves, so slice each leaf's free axis via a list comprehension."""
        lt = ir.ltiles_per_block[f_axis]
        f_width = info.f_tile * lt
        sbuf_expr = (
            f"[leaf[:, i_block_{f_axis} * {f_width} : i_block_{f_axis} * {f_width} + {f_width}] "
            f"for leaf in {sbuf_expr}]"
        )

    w.line(f"store_block({hbm_name}[{', '.join(hbm_slices)}], {sbuf_expr})")
