"""Lightweight validity checks over a ``KernelIR``'s knob assignment.

These are cheap, mechanical guards meant to filter random samples
before rendering — no rendering, no exceptions. Two entry points:

* :func:`is_valid` — boolean fast-path for sampler reject loops.
* :func:`validity_report` — structured list of every failed check
  with a fix hint. Use this in tuning loops to decide which knob to
  flip instead of discarding a near-miss IR.
"""

from dataclasses import dataclass

from nkigym.kernel_ir.ir import BufferScope, KernelIR, Op
from nkigym.kernel_ir.types import DimRole

_REDUCING_KINDS = frozenset({"NKIMatmul", "NKIActivationReduce", "NKIOnlineMatmul"})
"""Ops whose output SBUF is a loop-lived accumulator — its prologue
(``cur_<name> = <name>[...]; memset``) fires at
:func:`_accumulator_first_use_depth` rather than at the op's emission
depth. Must match :data:`nkigym.codegen.render._REDUCING_KINDS`."""

_MATMUL_KINDS = frozenset({"NKIMatmul"})
"""Matmul-style reducers whose PSUM accumulator is hoisted out of the
gadget — ``psum_<stem>`` is a physical buffer in the IR. Mirrors
:data:`nkigym.codegen.render._MATMUL_KINDS`. ``NKIOnlineMatmul`` is
excluded (per-K rescale keeps its PSUM scratch internal)."""


def _accumulator_buf_name(op: Op) -> str:
    """Buffer the reducer ``op`` accumulates into — PSUM sibling for
    :data:`_MATMUL_KINDS`, its own SBUF output otherwise."""
    out = op.outputs[0]
    if op.kind in _MATMUL_KINDS:
        if not out.startswith("sbuf_"):
            raise ValueError(f"matmul op {op.kind} expected sbuf_-prefixed output, got {out!r}")
        return "psum_" + out[len("sbuf_") :]
    return out


@dataclass(frozen=True)
class ValidityFailure:
    """One failed validity check, plus a concrete fix hint.

    Attributes:
        check: Identifier of the failed check — one of
            ``emission_depth_ceiling``, ``rotation_axis_closed``,
            ``transpose_scope_mismatch``, ``accumulator_coverage``.
        buffer: Buffer name the failure pertains to, or ``None`` when
            the failure is whole-IR.
        detail: Human-readable description of what went wrong
            (actual vs. expected numbers).
        fix_hint: One-line actionable suggestion naming the knob to
            adjust and the direction of adjustment.
    """

    check: str
    buffer: str | None
    detail: str
    fix_hint: str


def is_valid(ir: KernelIR) -> bool:
    """Return ``True`` iff every validity check passes."""
    return not validity_report(ir)


def validity_report(ir: KernelIR) -> list[ValidityFailure]:
    """Return every failed validity check on ``ir``.

    Empty list = valid. Each :class:`ValidityFailure` names the
    check, the buffer, the numerical detail, and a one-line fix
    hint. Useful when a tuning loop wants to nudge a single knob
    rather than discard the whole IR.
    """
    failures: list[ValidityFailure] = []
    failures.extend(_check_emission_depth(ir))
    failures.extend(_check_rotation_axes(ir))
    failures.extend(_check_transpose_scopes(ir))
    failures.extend(_check_accumulator_coverage(ir))
    failures.extend(_check_matmul_psum_outside_k(ir))
    return failures


def _check_matmul_psum_outside_k(ir: KernelIR) -> list[ValidityFailure]:
    """For every matmul-style op, PSUM prologue must fire OUTSIDE the K loop.

    The PSUM accumulator's memset fires at ``1 + max(pos of PSUM's
    block-indexed axes)`` in ``dim_order``. For the memset to run once
    per (M, N) block (not once per K iter), every **blocking dim with
    ``num_blocks > 1``** must sit at or deeper than that prologue
    depth. Blocking dims whose extent fits in a single tile have
    ``num_blocks == 1``: the loop runs once and cannot re-zero the
    accumulator, so their position is unconstrained.
    """
    failures: list[ValidityFailure] = []
    for op in ir.ops:
        if op.kind not in _MATMUL_KINDS or not op.outputs:
            continue
        psum_name = _accumulator_buf_name(op)
        if psum_name not in ir.physical_buffers:
            continue
        psum_block_axes = _block_indexed_axes(ir, psum_name)
        out_positions = [ir.dim_order.index(d) for d in psum_block_axes if d in ir.dim_order]
        block_positions = [
            ir.dim_order.index(d) for d in op.blocking_dims if d in ir.dim_order and ir.num_blocks(d) > 1
        ]
        if not out_positions or not block_positions:
            continue
        prologue_depth = 1 + max(out_positions)
        first_block_pos = min(block_positions)
        if first_block_pos < prologue_depth:
            failures.append(
                ValidityFailure(
                    check="matmul_psum_outside_k",
                    buffer=psum_name,
                    detail=(
                        f"PSUM prologue fires at depth {prologue_depth} (inside {op.blocking_dims}) "
                        f"but the outermost matmul blocking dim sits at position {first_block_pos} — "
                        f"the memset would re-zero the accumulator every K iteration"
                    ),
                    fix_hint=(
                        f"move every dim in {op.blocking_dims} to a position ≥ {prologue_depth} "
                        f"in dim_order, or widen {psum_name} scope so fewer output axes are "
                        f"block-indexed (OUTER — but PSUM is 2 MiB)"
                    ),
                )
            )
    return failures


def _accumulator_covers_closed_dims(ir: KernelIR) -> bool:
    """Matmul accumulator must span FULL extent on every output dim at
    position ≥ ``first_acc_position`` in ``dim_order``.

    Store fires once all ACC loops close — at that depth, any dim
    positioned at or below the first ACC dim is out of scope, so the
    store emits the full HBM span on those axes and expects the SBUF
    source to match that span.
    """
    return not _check_accumulator_coverage(ir)


def _check_accumulator_coverage(ir: KernelIR) -> list[ValidityFailure]:
    """List every output axis any reducer accumulator fails to cover.

    Applies to every op in :data:`_REDUCING_KINDS` (matmul, activation
    reduce). Each reducer's accumulator must span FULL extent on every
    output axis at position ≥ the reducer's ``first_acc_position`` in
    ``dim_order`` — otherwise the prologue/closure depth lands inside
    a loop that hasn't yet opened, producing ``i_block_<d>``
    ``NameError`` at render time.

    For matmul, ``first_acc_position`` equals :func:`_store_depth`.
    For activation_reduce (1D output, blocking along F), it equals the
    minimum position of the reducer's own blocking dims.
    """
    failures: list[ValidityFailure] = []
    for op in ir.ops:
        if op.kind not in _REDUCING_KINDS or not op.outputs:
            continue
        """Check the reducer's accumulator buffer. For matmul-style ops
        this is the PSUM sibling buffer; for ``NKIActivationReduce``
        it's the op's SBUF output. Matmul-style ops additionally need
        their SBUF drain target to cover the same closed dims — when
        the drain fires, the SBUF is indexed only on the still-open
        loops, so closed dims must be fully covered by the buffer."""
        acc_buf_names = [_accumulator_buf_name(op)]
        if op.kind in _MATMUL_KINDS:
            acc_buf_names.append(op.outputs[0])
        first_acc_pos = _reducer_first_acc_position(ir, op)
        for acc_buf_name in acc_buf_names:
            if acc_buf_name not in ir.buffer_scopes:
                continue
            buf = ir.physical_buffers[acc_buf_name]
            scope = ir.buffer_scopes[acc_buf_name]
            outer_axis = _outer_axis_in_order(ir, buf.p_axis, buf.f_axis)
            for axis in (buf.p_axis, buf.f_axis):
                if axis is None or axis not in ir.dim_order:
                    continue
                if ir.dim_order.index(axis) < first_acc_pos:
                    continue
                num_tiles = _num_tiles_for_scope(ir, axis, scope, is_outer=(axis == outer_axis))
                full = ir.dimensions[axis].dim_size // ir.dimensions[axis].physical_tile_size
                if num_tiles < full:
                    failures.append(
                        ValidityFailure(
                            check="accumulator_coverage",
                            buffer=acc_buf_name,
                            detail=(
                                f"axis {axis!r} covered by {num_tiles} tiles but needs {full} "
                                f"(position {ir.dim_order.index(axis)} ≥ first_acc_position "
                                f"{first_acc_pos} for {op.kind} → its loop is already closed "
                                f"when the accumulator is first referenced)"
                            ),
                            fix_hint=(
                                f"widen {acc_buf_name} scope to OUTER (or MIDDLE if "
                                f"{axis!r} is outermost), or move {axis!r} before any "
                                f"blocking dim of {op.kind} in dim_order so its loop "
                                f"stays open when the accumulator is referenced"
                            ),
                        )
                    )
    return failures


def _check_reducer_inner_dims(ir: KernelIR) -> list[ValidityFailure]:
    """Every dim inside a reducer's outermost blocking dim must be used
    by the reducer OR have ``num_blocks == 1``.

    The renderer lowers every ``dim_order`` dim to its own ``for
    i_block_<d>`` unconditionally. If a dim ``d`` sits at position
    ≥ ``min(blocking_dim_positions)`` of a reducer but is neither in
    ``blocking_dims`` nor in any dim of the reducer's output buffer,
    the reducer executes ``num_blocks[d]`` extra times per (outer)
    iteration — each call over-writing or over-accumulating the same
    destination slot, producing a multiplied accumulator. When
    ``num_blocks[d] == 1`` the loop runs once and the extra iteration
    doesn't exist.

    A dim is "used" by the reducer if (a) it is in ``blocking_dims``,
    or (b) it is a dim of the reducer's output physical buffer (the
    reducer's destination slot rotates along output dims per block).
    """
    failures: list[ValidityFailure] = []
    for op in ir.ops:
        if op.kind not in _REDUCING_KINDS or not op.outputs:
            continue
        positions = [ir.dim_order.index(d) for d in op.blocking_dims if d in ir.dim_order]
        if not positions:
            continue
        first_blocking = min(positions)
        out_name = op.outputs[0]
        out_dims = set(ir.physical_buffers[out_name].dim_ids) if out_name in ir.physical_buffers else set()
        used_dims = set(op.blocking_dims) | out_dims
        for i, d in enumerate(ir.dim_order):
            if i < first_blocking:
                continue
            if d in used_dims:
                continue
            if ir.num_blocks(d) <= 1:
                continue
            failures.append(
                ValidityFailure(
                    check="reducer_inner_dims",
                    buffer=out_name,
                    detail=(
                        f"dim {d!r} at position {i} in dim_order sits inside {op.kind}'s "
                        f"outermost blocking dim (position {first_blocking}) but is neither "
                        f"a blocking_dim nor an output dim of the reducer — its loop will "
                        f"re-execute the reducer num_blocks[{d!r}] times per outer iteration"
                    ),
                    fix_hint=(
                        f"move {d!r} to a position < {first_blocking} in dim_order so its "
                        f"loop closes before {op.kind} fires"
                    ),
                )
            )
    return failures


def _reducer_first_acc_position(ir: KernelIR, op: Op) -> int:
    """Earliest ``dim_order`` position of ``op``'s blocking dims.

    For matmul this collapses to :func:`_store_depth` (blocking = K,
    which is the single ACCUMULATION dim). For activation_reduce, the
    closure fires when the reducer's own blocking dim's loop closes,
    independent of global store placement.
    """
    positions = [ir.dim_order.index(d) for d in op.blocking_dims if d in ir.dim_order]
    if not positions:
        return len(ir.dim_order)
    return min(positions)


def _transpose_scopes_match(ir: KernelIR) -> bool:
    """Transpose src and dst must share the same ``buffer_scope``.

    ``transpose_block`` / ``dma_transpose_block`` require
    ``src.num_p_tiles == dst.num_f_tiles`` and vice-versa. Since the
    two buffers' axes are swapped, that equality holds iff both sides
    use the same scope-sizing rule on each dim — equivalent to
    matching scope labels.
    """
    return not _check_transpose_scopes(ir)


def _check_transpose_scopes(ir: KernelIR) -> list[ValidityFailure]:
    """List every transpose whose src/dst scopes disagree."""
    failures: list[ValidityFailure] = []
    for op in ir.ops:
        if op.kind not in ("NKITranspose", "NKIDMATranspose"):
            continue
        src = op.inputs.get("data")
        dst = op.outputs[0] if op.outputs else None
        if src is None or dst is None:
            continue
        if src not in ir.buffer_scopes or dst not in ir.buffer_scopes:
            continue
        src_scope = ir.buffer_scopes[src]
        dst_scope = ir.buffer_scopes[dst]
        if src_scope is not dst_scope:
            failures.append(
                ValidityFailure(
                    check="transpose_scope_mismatch",
                    buffer=dst,
                    detail=(
                        f"transpose {src!r}→{dst!r} has src scope {src_scope.name} "
                        f"but dst scope {dst_scope.name}; gadget requires matching "
                        f"num_p_tiles/num_f_tiles across the swapped axes"
                    ),
                    fix_hint=(
                        f"set buffer_scopes[{dst!r}] = buffer_scopes[{src!r}] "
                        f"(or vice-versa) — transpose pairs share one scope"
                    ),
                )
            )
    return failures


def _emission_depth_is_valid(ir: KernelIR) -> bool:
    """Each buffer's ``emission_depth`` must be ≤ its first use depth.

    Otherwise the consumer-side reference (``cur_<buf> = <buf>[...]``)
    emits before the allocation, triggering ``UnboundLocalError`` at
    sim time.
    """
    return not _check_emission_depth(ir)


def _check_emission_depth(ir: KernelIR) -> list[ValidityFailure]:
    """List every buffer whose ``emission_depth`` exceeds its first use."""
    failures: list[ValidityFailure] = []
    for name, depth in ir.emission_depth.items():
        fu = first_use_depth(ir, name)
        if depth > fu:
            failures.append(
                ValidityFailure(
                    check="emission_depth_ceiling",
                    buffer=name,
                    detail=(
                        f"emission_depth={depth} exceeds first_use_depth={fu} "
                        f"(allocation would land below the first reference)"
                    ),
                    fix_hint=(
                        f"lower emission_depth[{name!r}] to ≤ {fu}, or widen the "
                        f"buffer's scope / change dim_order to push first_use_depth deeper"
                    ),
                )
            )
    return failures


def first_use_depth(ir: KernelIR, buf_name: str) -> int:
    """Shallowest depth at which ``buf_name`` may be referenced by render.

    * PSUM accumulator for a matmul: its prologue binding fires at
      ``1 + max(dim_order.index(d))`` over the PSUM buffer's
      block-indexed axes (matches ``_accumulator_prologue_depth`` in
      render).
    * SBUF output of ``NKIMatmul``: first use is the drain depth
      (``min(pos(blocking_dim))``) — the drain binds ``cur_<sbuf_out>``
      before the store can read it.
    * SBUF accumulator for ``NKIActivationReduce`` or
      ``NKIOnlineMatmul``: the memset prologue fires at the
      block-indexed-axes depth (both memset their SBUF output and
      accumulate across a blocking loop).
    * Every other buffer: its producer op's depth.
    """
    if buf_name in ir.physical_buffers and ir.physical_buffers[buf_name].loc == "psum":
        return _accumulator_first_use_depth(ir, buf_name)
    producer = producer_op(ir, buf_name)
    if producer is not None and producer.kind in _MATMUL_KINDS and buf_name in producer.outputs:
        return _matmul_drain_depth(ir, producer)
    if (
        producer is not None
        and producer.kind in ("NKIActivationReduce", "NKIOnlineMatmul")
        and buf_name in producer.outputs
    ):
        return _accumulator_first_use_depth(ir, buf_name)
    if producer is None:
        return len(ir.dim_order)
    return op_depth(ir, producer)


def _matmul_drain_depth(ir: KernelIR, op: Op) -> int:
    """Drain depth for a matmul-style op — ``min(pos(blocking_dim))``.

    Mirrors :func:`nkigym.codegen.render._matmul_drain_depth`. The
    drain fires right after the outermost blocking dim's loop closes,
    turning the PSUM K-accumulator into the SBUF output.
    """
    positions = [ir.dim_order.index(d) for d in op.blocking_dims if d in ir.dim_order]
    if not positions:
        return len(ir.dim_order)
    return min(positions)


def _accumulator_first_use_depth(ir: KernelIR, buf_name: str) -> int:
    """Accumulator prologue depth — matches ``_accumulator_prologue_depth`` in render.

    Rule (same as loads): ``1 + max(dim_order.index(d))`` over the
    accumulator's block-indexed axes — the axes whose ``i_block_<d>``
    appears in the slot expression. OUTER-scoped accumulators have no
    block-indexed axes → depth 0.
    """
    block_axes = _block_indexed_axes(ir, buf_name)
    if not block_axes:
        return 0
    positions = [ir.dim_order.index(d) for d in block_axes if d in ir.dim_order]
    if not positions:
        return 0
    return 1 + max(positions)


def _rotation_axes_in_scope(ir: KernelIR) -> bool:
    """Rotation on an axis requires ``i_block_<axis>`` to be open at the buffer's first use.

    ``num_buffers.num_p_buffers`` / ``num_f_buffers`` emit
    ``[i_block_<axis> % N]`` at the ``cur_<buf>`` binding. If the
    axis's loop hasn't been entered yet, that reference triggers
    ``UnboundLocalError``.
    """
    return not _check_rotation_axes(ir)


def _check_rotation_axes(ir: KernelIR) -> list[ValidityFailure]:
    """List every rotation whose index axis is closed at first use."""
    failures: list[ValidityFailure] = []
    for name, nb in ir.num_buffers.items():
        depth = first_use_depth(ir, name)
        buf = ir.physical_buffers[name]
        if nb.num_p_buffers is not None and not axis_open(ir, buf.p_axis, depth):
            failures.append(
                ValidityFailure(
                    check="rotation_axis_closed",
                    buffer=name,
                    detail=(
                        f"num_p_buffers={nb.num_p_buffers} on p_axis={buf.p_axis!r} "
                        f"but that axis is not open at first_use_depth={depth} "
                        f"(position {_axis_position(ir, buf.p_axis)} in dim_order)"
                    ),
                    fix_hint=(
                        f"set num_buffers[{name!r}].num_p_buffers=None, or move "
                        f"{buf.p_axis!r} above position {depth} in dim_order so its "
                        f"loop is open when {name!r} is first referenced"
                    ),
                )
            )
        if nb.num_f_buffers is not None and buf.f_axis is not None and not axis_open(ir, buf.f_axis, depth):
            failures.append(
                ValidityFailure(
                    check="rotation_axis_closed",
                    buffer=name,
                    detail=(
                        f"num_f_buffers={nb.num_f_buffers} on f_axis={buf.f_axis!r} "
                        f"but that axis is not open at first_use_depth={depth} "
                        f"(position {_axis_position(ir, buf.f_axis)} in dim_order)"
                    ),
                    fix_hint=(
                        f"set num_buffers[{name!r}].num_f_buffers=None, or move "
                        f"{buf.f_axis!r} above position {depth} in dim_order so its "
                        f"loop is open when {name!r} is first referenced"
                    ),
                )
            )
    return failures


def _axis_position(ir: KernelIR, axis: str | None) -> int | str:
    """Position of ``axis`` in ``dim_order``, or ``'absent'`` when missing."""
    if axis is None or axis not in ir.dim_order:
        return "absent"
    return ir.dim_order.index(axis)


def axis_open(ir: KernelIR, axis: str, depth: int) -> bool:
    """``i_block_<axis>`` is open iff the axis sits above ``depth`` in ``dim_order``."""
    if axis not in ir.dim_order:
        return False
    return ir.dim_order.index(axis) < depth


def producer_op(ir: KernelIR, buf_name: str) -> Op | None:
    """The single op that writes ``buf_name`` as an output, or ``None``."""
    for op in ir.ops:
        if buf_name in op.outputs:
            return op
    return None


def op_depth(ir: KernelIR, op: Op) -> int:
    """Loop-nest depth at which ``op`` is emitted in ``render_ir``.

    For compute ops this matches :func:`nkigym.codegen.render._phase_needs_dim`:
    the op's emission depth is one past the deepest ``dim_order``
    position of any dim it needs (in ``blocking_dims`` or as a
    block-indexed axis on any of its buffers). Elided dims that the
    op does not reference do not push the op deeper — if the render
    skips their loop, ``first_use_depth`` must skip it too or
    ``emission_depth`` ceilings open a hole where allocations fall
    outside every emitted loop.
    """
    if op.kind in _LOAD_KINDS or (op.kind == "NKIDMATranspose" and _is_hbm_sourced(ir, op)):
        return _load_depth(ir, op)
    if op.kind == "NKIStore":
        return _store_depth(ir)
    return _compute_op_depth(ir, op)


def _compute_op_depth(ir: KernelIR, op: Op) -> int:
    """Deepest ``dim_order`` position used by ``op``, plus one.

    A compute op uses a dim iff (a) it is in ``op.blocking_dims`` or
    (b) any of its input or output buffers has more than
    ``ltiles_per_block[dim]`` tiles along that dim (the same rule
    :func:`nkigym.codegen.render._phase_needs_dim` uses to decide
    whether to open a loop for the op).
    """
    used_dims: set[str] = set(op.blocking_dims)
    for name in list(op.inputs.values()) + list(op.outputs):
        if name not in ir.physical_buffers:
            continue
        buf = ir.physical_buffers[name]
        scope = ir.buffer_scopes.get(name, BufferScope.INNER)
        outer_axis = _outer_axis_in_order(ir, buf.p_axis, buf.f_axis)
        for axis in (buf.p_axis, buf.f_axis):
            if axis is None:
                continue
            num_tiles = _num_tiles_for_scope(ir, axis, scope, is_outer=(axis == outer_axis))
            ltile = ir.ltiles_per_block.get(axis, 1)
            if num_tiles > ltile:
                used_dims.add(axis)
    positions = [ir.dim_order.index(d) for d in used_dims if d in ir.dim_order]
    if not positions:
        return 0
    return 1 + max(positions)


_LOAD_KINDS = frozenset({"NKILoad"})


def _is_hbm_sourced(ir: KernelIR, op: Op) -> bool:
    """``NKIDMATranspose`` whose ``data`` is a kernel parameter (HBM)."""
    return op.inputs.get("data") in ir.param_names


def _load_depth(ir: KernelIR, op: Op) -> int:
    """Load fires inside every block-loop whose index appears in its destination slice."""
    dst = op.outputs[0]
    block_axes = _block_indexed_axes(ir, dst)
    if not block_axes:
        return 0
    positions = [ir.dim_order.index(a) for a in block_axes if a in ir.dim_order]
    if not positions:
        return 0
    return max(positions) + 1


def _store_depth(ir: KernelIR) -> int:
    """Store fires at the first ACCUMULATION position in ``dim_order``."""
    for i, d in enumerate(ir.dim_order):
        if ir.dimensions[d].role is DimRole.ACCUMULATION:
            return i
    return len(ir.dim_order)


def _block_indexed_axes(ir: KernelIR, buf_name: str) -> list[str]:
    """Axes whose per-block width is < full dim — i.e. the buffer is sliced along them."""
    buf = ir.physical_buffers[buf_name]
    scope = ir.buffer_scopes.get(buf_name, BufferScope.INNER)
    outer_axis = _outer_axis_in_order(ir, buf.p_axis, buf.f_axis)
    axes: list[str] = []
    for axis in (buf.p_axis, buf.f_axis):
        if axis is None:
            continue
        num_tiles = _num_tiles_for_scope(ir, axis, scope, is_outer=(axis == outer_axis))
        if num_tiles * ir.dimensions[axis].physical_tile_size < ir.dimensions[axis].dim_size:
            axes.append(axis)
    return axes


def _num_tiles_for_scope(ir: KernelIR, axis: str, scope: BufferScope, is_outer: bool) -> int:
    """Tile count along ``axis`` under ``scope`` — matches ``_scope_extents`` in render."""
    info = ir.dimensions[axis]
    ltiles = ir.ltiles_per_block[axis]
    full = info.dim_size // info.physical_tile_size
    if scope is BufferScope.INNER:
        return ltiles
    if scope is BufferScope.OUTER:
        return full
    return ltiles if is_outer else full


def _outer_axis_in_order(ir: KernelIR, p_axis: str, f_axis: str | None) -> str:
    """Whichever of ``(p_axis, f_axis)`` appears first in ``dim_order``."""
    if f_axis is None:
        return p_axis
    return p_axis if ir.dim_order.index(p_axis) < ir.dim_order.index(f_axis) else f_axis
