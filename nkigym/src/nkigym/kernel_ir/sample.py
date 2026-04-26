"""Random sampler over the full KernelIR knob space.

Samples are built one knob at a time in dependency order, each knob
conditioned on the partial IR so far. No rejection loop — every draw
satisfies :func:`validate.is_valid` by construction.

Order:

1. ``dim_order`` — free permutation.
2. ``ltiles_per_block`` — free divisor per dim.
3. ``buffer_scopes`` — free per buffer.
4. ``emission_depth`` + ``num_buffers`` — joint pass, largest-base
   buffer first. Each draw picks a valid depth with room for the
   base, then a rotation pair that fits the remaining SBUF budget.
"""

import random
from dataclasses import replace

from nkigym.kernel_ir.ir import BufferScope, KernelIR, NumBuffers
from nkigym.kernel_ir.validate import (
    _accumulator_covers_closed_dims,
    _find_matmul_op,
    _num_tiles_for_scope,
    _outer_axis_in_order,
    axis_open,
    first_use_depth,
)

_SBUF_CAP_BYTES = 24 * 1024 * 1024
"""Trn2 SBUF size. Per-allocation AND per-emission-depth cap.

Each ``allocate_buffers`` call must fit within this. The SUM of all
buffers emitted at the same ``emission_depth`` must also fit — the
depth-0 group is live across the whole kernel and every deeper group
co-exists with its ancestors within one iteration of the enclosing
loops."""

_NUM_BUFFERS_CAP = 4
"""Per-axis ceiling on ``NumBuffers.num_p_buffers`` / ``num_f_buffers``.

Deeper multi-buffering mostly chews SBUF without hiding additional
DMA latency; 4 slots is enough to overlap compute+load for most
kernels we care about."""

_PSUM_TILE_CAP = 8
"""Ceiling on gadget PSUM tile count. PSUM has 8 banks per partition.

Gadgets (``matmul_block``, ``transpose_block``) allocate fresh PSUM
tiles inside their inner loops. With
``--enable-instruction-scheduling=false`` (always on for profiling
runs), banks can't rotate across iterations, so the inner-loop
product must fit in 8 banks.

Enforced in ``sample_ltiles_per_block``:

* Matmul: ``ltiles_per_block[m_dim] × ltiles_per_block[n_dim] ≤ 8``.
* Transpose: ``ltiles_per_block[p_dim] × ltiles_per_block[f_dim] ≤ 8``
  (both gadgets share the innermost body, so they add up).
"""

_SAMPLE_RETRIES = 100
"""Max attempts for a single :func:`sample` call.

Budget-aware knobs can raise when upstream choices produce infeasible
buffers (e.g. three OUTER-scoped 8 MiB buffers with no depth headroom).
A fresh redraw usually succeeds; ``_SAMPLE_RETRIES`` bounds the retry
loop before giving up."""

_DTYPE_BYTES = {"bfloat16": 2, "bf16": 2, "float16": 2, "fp16": 2, "float32": 4, "fp32": 4}


def sample(ir: KernelIR, rng: random.Random | None = None) -> KernelIR:
    """Return a new ``KernelIR`` with every tunable knob randomized.

    Retries the full draw up to ``_SAMPLE_RETRIES`` times when budget-
    aware stages raise ``RuntimeError`` (infeasible co-residence).

    Args:
        ir: Source IR — typically the canonical-default output of
            ``build_ir``. Non-tunable fields (``dimensions``,
            ``logical_tensors``, ``physical_buffers``, ``ops``,
            ``edges``) pass through unchanged.
        rng: Optional ``random.Random`` for deterministic runs.

    Returns:
        A valid ``KernelIR`` with randomized ``dim_order``,
        ``ltiles_per_block``, ``buffer_scopes``, ``num_buffers``, and
        ``emission_depth``.
    """
    rng = rng or random.Random()
    last_err: RuntimeError | None = None
    for _ in range(_SAMPLE_RETRIES):
        try:
            partial = replace(
                ir, dim_order=sample_dim_order(ir, rng), ltiles_per_block=sample_ltiles_per_block(ir, rng)
            )
            partial = replace(partial, buffer_scopes=sample_buffer_scopes(partial, rng))
            emission_depth, num_buffers = sample_emission_depth_and_num_buffers(partial, rng)
            return replace(partial, emission_depth=emission_depth, num_buffers=num_buffers)
        except RuntimeError as e:
            last_err = e
    raise RuntimeError(f"sample: no feasible IR within {_SAMPLE_RETRIES} retries; last error: {last_err}")


def sample_dim_order(ir: KernelIR, rng: random.Random) -> list[str]:
    """One random permutation of ``ir.dimensions`` that passes
    :func:`validate._check_reducer_inner_dims`.

    A reducer's blocking dim must have every inner dim of ``dim_order``
    (positions ≥ its own) be either a blocking_dim or an output dim of
    that reducer. Otherwise the mechanical renderer opens a loop for
    an unused inner dim and the reducer re-accumulates on every step.

    We filter by enumerating all ``N!`` permutations and keeping the
    valid ones — cheap for ``N ≤ 5``, and rmsnorm/matmul sit at ``N=3``.
    """
    dims = list(ir.dimensions)
    import itertools

    candidates = [list(p) for p in itertools.permutations(dims)]
    valid = [p for p in candidates if _dim_order_respects_reducer_inner_dims(ir, p)]
    if not valid:
        raise RuntimeError("sample_dim_order: no permutation satisfies reducer_inner_dims invariant")
    return rng.choice(valid)


def _dim_order_respects_reducer_inner_dims(ir: KernelIR, dim_order: list[str]) -> bool:
    """True iff every reducer's inner dims are all used by the reducer.

    Mirrors :func:`validate._check_reducer_inner_dims` but takes the
    candidate ``dim_order`` directly so we can filter permutations
    before committing to one.
    """
    for op in ir.ops:
        if op.kind not in _REDUCER_KINDS or not op.outputs:
            continue
        positions = [dim_order.index(d) for d in op.blocking_dims if d in dim_order]
        if not positions:
            continue
        first_blocking = min(positions)
        out_name = op.outputs[0]
        out_dims = set(ir.physical_buffers[out_name].dim_ids) if out_name in ir.physical_buffers else set()
        used = set(op.blocking_dims) | out_dims
        for i, d in enumerate(dim_order):
            if i < first_blocking:
                continue
            if d not in used:
                return False
    return True


_REDUCER_KINDS = frozenset({"NKIMatmul", "NKIActivationReduce"})
"""Must match :data:`nkigym.codegen.render._REDUCING_KINDS` — keeping
this local so the sampler has no import dependency on render."""


def sample_ltiles_per_block(ir: KernelIR, rng: random.Random) -> dict[str, int]:
    """Per-dim random divisor of ``num_ltile`` along that dim.

    PSUM cap: for every PSUM-using gadget call, the product of
    ``ltiles_per_block`` on its inner-loop dims must be ≤
    ``_PSUM_TILE_CAP``. Matmul uses (M, N); transpose uses (P, F).
    All caps are enforced jointly — pairs may share a dim, so the
    constraint space is solved iteratively until every pair fits.
    """
    result = {d: rng.choice(_divisors(_num_ltile(ir, d))) for d in ir.dimensions}
    pairs = _psum_gadget_dim_pairs(ir)
    for _ in range(len(pairs) + 1):
        if all(result[a] * result[b] <= _PSUM_TILE_CAP for a, b in pairs):
            return result
        for a, b in pairs:
            if result[a] * result[b] <= _PSUM_TILE_CAP:
                continue
            a_choices = _divisors(_num_ltile(ir, a))
            b_choices = _divisors(_num_ltile(ir, b))
            fits = [(x, y) for x in a_choices for y in b_choices if x * y <= _PSUM_TILE_CAP]
            result[a], result[b] = rng.choice(fits)
    raise RuntimeError(
        f"sample_ltiles_per_block: could not satisfy PSUM caps jointly for pairs {pairs}; " f"last: {result}"
    )


def _psum_gadget_dim_pairs(ir: KernelIR) -> list[tuple[str, str]]:
    """Inner-loop dim pairs for every PSUM-using gadget in ``ir``."""
    pairs: list[tuple[str, str]] = []
    for op in ir.ops:
        if op.kind == "NKIMatmul":
            pairs.append((op.axis_map["M"], op.axis_map["N"]))
        elif op.kind == "NKITranspose":
            pairs.append((op.axis_map["P"], op.axis_map["F"]))
    return pairs


def sample_buffer_scopes(ir: KernelIR, rng: random.Random) -> dict[str, BufferScope]:
    """Per-buffer random choice from ``{INNER, MIDDLE, OUTER}``.

    Three constraints bias the draw:

    * Transpose src/dst pairs share one scope — the transpose gadget
      requires matching tile counts on the swapped axes.
    * The matmul accumulator must span FULL extent on every output dim
      at position ≥ ``first_acc_position`` in ``dim_order`` (see
      :func:`validate._accumulator_covers_closed_dims`). Scopes that
      would under-cover a closed output dim are dropped from its
      choice list.
    * Scopes whose base allocation exceeds ``_SBUF_CAP_BYTES`` are
      dropped — a single ``allocate_buffers`` call cannot be larger
      than SBUF even with ``num_buffers=None``.
    """
    acc_buf = _matmul_output_name(ir)
    result: dict[str, BufferScope] = {}
    for name in _tunable_buffers(ir):
        acc_filtered = _accumulator_scope_choices(ir, name) if name == acc_buf else list(BufferScope)
        choices = [s for s in acc_filtered if _base_size_for_scope(ir, name, s) <= _SBUF_CAP_BYTES]
        if not choices:
            raise RuntimeError(
                f"sample_buffer_scopes: buffer {name} has no scope whose base fits "
                f"_SBUF_CAP_BYTES={_SBUF_CAP_BYTES}; upstream dim_order/ltiles_per_block "
                f"produced an infeasible buffer"
            )
        result[name] = rng.choice(choices)
    for op in ir.ops:
        if op.kind not in ("NKITranspose", "NKIDMATranspose"):
            continue
        src = op.inputs.get("data")
        dst = op.outputs[0] if op.outputs else None
        if src in result and dst in result:
            result[dst] = result[src]
    return result


def _base_size_for_scope(ir: KernelIR, name: str, scope: BufferScope) -> int:
    """Bytes of one ``allocate_buffers`` call if ``name`` were sized by ``scope``."""
    trial = replace(ir, buffer_scopes={**ir.buffer_scopes, name: scope})
    return _buffer_base_size(trial, name)


def _matmul_output_name(ir: KernelIR) -> str | None:
    """SBUF name that holds the matmul accumulator, or ``None``."""
    op = _find_matmul_op(ir)
    return op.outputs[0] if op and op.outputs else None


def _accumulator_scope_choices(ir: KernelIR, name: str) -> list[BufferScope]:
    """Scopes under which the accumulator covers every closed output dim."""
    valid: list[BufferScope] = []
    for scope in BufferScope:
        trial = replace(ir, buffer_scopes={**ir.buffer_scopes, name: scope})
        if _accumulator_covers_closed_dims(trial):
            valid.append(scope)
    return valid


def sample_emission_depth_and_num_buffers(
    ir: KernelIR, rng: random.Random
) -> tuple[dict[str, int], dict[str, NumBuffers]]:
    """Joint draw — every buffer gets a depth + rotation in one pass.

    Largest-base buffers claim budget first. For each buffer:

    * ``emission_depth`` is drawn from ``[0, first_use_depth]``
      restricted to depths where the buffer's base size still fits in
      remaining SBUF budget.
    * ``num_buffers`` is drawn from the ``(None | divisors(num_ltile))``
      product for each axis (rotation only when the axis's loop is
      open at first-use depth), restricted to pairs whose multiplied
      size fits the remaining budget at the chosen depth.

    Raises ``RuntimeError`` when no depth has room for a base — that
    means upstream knobs produced buffers that cannot co-exist in
    SBUF regardless of ``num_buffers``.
    """
    bases = {name: _buffer_base_size(ir, name) for name in _tunable_buffers(ir)}
    used: dict[int, int] = {}
    emission: dict[str, int] = {}
    num_bufs: dict[str, NumBuffers] = {}
    for name, base in sorted(bases.items(), key=lambda kv: -kv[1]):
        buf = ir.physical_buffers[name]
        fu_depth = first_use_depth(ir, name)
        depth_choices = [d for d in range(fu_depth + 1) if used.get(d, 0) + base <= _SBUF_CAP_BYTES]
        if not depth_choices:
            raise RuntimeError(
                f"sample: buffer {name} (base {base} B) cannot fit at any depth ≤ "
                f"{fu_depth} under cap {_SBUF_CAP_BYTES} — co-resident bases exceed SBUF"
            )
        depth = rng.choice(depth_choices)
        remaining = _SBUF_CAP_BYTES - used.get(depth, 0)
        p_raw: list[int | None] = [None]
        if axis_open(ir, buf.p_axis, fu_depth):
            p_raw.extend(d for d in _divisors(_num_ltile(ir, buf.p_axis)) if d <= _NUM_BUFFERS_CAP)
        f_raw: list[int | None] = [None]
        if buf.f_axis is not None and axis_open(ir, buf.f_axis, fu_depth):
            f_raw.extend(d for d in _divisors(_num_ltile(ir, buf.f_axis)) if d <= _NUM_BUFFERS_CAP)
        pairs = [(p, f) for p in p_raw for f in f_raw if base * (p or 1) * (f or 1) <= remaining]
        p_choice, f_choice = rng.choice(pairs)
        used[depth] = used.get(depth, 0) + base * (p_choice or 1) * (f_choice or 1)
        emission[name] = depth
        num_bufs[name] = NumBuffers(num_p_buffers=p_choice, num_f_buffers=f_choice)
    return emission, num_bufs


def _buffer_base_size(ir: KernelIR, name: str) -> int:
    """Bytes occupied by one ``allocate_buffers`` call with ``num_*_buffers=None``."""
    buf = ir.physical_buffers[name]
    p_tile, f_tile = buf.tile
    n_p, n_f = _scope_tile_counts(ir, name)
    return p_tile * f_tile * n_p * n_f * _DTYPE_BYTES[buf.dtype]


def _scope_tile_counts(ir: KernelIR, name: str) -> tuple[int, int]:
    """``(num_p_tiles, num_f_tiles)`` under the buffer's current scope."""
    buf = ir.physical_buffers[name]
    scope = ir.buffer_scopes[name]
    outer_axis = _outer_axis_in_order(ir, buf.p_axis, buf.f_axis)
    n_p = _num_tiles_for_scope(ir, buf.p_axis, scope, is_outer=(buf.p_axis == outer_axis))
    n_f = 1
    if buf.f_axis is not None:
        n_f = _num_tiles_for_scope(ir, buf.f_axis, scope, is_outer=(buf.f_axis == outer_axis))
    return n_p, n_f


def knob_signature(ir: KernelIR) -> tuple:
    """Hashable fingerprint of every tunable knob — for deduping samples."""
    return (
        tuple(ir.dim_order),
        tuple(sorted(ir.ltiles_per_block.items())),
        tuple(sorted((k, v.value) for k, v in ir.buffer_scopes.items())),
        tuple(sorted((k, nb.num_p_buffers, nb.num_f_buffers) for k, nb in ir.num_buffers.items())),
        tuple(sorted(ir.emission_depth.items())),
    )


def _tunable_buffers(ir: KernelIR) -> list[str]:
    """Non-HBM physical buffer names — the ones carrying tunable knobs."""
    return [n for n in ir.physical_buffers if not n.startswith("hbm_")]


def _num_ltile(ir: KernelIR, dim_id: str) -> int:
    """Logical-tile count along ``dim_id`` — ``dim_size // logical_tile_size``."""
    info = ir.dimensions[dim_id]
    return info.dim_size // info.logical_tile_size


def _divisors(n: int) -> list[int]:
    """All positive divisors of ``n``, ascending."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    small: list[int] = []
    large: list[int] = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            small.append(i)
            if i * i != n:
                large.append(n // i)
        i += 1
    return small + large[::-1]
