"""Random sampler over the KernelIR knob space (post-2N refactor).

Samples one knob at a time; a final :func:`validate.is_valid` filter
rejects infeasible draws. The knob space:

1. ``loop_order`` — random permutation of the ``{d}.block``/``{d}.tile``
   entries that preserves ``{d}.block`` before ``{d}.tile`` per dim.
2. ``ltiles_per_block`` — random divisor per dim (subject to a PSUM-
   banks cap that keeps matmul's inner tile count ≤ 8).
3. ``buffer_scopes`` — random ``DimScope`` per-dim per-buffer, subject
   to loop-order feasibility, accumulator-coverage rules, and
   SBUF/PSUM byte cap.
"""

import itertools
import random
from dataclasses import replace

from nkigym.kernel_ir.ir import DimScope, KernelIR
from nkigym.kernel_ir.validate import is_valid

_SBUF_CAP_BYTES = 24 * 1024 * 1024
_PSUM_CAP_BYTES = 2 * 1024 * 1024
_PSUM_TILE_CAP = 8
_SAMPLE_RETRIES = 100

_DTYPE_BYTES = {"bfloat16": 2, "bf16": 2, "float16": 2, "fp16": 2, "float32": 4, "fp32": 4}

_MATMUL_KINDS = frozenset({"NKIMatmul"})


def sample(ir: KernelIR, rng: random.Random | None = None) -> KernelIR:
    """Return a new ``KernelIR`` with every tunable knob randomized."""
    rng = rng or random.Random()
    last_err: str | None = None
    for _ in range(_SAMPLE_RETRIES):
        try:
            candidate = replace(
                ir, loop_order=sample_loop_order(ir, rng), ltiles_per_block=sample_ltiles_per_block(ir, rng)
            )
            candidate = replace(candidate, buffer_scopes=sample_buffer_scopes(candidate, rng))
            if is_valid(candidate):
                return candidate
        except RuntimeError as e:
            last_err = str(e)
    raise RuntimeError(f"sample: no feasible IR within {_SAMPLE_RETRIES} retries; last error: {last_err}")


def sample_loop_order(ir: KernelIR, rng: random.Random) -> list[str]:
    """Random permutation of 2N entries preserving the per-dim
    ``{d}.block`` before ``{d}.tile`` invariant."""
    dims = list(ir.dimensions)
    entries = [f"{d}.block" for d in dims] + [f"{d}.tile" for d in dims]
    for _ in range(200):
        candidate = list(entries)
        rng.shuffle(candidate)
        ok = True
        for d in dims:
            if candidate.index(f"{d}.block") > candidate.index(f"{d}.tile"):
                ok = False
                break
        if ok:
            return candidate
    raise RuntimeError("sample_loop_order: no valid permutation after 200 tries")


def sample_ltiles_per_block(ir: KernelIR, rng: random.Random) -> dict[str, int]:
    """Per-dim random divisor of ``num_ltile`` with a PSUM-banks cap."""
    result = {d: rng.choice(_divisors(ir.num_ltile(d))) for d in ir.dimensions}
    pairs = _psum_gadget_dim_pairs(ir)
    for _ in range(len(pairs) + 1):
        if all(result[a] * result[b] <= _PSUM_TILE_CAP for a, b in pairs):
            return result
        for a, b in pairs:
            if result[a] * result[b] <= _PSUM_TILE_CAP:
                continue
            a_choices = _divisors(ir.num_ltile(a))
            b_choices = _divisors(ir.num_ltile(b))
            fits = [(x, y) for x in a_choices for y in b_choices if x * y <= _PSUM_TILE_CAP]
            result[a], result[b] = rng.choice(fits)
    raise RuntimeError(f"sample_ltiles_per_block: could not satisfy PSUM caps jointly for pairs {pairs}")


def _psum_gadget_dim_pairs(ir: KernelIR) -> list[tuple[str, str]]:
    """Inner-loop dim pairs for every PSUM-using gadget in ``ir``."""
    pairs: list[tuple[str, str]] = []
    for op in ir.ops:
        if op.kind in _MATMUL_KINDS:
            pairs.append((op.axis_map["M"], op.axis_map["N"]))
    return pairs


def sample_buffer_scopes(ir: KernelIR, rng: random.Random) -> dict[str, dict[str, DimScope]]:
    """Random per-dim ``DimScope`` per buffer, subject to loop-order
    feasibility, accumulator coverage, and byte cap."""
    matmul_sbuf_outputs = _matmul_sbuf_outputs(ir)
    reducing_by_buf = _reducing_dims_per_buffer(ir)
    scopes: dict[str, dict[str, DimScope]] = {}
    for name, buf in ir.physical_buffers.items():
        if buf.loc == "hbm":
            continue
        existing = ir.buffer_scopes.get(name, {})
        dims = list(existing.keys()) if existing else list(buf.dim_ids)
        base_size = _dtype_bytes(buf.dtype) * buf.tile[0] * buf.tile[1]
        cap = _PSUM_CAP_BYTES if buf.loc == "psum" else _SBUF_CAP_BYTES
        reducing = reducing_by_buf.get(name, set())
        candidates = _enumerate_feasible_scopes(
            ir,
            dims,
            base_size,
            cap,
            is_matmul_sbuf=name in matmul_sbuf_outputs,
            is_psum=buf.loc == "psum",
            reducing=reducing,
            storage_dims=set(buf.dim_ids),
            matmul_blocking=reducing,
        )
        if not candidates:
            raise RuntimeError(
                f"sample_buffer_scopes: buffer {name!r} has no feasible per-dim scope combination "
                f"under cap {cap} and current loop_order"
            )
        scopes[name] = rng.choice(candidates)
    return scopes


def _matmul_sbuf_outputs(ir: KernelIR) -> set[str]:
    """SBUF output buffer names of every matmul op."""
    return {op.outputs[0] for op in ir.ops if op.kind in _MATMUL_KINDS and op.outputs}


def _reducing_dims_per_buffer(ir: KernelIR) -> dict[str, set[str]]:
    """Map buffer name → reducing dims of its producing matmul (SBUF)
    or the matmul whose PSUM sibling it is."""
    result: dict[str, set[str]] = {}
    for op in ir.ops:
        if op.kind not in _MATMUL_KINDS or not op.outputs:
            continue
        sbuf_out = op.outputs[0]
        result[sbuf_out] = set(op.blocking_dims)
        psum_name = "psum_" + sbuf_out[len("sbuf_") :]
        result[psum_name] = set(op.blocking_dims)
    return result


def _enumerate_feasible_scopes(
    ir: KernelIR,
    dims: list[str],
    base_tile_bytes: int,
    cap: int,
    *,
    is_matmul_sbuf: bool,
    is_psum: bool,
    reducing: set[str],
    storage_dims: set[str],
    matmul_blocking: set[str],
) -> list[dict[str, DimScope]]:
    """Feasible per-dim scope assignments under all constraints.

    Filters:

    1. Byte size ≤ ``cap`` (storage dims only contribute).
    2. Loop-order feasibility: ``lower <= upper`` for emission depth
       across every dim in ``scope_map`` + PSUM-outside-K clamp.
    3. Matmul SBUF accumulator: non-reducing dim constraint position
       must be < first reducing-block position (else the buffer
       re-allocates inside K).
    """
    result: list[dict[str, DimScope]] = []
    first_red_pos = _first_reducing_block_pos(ir, matmul_blocking)
    for combo in itertools.product(DimScope, repeat=len(dims)):
        scope_map = dict(zip(dims, combo))
        if not _scope_size_ok(ir, scope_map, storage_dims, base_tile_bytes, cap):
            continue
        if not _scope_loop_order_ok(ir, scope_map, is_psum=is_psum, reducing=reducing):
            continue
        if is_matmul_sbuf and first_red_pos is not None:
            if not _scope_accumulator_coverage_ok(ir, scope_map, matmul_blocking, first_red_pos):
                continue
        result.append(scope_map)
    return result


def _scope_size_ok(
    ir: KernelIR, scope_map: dict[str, DimScope], storage_dims: set[str], base_tile_bytes: int, cap: int
) -> bool:
    """Byte-size check — only dims in ``storage_dims`` scale the size."""
    size = base_tile_bytes
    for d, scope in scope_map.items():
        if d in storage_dims:
            size *= _tiles_for_scope(ir, d, scope)
    return size <= cap


def _scope_loop_order_ok(ir: KernelIR, scope_map: dict[str, DimScope], *, is_psum: bool, reducing: set[str]) -> bool:
    """Emission-depth feasibility check — mirrors render's
    ``_buffer_emission_depth`` invariant (``lower <= upper``).

    PSUM reducing dim may be ``PER_BLOCK`` (Option B — greedy drain
    after K.tile closes) or ``FULL`` (Option A — single drain after
    K.block closes). ``PER_TILE`` on a reducing dim is still invalid
    (emission depth would land inside K.tile — re-memset wipes the
    accumulation mid-reduction).
    """
    if is_psum:
        for d in reducing:
            if scope_map.get(d) is DimScope.PER_TILE:
                return False
    lower = 0
    upper = len(ir.loop_order)
    for d, scope in scope_map.items():
        block_pos = ir.loop_order.index(f"{d}.block") + 1
        tile_pos = ir.loop_order.index(f"{d}.tile") + 1
        if scope is DimScope.FULL:
            upper = min(upper, block_pos - 1)
        elif scope is DimScope.PER_BLOCK:
            lower = max(lower, block_pos)
        elif scope is DimScope.PER_TILE:
            lower = max(lower, tile_pos)
    if is_psum:
        for d in reducing:
            tile_entry = f"{d}.tile"
            if tile_entry in ir.loop_order:
                upper = min(upper, ir.loop_order.index(tile_entry))
    return upper >= lower


def _first_reducing_block_pos(ir: KernelIR, blocking: set[str]) -> int | None:
    """Earliest ``.block`` position among the matmul's blocking dims."""
    positions = [ir.loop_order.index(f"{d}.block") for d in blocking if f"{d}.block" in ir.loop_order]
    if not positions:
        return None
    return min(positions)


def _scope_accumulator_coverage_ok(
    ir: KernelIR, scope_map: dict[str, DimScope], blocking: set[str], first_red_pos: int
) -> bool:
    """Non-reducing dim constraint must land strictly before the first
    reducing-block loop."""
    for d, scope in scope_map.items():
        if d in blocking:
            continue
        if scope is DimScope.FULL:
            continue
        if scope is DimScope.PER_BLOCK:
            if ir.loop_order.index(f"{d}.block") >= first_red_pos:
                return False
        else:
            if ir.loop_order.index(f"{d}.tile") >= first_red_pos:
                return False
    return True


def _tiles_for_scope(ir: KernelIR, dim: str, scope: DimScope) -> int:
    if scope is DimScope.PER_TILE:
        return 1
    if scope is DimScope.PER_BLOCK:
        return ir.ltiles_per_block[dim]
    return ir.num_ltile(dim)


def _dtype_bytes(dtype: str) -> int:
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"Unknown dtype: {dtype!r}")
    return _DTYPE_BYTES[dtype]


def knob_signature(ir: KernelIR) -> tuple:
    """Hashable fingerprint of every tunable knob — for deduping samples."""
    buf_sig = tuple(
        (name, tuple(sorted((d, s.value) for d, s in smap.items()))) for name, smap in sorted(ir.buffer_scopes.items())
    )
    return (tuple(ir.loop_order), tuple(sorted(ir.ltiles_per_block.items())), buf_sig)


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
