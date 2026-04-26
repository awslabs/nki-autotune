"""Lightweight validity checks over a ``KernelIR``'s knob assignment.

These are cheap, mechanical guards meant to filter random samples
before rendering — no rendering, no exceptions. Two entry points:

* :func:`is_valid` — boolean fast-path for sampler reject loops.
* :func:`validity_report` — structured list of every failed check
  with a fix hint.

Post-2N-refactor invariants:

* ``loop_order`` has ``2 × len(dimensions)`` entries — ``{d}.block``
  and ``{d}.tile`` per dim, with ``{d}.block`` always preceding
  ``{d}.tile``.
* ``buffer_scopes[name]`` is a per-dim map ``{dim: DimScope}``.
* Every buffer has a feasible emission depth under the current
  ``loop_order`` / ``buffer_scopes``.
* Matmul SBUF accumulator non-reducing dims that sit after the
  reducing dim's ``.block`` must be FULL (else the buffer
  re-allocates inside K and accumulation breaks).
* SBUF accumulator emission depth ≤ PSUM (drain) emission depth.
"""

from dataclasses import dataclass

from nkigym.kernel_ir.ir import DimScope, KernelIR

_MATMUL_KINDS = frozenset({"NKIMatmul"})


@dataclass(frozen=True)
class ValidityFailure:
    """One failed validity check, plus a concrete fix hint.

    Attributes:
        check: Identifier of the failed check.
        buffer: Buffer name the failure pertains to, or ``None`` when
            whole-IR.
        detail: Human-readable description.
        fix_hint: Actionable suggestion.
    """

    check: str
    buffer: str | None
    detail: str
    fix_hint: str


def is_valid(ir: KernelIR) -> bool:
    """Return ``True`` iff every validity check passes."""
    return not validity_report(ir)


def validity_report(ir: KernelIR) -> list[ValidityFailure]:
    """Return every failed validity check on ``ir``."""
    failures: list[ValidityFailure] = []
    failures.extend(_check_loop_order_pairs(ir))
    failures.extend(_check_buffer_scope_axes(ir))
    failures.extend(_check_feasible_emission_depth(ir))
    failures.extend(_check_accumulator_coverage(ir))
    failures.extend(_check_sbuf_outlives_drain(ir))
    return failures


def _check_loop_order_pairs(ir: KernelIR) -> list[ValidityFailure]:
    """``loop_order`` must carry 2N entries with ``{d}.block`` preceding
    ``{d}.tile`` for every dim."""
    failures: list[ValidityFailure] = []
    n = len(ir.dimensions)
    if len(ir.loop_order) != 2 * n:
        failures.append(
            ValidityFailure(
                check="loop_order_length",
                buffer=None,
                detail=f"loop_order has {len(ir.loop_order)} entries, expected 2*{n} = {2 * n}",
                fix_hint="loop_order must carry one {d}.block and one {d}.tile per dim",
            )
        )
        return failures
    for d in ir.dimensions:
        block = f"{d}.block"
        tile = f"{d}.tile"
        if block not in ir.loop_order or tile not in ir.loop_order:
            failures.append(
                ValidityFailure(
                    check="loop_order_missing",
                    buffer=None,
                    detail=f"loop_order missing {block!r} or {tile!r}",
                    fix_hint=f"add {block!r} and {tile!r} to loop_order",
                )
            )
            continue
        if ir.loop_order.index(block) >= ir.loop_order.index(tile):
            failures.append(
                ValidityFailure(
                    check="loop_order_block_before_tile",
                    buffer=None,
                    detail=f"{block!r} at position {ir.loop_order.index(block)} "
                    f">= {tile!r} at position {ir.loop_order.index(tile)}",
                    fix_hint=f"swap so {block!r} precedes {tile!r}",
                )
            )
    return failures


def _check_buffer_scope_axes(ir: KernelIR) -> list[ValidityFailure]:
    """Every non-HBM buffer's ``buffer_scopes`` keys must be a subset of
    its ``dim_ids`` — except:

    * Matmul PSUM siblings may also carry the reducing dim (its scope
      affects emission depth without contributing to storage shape).
    * Matmul SBUF outputs may omit the reducing dim (codegen pins it
      to FULL).
    """
    failures: list[ValidityFailure] = []
    psum_extras = _psum_reducing_dims(ir)
    for name, buf in ir.physical_buffers.items():
        if buf.loc == "hbm":
            continue
        if name not in ir.buffer_scopes:
            failures.append(
                ValidityFailure(
                    check="buffer_scopes_missing",
                    buffer=name,
                    detail=f"buffer {name!r} has no buffer_scopes entry",
                    fix_hint=f"add buffer_scopes[{name!r}] with per-dim DimScope entries",
                )
            )
            continue
        allowed = set(buf.dim_ids) | psum_extras.get(name, set())
        scope_map = ir.buffer_scopes[name]
        for d in scope_map:
            if d not in allowed:
                failures.append(
                    ValidityFailure(
                        check="buffer_scopes_unknown_dim",
                        buffer=name,
                        detail=f"buffer_scopes[{name!r}] references {d!r} not in allowed dims {sorted(allowed)}",
                        fix_hint=f"remove {d!r} from buffer_scopes[{name!r}]",
                    )
                )
    return failures


def _check_feasible_emission_depth(ir: KernelIR) -> list[ValidityFailure]:
    """Every buffer must have a feasible emission depth under its per-dim scopes.

    Considers the implicit-FULL rule for matmul SBUF accumulators and
    the PSUM-outside-K rule for matmul PSUM siblings.
    """
    failures: list[ValidityFailure] = []
    psum_reducing = _psum_reducing_dims(ir)
    for name, buf in ir.physical_buffers.items():
        if buf.loc == "hbm":
            continue
        if name not in ir.buffer_scopes:
            continue
        scope_map = _effective_scope_map(ir, name)
        lower = 0
        upper = len(ir.loop_order)
        for d, scope in scope_map.items():
            block_entry = f"{d}.block"
            tile_entry = f"{d}.tile"
            if block_entry not in ir.loop_order or tile_entry not in ir.loop_order:
                continue
            block_pos = ir.loop_order.index(block_entry) + 1
            tile_pos = ir.loop_order.index(tile_entry) + 1
            if scope is DimScope.FULL:
                upper = min(upper, block_pos - 1)
            elif scope is DimScope.PER_BLOCK:
                lower = max(lower, block_pos)
            elif scope is DimScope.PER_TILE:
                lower = max(lower, tile_pos)
        if buf.loc == "psum":
            for d in psum_reducing.get(name, set()):
                tile_entry = f"{d}.tile"
                if tile_entry in ir.loop_order:
                    upper = min(upper, ir.loop_order.index(tile_entry) + 1 - 1)
        if upper < lower:
            failures.append(
                ValidityFailure(
                    check="infeasible_emission_depth",
                    buffer=name,
                    detail=f"no feasible depth (lower={lower}, upper={upper}) for buffer_scopes "
                    f"{ {k: v.value for k, v in scope_map.items()} }",
                    fix_hint="widen buffer_scopes or move loop_order entries",
                )
            )
    return failures


def _check_accumulator_coverage(ir: KernelIR) -> list[ValidityFailure]:
    """Matmul SBUF accumulator must be allocated OUTSIDE every reducing
    block loop of its producing matmul.

    For every non-reducing dim ``d`` whose constraint position
    (``pos({d}.block)`` for PER_BLOCK, ``pos({d}.tile)`` for
    PER_TILE) is ≥ the reducing dim's ``{k}.block`` position, the
    accumulator re-allocates inside the K loop and every iteration
    gets a fresh buffer — accumulation is broken. Those dims must be
    FULL on the accumulator.
    """
    failures: list[ValidityFailure] = []
    for op in ir.ops:
        if op.kind not in _MATMUL_KINDS or not op.outputs:
            continue
        sbuf_name = op.outputs[0]
        buf = ir.physical_buffers.get(sbuf_name)
        if buf is None or buf.loc != "sbuf":
            continue
        scope_map = _effective_scope_map(ir, sbuf_name)
        reducing_block_positions = [
            ir.loop_order.index(f"{d}.block") for d in op.blocking_dims if f"{d}.block" in ir.loop_order
        ]
        if not reducing_block_positions:
            continue
        first_red_pos = min(reducing_block_positions)
        for d in buf.dim_ids:
            if d in op.blocking_dims:
                continue
            scope = scope_map.get(d)
            if scope is DimScope.FULL:
                continue
            if scope is DimScope.PER_BLOCK:
                constraint_pos = ir.loop_order.index(f"{d}.block")
            elif scope is DimScope.PER_TILE:
                constraint_pos = ir.loop_order.index(f"{d}.tile")
            else:
                continue
            if constraint_pos >= first_red_pos:
                failures.append(
                    ValidityFailure(
                        check="accumulator_coverage",
                        buffer=sbuf_name,
                        detail=(
                            f"dim {d!r} scope {scope.value} forces accumulator inside the reducing "
                            f"block loop (constraint pos {constraint_pos} >= first_reducing_pos "
                            f"{first_red_pos})"
                        ),
                        fix_hint=(
                            f"set buffer_scopes[{sbuf_name!r}][{d!r}] = FULL or move {d!r}'s "
                            f"block/tile loops before the reducing dim in loop_order"
                        ),
                    )
                )
    return failures


def _check_sbuf_outlives_drain(ir: KernelIR) -> list[ValidityFailure]:
    """SBUF accumulator emission depth must be ≤ PSUM (drain) depth.

    The drain reads PSUM and writes into SBUF at the depth where
    PSUM's lifetime is established. SBUF must already be allocated
    there — otherwise the drain writes to a buffer that doesn't yet
    exist.
    """
    failures: list[ValidityFailure] = []
    for op in ir.ops:
        if op.kind not in _MATMUL_KINDS or not op.outputs:
            continue
        sbuf_name = op.outputs[0]
        psum_name = "psum_" + sbuf_name[len("sbuf_") :]
        if sbuf_name not in ir.buffer_scopes or psum_name not in ir.buffer_scopes:
            continue
        sbuf_depth = _emission_depth_lower(ir, sbuf_name, is_psum=False)
        psum_depth = _emission_depth_lower(ir, psum_name, is_psum=True)
        if sbuf_depth is None or psum_depth is None:
            continue
        if sbuf_depth > psum_depth:
            failures.append(
                ValidityFailure(
                    check="sbuf_outlives_drain",
                    buffer=sbuf_name,
                    detail=(
                        f"SBUF accumulator emission depth {sbuf_depth} is deeper than PSUM "
                        f"emission (drain) depth {psum_depth}"
                    ),
                    fix_hint=(
                        f"widen buffer_scopes[{sbuf_name!r}] with more FULL dims so its emission "
                        f"depth is ≤ {psum_depth}"
                    ),
                )
            )
    return failures


def _psum_reducing_dims(ir: KernelIR) -> dict[str, set[str]]:
    """Map PSUM buffer name → reducing dims it's allowed to carry."""
    result: dict[str, set[str]] = {}
    for op in ir.ops:
        if op.kind not in _MATMUL_KINDS or not op.outputs:
            continue
        sbuf_out = op.outputs[0]
        if not sbuf_out.startswith("sbuf_"):
            continue
        psum_name = "psum_" + sbuf_out[len("sbuf_") :]
        result[psum_name] = set(op.blocking_dims)
    return result


def _reducing_dims_for_buffer(ir: KernelIR, name: str) -> set[str]:
    """Reducing dims for matmul accumulator SBUF outputs."""
    for op in ir.ops:
        if op.kind not in _MATMUL_KINDS or not op.outputs:
            continue
        if name == op.outputs[0]:
            return set(op.blocking_dims)
    return set()


def _effective_scope_map(ir: KernelIR, name: str) -> dict[str, DimScope]:
    """Scope map with implicit FULL for reducing dims of SBUF outputs."""
    scope_map = dict(ir.buffer_scopes[name])
    buf = ir.physical_buffers[name]
    if buf.loc == "sbuf":
        reducing = _reducing_dims_for_buffer(ir, name)
        for d in reducing:
            if d in buf.dim_ids:
                scope_map.setdefault(d, DimScope.FULL)
    return scope_map


def _emission_depth_lower(ir: KernelIR, name: str, *, is_psum: bool) -> int | None:
    """Mirror of render's ``_buffer_emission_depth``: compute the
    minimal feasible emission depth for validation purposes."""
    if name not in ir.physical_buffers or name not in ir.buffer_scopes:
        return None
    buf = ir.physical_buffers[name]
    scope_map = _effective_scope_map(ir, name)
    reducing = _psum_reducing_dims(ir).get(name, set()) if is_psum else set()
    lower = 0
    upper = len(ir.loop_order)
    for d, scope in scope_map.items():
        if f"{d}.block" not in ir.loop_order or f"{d}.tile" not in ir.loop_order:
            continue
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
            if f"{d}.tile" in ir.loop_order:
                upper = min(upper, ir.loop_order.index(f"{d}.tile"))
    if upper < lower:
        return None
    _ = buf
    return lower
