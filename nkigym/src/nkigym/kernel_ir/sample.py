"""Random sampler over the full KernelIR knob space.

Samples are built one knob at a time in dependency order, each knob
conditioned on the partial IR so far. No rejection loop — every draw
satisfies :func:`validate.is_valid` by construction.

Order:

1. ``dim_order`` — free permutation.
2. ``ltiles_per_block`` — free divisor per dim.
3. ``buffer_scopes`` — free per buffer.
4. ``num_buffers`` — per buffer per axis; rotation on an axis is only
   offered when that axis's loop is open at the producer op's depth.
5. ``emission_depth`` — per buffer integer in ``[0, producer_op_depth]``.
"""

import random
from dataclasses import replace

from nkigym.kernel_ir.ir import BufferScope, KernelIR, NumBuffers
from nkigym.kernel_ir.validate import axis_open, op_depth, producer_op


def sample(ir: KernelIR, rng: random.Random | None = None) -> KernelIR:
    """Return a new ``KernelIR`` with every tunable knob randomized.

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
    partial = replace(ir, dim_order=sample_dim_order(ir, rng), ltiles_per_block=sample_ltiles_per_block(ir, rng))
    partial = replace(partial, buffer_scopes=sample_buffer_scopes(partial, rng))
    partial = replace(partial, num_buffers=sample_num_buffers(partial, rng))
    return replace(partial, emission_depth=sample_emission_depth(partial, rng))


def sample_dim_order(ir: KernelIR, rng: random.Random) -> list[str]:
    """One random permutation of ``ir.dimensions`` — ``N!`` options."""
    dims = list(ir.dimensions)
    rng.shuffle(dims)
    return dims


def sample_ltiles_per_block(ir: KernelIR, rng: random.Random) -> dict[str, int]:
    """Per-dim random divisor of ``num_ltile`` along that dim."""
    return {d: rng.choice(_divisors(_num_ltile(ir, d))) for d in ir.dimensions}


def sample_buffer_scopes(ir: KernelIR, rng: random.Random) -> dict[str, BufferScope]:
    """Per-buffer random choice from ``{INNER, MIDDLE, OUTER}``.

    Transpose src/dst pairs share one scope — the transpose gadget
    requires matching tile counts on the swapped axes, which is only
    true when both sides use the same scope-sizing rule.
    """
    scopes = list(BufferScope)
    result = {name: rng.choice(scopes) for name in _tunable_buffers(ir)}
    for op in ir.ops:
        if op.kind not in ("NKITranspose", "NKIDMATranspose"):
            continue
        src = op.inputs.get("data")
        dst = op.outputs[0] if op.outputs else None
        if src in result and dst in result:
            result[dst] = result[src]
    return result


def sample_num_buffers(ir: KernelIR, rng: random.Random) -> dict[str, NumBuffers]:
    """Per-buffer per-axis rotation factor.

    ``None`` is always allowed. Integer rotation is offered only when
    the buffer's producer op runs inside the axis's block loop — the
    rotation index is emitted at producer depth, so the axis's
    ``i_block_<axis>`` must already be bound.
    """
    result: dict[str, NumBuffers] = {}
    for name in _tunable_buffers(ir):
        buf = ir.physical_buffers[name]
        producer = producer_op(ir, name)
        depth = op_depth(ir, producer) if producer is not None else len(ir.dim_order)
        p_choices: list[int | None] = [None]
        if axis_open(ir, buf.p_axis, depth):
            p_choices.extend(_divisors(_num_ltile(ir, buf.p_axis)))
        f_choices: list[int | None] = [None]
        if buf.f_axis is not None and axis_open(ir, buf.f_axis, depth):
            f_choices.extend(_divisors(_num_ltile(ir, buf.f_axis)))
        result[name] = NumBuffers(num_p_buffers=rng.choice(p_choices), num_f_buffers=rng.choice(f_choices))
    return result


def sample_emission_depth(ir: KernelIR, rng: random.Random) -> dict[str, int]:
    """Per-buffer random depth in ``[0, producer_op_depth]``.

    Allocating deeper than the producer would place the allocation
    after its first use (``cur_<buf> = <buf>[...]``).
    """
    result: dict[str, int] = {}
    for name in _tunable_buffers(ir):
        producer = producer_op(ir, name)
        max_depth = op_depth(ir, producer) if producer is not None else len(ir.dim_order)
        result[name] = rng.choice(list(range(max_depth + 1)))
    return result


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
