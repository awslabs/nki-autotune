"""Random sampler over the full KernelIR knob space.

For an IR with ``N`` dimensions and ``B`` tunable (non-HBM) physical
buffers, the combinatorial space is:

* ``dim_order`` — ``N!`` permutations of ``ir.dimensions``.
* ``ltiles_per_block`` — per-dim divisor of ``num_ltile`` along that
  dim; ``prod_d divisors(num_ltile_d)`` total.
* ``buffer_scopes`` — per-buffer one of ``{INNER, MIDDLE, OUTER}``;
  ``3^B`` total.
* ``num_buffers`` — per-buffer per-axis choice from
  ``{None} ∪ divisors(num_ltile_axis)``; product across buffers/axes.
* ``emission_depth`` — per-buffer integer in ``[0, N]``;
  ``(N+1)^B`` total.

``sample`` draws ONE random assignment for every knob and rejects
assignments that fail ``validate.is_valid``, retrying up to
``_MAX_RETRIES`` times. Samples that pass may still fail render /
compile / sim for other reasons.
"""

import random
from dataclasses import replace

from nkigym.kernel_ir.ir import BufferScope, KernelIR, NumBuffers
from nkigym.kernel_ir.validate import is_valid

_MAX_RETRIES = 100


def sample(ir: KernelIR, rng: random.Random | None = None) -> KernelIR:
    """Return a new ``KernelIR`` with every tunable knob randomized.

    The returned IR is guaranteed to pass :func:`validate.is_valid`.
    If a draw fails validity, only ``emission_depth`` is redrawn (the
    only knob whose random choice can violate a structural invariant);
    retries up to ``_MAX_RETRIES`` before raising.

    Args:
        ir: Source IR — typically the canonical-default output of
            ``build_ir``. Non-tunable fields (``dimensions``,
            ``logical_tensors``, ``physical_buffers``, ``ops``,
            ``edges``, etc.) are passed through unchanged.
        rng: Optional ``random.Random`` for deterministic runs. A
            fresh default-seeded generator is used when omitted.

    Returns:
        A valid ``KernelIR`` with randomized ``dim_order``,
        ``ltiles_per_block``, ``buffer_scopes``, ``num_buffers``, and
        ``emission_depth``.
    """
    rng = rng or random.Random()
    candidate = replace(
        ir,
        dim_order=sample_dim_order(ir, rng),
        ltiles_per_block=sample_ltiles_per_block(ir, rng),
        buffer_scopes=sample_buffer_scopes(ir, rng),
        num_buffers=sample_num_buffers(ir, rng),
        emission_depth=sample_emission_depth(ir, rng),
    )
    for _ in range(_MAX_RETRIES):
        if is_valid(candidate):
            return candidate
        candidate = replace(
            candidate,
            buffer_scopes=sample_buffer_scopes(candidate, rng),
            num_buffers=sample_num_buffers(candidate, rng),
            emission_depth=sample_emission_depth(candidate, rng),
        )
    raise RuntimeError(f"sample: could not satisfy is_valid within {_MAX_RETRIES} retries")


def sample_dim_order(ir: KernelIR, rng: random.Random) -> list[str]:
    """One random permutation of ``ir.dimensions`` — ``N!`` options."""
    dims = list(ir.dimensions)
    rng.shuffle(dims)
    return dims


def sample_ltiles_per_block(ir: KernelIR, rng: random.Random) -> dict[str, int]:
    """Per-dim random divisor of ``num_ltile`` along that dim."""
    return {d: rng.choice(_divisors(_num_ltile(ir, d))) for d in ir.dimensions}


def sample_buffer_scopes(ir: KernelIR, rng: random.Random) -> dict[str, BufferScope]:
    """Per-buffer random choice from ``{INNER, MIDDLE, OUTER}``."""
    scopes = list(BufferScope)
    return {name: rng.choice(scopes) for name in _tunable_buffers(ir)}


def sample_num_buffers(ir: KernelIR, rng: random.Random) -> dict[str, NumBuffers]:
    """Per-buffer per-axis choice from ``{None} ∪ divisors(num_ltile)``."""
    result: dict[str, NumBuffers] = {}
    for name in _tunable_buffers(ir):
        buf = ir.physical_buffers[name]
        p_choices: list[int | None] = [None, *_divisors(_num_ltile(ir, buf.p_axis))]
        f_choices: list[int | None] = (
            [None, *_divisors(_num_ltile(ir, buf.f_axis))] if buf.f_axis is not None else [None]
        )
        result[name] = NumBuffers(num_p_buffers=rng.choice(p_choices), num_f_buffers=rng.choice(f_choices))
    return result


def sample_emission_depth(ir: KernelIR, rng: random.Random) -> dict[str, int]:
    """Per-buffer random depth in ``[0, N]`` — ``(N+1)^B`` combos."""
    depths = list(range(len(ir.dimensions) + 1))
    return {name: rng.choice(depths) for name in _tunable_buffers(ir)}


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
