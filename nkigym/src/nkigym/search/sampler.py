"""Sampler: draw unique rendered variants from a seed KernelIR via rejection sampling."""

import random
from pathlib import Path

from nkigym.codegen import render_ir
from nkigym.kernel_ir import KernelIR
from nkigym.kernel_ir.ir import sample_valid_ir


def sample_variants(
    ir: KernelIR,
    num_variants: int,
    rng: random.Random,
    cache_dir: Path | None = None,
    max_tries_per_variant: int = 1_000_000,
) -> list[tuple[str, KernelIR, str]]:
    """Return ``num_variants`` unique ``(name, KernelIR, source)`` triples.

    Each draw calls ``sample_valid_ir`` (joint rejection sampling
    on ``group_dim_orders`` and ``tensor_placements``) and renders
    to source. Duplicates (same rendered source) are skipped. When
    ``cache_dir`` is set, each variant gets its own directory
    ``<cache_dir>/kernels/<name>/`` containing ``ir_<N>.md`` (a
    dump of the sampled ``KernelIR`` for variant N); the kernel
    source itself is written by ``remote_profile``. Raises
    ``RuntimeError`` if uniqueness can't be reached within
    ``max_tries_per_variant * num_variants`` draws.
    """
    kernels_dir = cache_dir / "kernels" if cache_dir is not None else None
    if kernels_dir is not None:
        kernels_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    variants: list[tuple[str, KernelIR, str]] = []
    tries = 0
    budget = max_tries_per_variant * num_variants
    while len(variants) < num_variants and tries < budget:
        candidate = sample_valid_ir(ir, rng)
        source = render_ir(candidate)
        tries += 1
        if source in seen:
            continue
        seen.add(source)
        idx = len(variants)
        name = f"kernel_{idx}"
        if kernels_dir is not None:
            variant_dir = kernels_dir / name
            variant_dir.mkdir(parents=True, exist_ok=True)
            (variant_dir / f"ir_{idx}.md").write_text(repr(candidate))
        variants.append((name, candidate, source))
    if len(variants) < num_variants:
        raise RuntimeError(f"Only {len(variants)}/{num_variants} unique variants after {tries} draws")
    return variants
