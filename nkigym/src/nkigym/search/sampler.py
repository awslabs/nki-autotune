"""Draw kernel variants: kernel_0 is the naive baseline; 1..N are stochastically rewritten."""

import random
from pathlib import Path

from nkigym.codegen import render_ir
from nkigym.codegen.buffers import _SBUF_BUFFER_CACHE
from nkigym.kernel_ir import KernelContext, KernelGraph, KernelIR, sample_valid_ir
from nkigym.kernel_ir.context.build import REWRITES


def sample_variants(
    naive_ctx: KernelContext,
    naive_graph: KernelGraph,
    ctx: KernelContext,
    graph: KernelGraph,
    num_variants: int,
    rng: random.Random,
    cache_dir: Path | None = None,
    max_tries_per_variant: int = 1_000_000,
) -> list[tuple[str, KernelIR, str]]:
    """Return ``num_variants`` unique ``(name, KernelIR, source)`` triples.

    The first output is always ``kernel_0`` — the raw baseline
    built from ``(naive_ctx, naive_graph)`` with NO rewrites
    applied. It samples only per-group codegen state. The
    remaining ``num_variants - 1`` kernels are drawn from
    ``(ctx, graph)`` with the full ``REWRITES`` registry
    sampled per draw.
    """
    _SBUF_BUFFER_CACHE.clear()
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    variants: list[tuple[str, KernelIR, str]] = []
    seen: set[str] = set()
    _emit_naive_kernel_0(naive_ctx, naive_graph, rng, cache_dir, variants, seen)
    _sample_remaining(ctx, graph, num_variants, rng, cache_dir, variants, seen, max_tries_per_variant)
    if len(variants) < num_variants:
        raise RuntimeError(f"Only {len(variants)}/{num_variants} unique variants produced")
    return variants


def _emit_naive_kernel_0(
    naive_ctx: KernelContext,
    naive_graph: KernelGraph,
    rng: random.Random,
    cache_dir: Path | None,
    variants: list[tuple[str, KernelIR, str]],
    seen: set[str],
) -> None:
    """Sample per-group codegen state for the raw baseline graph and record as kernel_0."""
    candidate = sample_valid_ir(naive_ctx, naive_graph, rng, rewrites=[])
    source = render_ir(candidate)
    _record_kernel(0, candidate, source, cache_dir, variants, seen)


def _sample_remaining(
    ctx: KernelContext,
    graph: KernelGraph,
    num_variants: int,
    rng: random.Random,
    cache_dir: Path | None,
    variants: list[tuple[str, KernelIR, str]],
    seen: set[str],
    max_tries_per_variant: int,
) -> None:
    """Fill kernels 1..N-1 by sampling rewrites + codegen state from ``(ctx, graph)``."""
    tries = 0
    budget = max_tries_per_variant * num_variants
    while len(variants) < num_variants and tries < budget:
        candidate = sample_valid_ir(ctx, graph, rng, rewrites=REWRITES)
        source = render_ir(candidate)
        tries += 1
        if source in seen:
            continue
        _record_kernel(len(variants), candidate, source, cache_dir, variants, seen)


def _record_kernel(
    idx: int,
    candidate: KernelIR,
    source: str,
    cache_dir: Path | None,
    variants: list[tuple[str, KernelIR, str]],
    seen: set[str],
) -> None:
    """Write the kernel's IR + op_graph to ``cache_dir`` and append to ``variants``."""
    seen.add(source)
    name = f"kernel_{idx}"
    if cache_dir is not None:
        variant_dir = cache_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        (variant_dir / f"ir_{idx}.md").write_text(repr(candidate))
        candidate.graph.render(variant_dir / "op_graph")
    variants.append((name, candidate, source))
