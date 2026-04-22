"""Hierarchical two-layer draw: outer picks op-graph variant, inner draws per-group state."""

import random
from pathlib import Path

from nkigym.codegen import render_ir
from nkigym.codegen.buffers import _SBUF_BUFFER_CACHE
from nkigym.kernel_ir import KernelContext, KernelGraph, KernelIR, sample_valid_ir
from nkigym.kernel_ir.context.build import MERGE_REWRITES


def sample_variants(
    ctx: KernelContext,
    graph_variants: list[KernelGraph],
    num_variants: int,
    rng: random.Random,
    cache_dir: Path | None = None,
    max_tries_per_variant: int = 1_000_000,
) -> list[tuple[str, KernelIR, str]]:
    """Return ``num_variants`` unique ``(name, KernelIR, source)`` triples."""
    if not graph_variants:
        raise ValueError("graph_variants must be non-empty")
    _SBUF_BUFFER_CACHE.clear()
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    variants: list[tuple[str, KernelIR, str]] = []
    tries = 0
    budget = max_tries_per_variant * num_variants
    while len(variants) < num_variants and tries < budget:
        chosen_graph = rng.choice(graph_variants)
        seed_ir = KernelIR(context=ctx, graph=chosen_graph)
        candidate = sample_valid_ir(seed_ir, rng, merge_rewrites=MERGE_REWRITES)
        source = render_ir(candidate)
        tries += 1
        if source in seen:
            continue
        seen.add(source)
        idx = len(variants)
        name = f"kernel_{idx}"
        if cache_dir is not None:
            variant_dir = cache_dir / name
            variant_dir.mkdir(parents=True, exist_ok=True)
            (variant_dir / f"ir_{idx}.md").write_text(repr(candidate))
            candidate.graph.render(variant_dir / "op_graph")
        variants.append((name, candidate, source))
    if len(variants) < num_variants:
        raise RuntimeError(f"Only {len(variants)}/{num_variants} unique variants after {tries} draws")
    return variants
