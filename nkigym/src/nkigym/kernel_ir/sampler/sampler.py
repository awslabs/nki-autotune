"""Sampler: draw one valid ``KernelIR`` from ``(ctx, graph)`` + a rewrite registry.

Each ``sample_valid_ir`` call:

1. Samples a rewrite count ``k`` uniformly in ``[0, num_ops - 1]``.
2. Applies ``k`` rewrites from the unified ``REWRITES`` registry;
   at each step, draws one ``(pattern, instance)`` uniformly from
   the current match set across all patterns.
3. Rejection-samples ``dim_order`` / ``ltiles_per_block`` /
   ``tensor_placements`` for the resulting graph until ``validate``
   passes.

There is no outer/inner split. Every rewrite (trivial fusion,
online fusion, DMA-transpose fusion) is a ``PatternRewrite`` in
one registry, sampled in one loop.
"""

import random
from dataclasses import replace

from nkigym.kernel_ir.context.context import KernelContext
from nkigym.kernel_ir.graph.fusion_group import FusionGroup
from nkigym.kernel_ir.graph.graph import KernelGraph, rebuild_edges
from nkigym.kernel_ir.ir import KernelIR
from nkigym.kernel_ir.rewrites.pattern_rewrite import PatternRewrite
from nkigym.kernel_ir.validate.emission import compute_staged_set
from nkigym.kernel_ir.validate.rules import tier_depth_range, validate
from nkigym.ops.base import NKIOp

TIERS = ("per_tile", "per_block", "full")


def sample_valid_ir(
    context: KernelContext,
    graph: KernelGraph,
    rng: random.Random,
    rewrites: list[PatternRewrite],
    max_tries: int = 1_000_000,
) -> KernelIR:
    """Sample rewrites + per-group codegen state; return a valid ``KernelIR``."""
    k_max = max(0, len(context.op_inputs) - 1)
    context, graph = _apply_stochastic_rewrites(context, graph, rewrites, rng, k_max)
    tensor_kinds = _tensor_buffers(context, graph)
    sbuf_tensors = {name for name, kinds in tensor_kinds.items() if "sbuf" in kinds}
    staged = compute_staged_set(context, graph)
    seed_graph = _assign_initial_state(context, graph, tensor_kinds, sbuf_tensors)
    rebuild_edges(seed_graph, context)
    forced_full = _forced_full_pairs(context, seed_graph)
    op_to_group = {id(op): gi for gi, group in enumerate(seed_graph.groups) for op in group.ops}
    divisor_choices = _enumerate_lpb_choices(context)
    for _ in range(max_tries):
        iter_context = replace(context, ltiles_per_block={d: rng.choice(divisor_choices[d]) for d in divisor_choices})
        dim_orders = [_rand_blocking_inner_perm(o, b, rng) for o, b in _group_axis_splits(iter_context, seed_graph)]
        new_groups: list[FusionGroup] = []
        for gi, old in enumerate(seed_graph.groups):
            new_groups.append(
                replace(
                    old,
                    dim_order=dim_orders[gi],
                    tensor_placements=_sample_group_placements(
                        iter_context, seed_graph, gi, dim_orders[gi], forced_full, rng
                    ),
                )
            )
        candidate_graph = KernelGraph(groups=new_groups, edges=list(seed_graph.edges))
        candidate_ir = KernelIR(context=iter_context, graph=candidate_graph)
        if validate(candidate_ir, op_to_group, staged):
            return candidate_ir
    raise RuntimeError(f"No valid IR found after {max_tries} samples")


def _apply_stochastic_rewrites(
    context: KernelContext, graph: KernelGraph, rewrites: list[PatternRewrite], rng: random.Random, k_max: int
) -> tuple[KernelContext, KernelGraph]:
    """Apply ``k`` rewrites where ``k`` is drawn uniformly in ``[0, k_max]``."""
    k = rng.randint(0, k_max)
    current_ctx, current_graph = context, graph
    for _ in range(k):
        matches: list[tuple[PatternRewrite, object]] = []
        for pattern in rewrites:
            for instance in pattern.match(current_ctx, current_graph):
                matches.append((pattern, instance))
        if not matches:
            break
        pattern, instance = rng.choice(matches)
        current_ctx, current_graph = pattern.apply(current_ctx, current_graph, instance)
    return current_ctx, current_graph


def _enumerate_lpb_choices(context: KernelContext) -> dict[str, list[int]]:
    """Per-dim sorted divisors of ``dim_size // logical_tile_size`` — the legal ``ltiles_per_block`` set."""
    return {d: _divisors(di.dim_size // di.logical_tile_size) for d, di in context.dimensions.items()}


def _divisors(n: int) -> list[int]:
    """Sorted positive divisors of ``n`` (O(sqrt(n)))."""
    if n <= 0:
        raise ValueError(f"divisors requires n >= 1, got {n}")
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


def _tensor_buffers(context: KernelContext, graph: KernelGraph) -> dict[str, set[str]]:
    """Return ``{tensor_name: set_of_buffer_kinds}`` by walking the graph."""
    kinds: dict[str, set[str]] = {name: set() for name in context.logical_tensors}
    for group in graph.groups:
        for op in group.ops:
            isa_loc = type(op).ISA_LOC
            for oname in context.op_outputs.get(op, []):
                kinds.setdefault(oname, set()).add("psum" if isa_loc == "psum" else "sbuf")
    for group in graph.groups:
        for op in group.ops:
            input_locs = type(op).INPUT_LOCS
            for role, tname in context.op_inputs.get(op, {}).items():
                if tname in kinds and input_locs.get(role, "sbuf") == "sbuf":
                    kinds[tname].add("sbuf")
    return kinds


def _assign_initial_state(
    context: KernelContext, graph: KernelGraph, tensor_kinds: dict[str, set[str]], sbuf_tensors: set[str]
) -> KernelGraph:
    """Populate ``dim_order`` / ``buffer_degrees`` / ``tensor_placements`` on each group."""
    new_groups: list[FusionGroup] = []
    for group in graph.groups:
        touched = _group_touched_tensors(context, group)
        new_groups.append(
            replace(
                group,
                dim_order=_group_dims(context, group),
                buffer_degrees=_init_group_buffer_degrees(context, group, touched, tensor_kinds),
                tensor_placements=_init_group_tensor_placements(context, touched, sbuf_tensors),
            )
        )
    return KernelGraph(groups=new_groups, edges=list(graph.edges))


def _group_touched_tensors(context: KernelContext, group: FusionGroup) -> set[str]:
    """Return every tensor the group's ops touch (inputs + outputs)."""
    touched: set[str] = set()
    for op in group.ops:
        for name in context.op_inputs.get(op, {}).values():
            if name in context.logical_tensors:
                touched.add(name)
        for name in context.op_outputs.get(op, []):
            if name in context.logical_tensors:
                touched.add(name)
    return touched


def _group_dims(context: KernelContext, group: FusionGroup) -> list[str]:
    """Return every dim any op in the group touches, sorted by dim_id."""
    dims: set[str] = set()
    for op in group.ops:
        for name in _op_touched_tensors(context, op):
            tinfo = context.logical_tensors.get(name)
            if tinfo is not None:
                dims.update(tinfo.dim_ids)
    return sorted(dims)


def _op_touched_tensors(context: KernelContext, op: NKIOp) -> list[str]:
    """Inputs + outputs for one op."""
    return [*context.op_inputs.get(op, {}).values(), *context.op_outputs.get(op, [])]


def _init_group_buffer_degrees(
    context: KernelContext, group: FusionGroup, touched: set[str], tensor_kinds: dict[str, set[str]]
) -> dict[tuple[str, str, str], int]:
    """Set every physical-buffer degree to 1 for one group."""
    psum_producers: set[str] = set()
    for op in group.ops:
        if type(op).ISA_LOC == "psum":
            psum_producers.update(context.op_outputs.get(op, []))
    degrees: dict[tuple[str, str, str], int] = {}
    for tname in touched:
        kinds = tensor_kinds.get(tname, set())
        if "sbuf" in kinds:
            for dim_id in context.logical_tensors[tname].dim_ids:
                degrees[("sbuf", tname, dim_id)] = 1
        if "psum" in kinds and tname in psum_producers:
            for dim_id in context.logical_tensors[tname].dim_ids:
                degrees[("psum", tname, dim_id)] = 1
    return degrees


def _init_group_tensor_placements(
    context: KernelContext, touched: set[str], sbuf_tensors: set[str]
) -> dict[tuple[str, str, str], str]:
    """Seed every ``("sbuf", tensor, dim)`` placement to ``per_tile``."""
    placements: dict[tuple[str, str, str], str] = {}
    for tname in touched & sbuf_tensors:
        for dim_id in context.logical_tensors[tname].dim_ids:
            placements[("sbuf", tname, dim_id)] = "per_tile"
    return placements


def _group_axis_splits(context: KernelContext, graph: KernelGraph) -> list[tuple[list[str], set[str]]]:
    """For each group return ``(ordered_dims, blocking_dims)``."""
    splits: list[tuple[list[str], set[str]]] = []
    for group in graph.groups:
        order = list(group.dim_order)
        blocking: set[str] = set()
        for op in group.ops:
            blocking |= context.op_blocking_dims.get(op, set()) & set(order)
        splits.append((order, blocking))
    return splits


def _rand_blocking_inner_perm(order: list[str], blocking: set[str], rng: random.Random) -> list[str]:
    """Uniform random permutation with blocking-innermost constraint."""
    non_blocking = [d for d in order if d not in blocking]
    blocking_list = [d for d in order if d in blocking]
    rng.shuffle(non_blocking)
    rng.shuffle(blocking_list)
    return non_blocking + blocking_list


def _forced_full_pairs(context: KernelContext, graph: KernelGraph) -> set[tuple[int, str, str]]:
    """Return ``(group_idx, tensor, dim)`` triples forced to ``full`` by cross-group rules."""
    tensor_to_groups = _build_tensor_to_groups(context, graph)
    group_dim_sets = [set(group.dim_order) for group in graph.groups]
    forced: set[tuple[int, str, str]] = set()
    for tname, gis in tensor_to_groups.items():
        if len(gis) < 2:
            continue
        tensor_dims = set(context.logical_tensors[tname].dim_ids)
        for gi in gis:
            for d in group_dim_sets[gi] & tensor_dims:
                forced.add((gi, tname, d))
    return forced


def _build_tensor_to_groups(context: KernelContext, graph: KernelGraph) -> dict[str, set[int]]:
    """Map each tensor name to the set of group indices whose ops touch it."""
    result: dict[str, set[int]] = {}
    for gi, group in enumerate(graph.groups):
        for op in group.ops:
            for name in _op_touched_tensors(context, op):
                if name in context.logical_tensors:
                    result.setdefault(name, set()).add(gi)
    return result


def _sample_group_placements(
    context: KernelContext,
    graph: KernelGraph,
    group_idx: int,
    dim_order: list[str],
    forced_full: set[tuple[int, str, str]],
    rng: random.Random,
) -> dict[tuple[str, str, str], str]:
    """Draw one group's ``tensor_placements`` map."""
    group = graph.groups[group_idx]
    touched = {name for (_kind, name, _d) in group.tensor_placements}
    pos = {d: i for i, d in enumerate(dim_order)}
    n = len(dim_order)
    placements: dict[tuple[str, str, str], str] = {}
    for name in touched:
        tinfo = context.logical_tensors[name]
        depth = rng.randint(0, 2 * n) if n > 0 else 0
        for d in tinfo.dim_ids:
            key = ("sbuf", name, d)
            if (group_idx, name, d) in forced_full:
                placements[key] = "full"
            elif d in pos:
                placements[key] = _pick_tier_containing(pos[d], n, depth, rng)
            else:
                placements[key] = rng.choice(TIERS)
    return placements


def _pick_tier_containing(position: int, n: int, depth: int, rng: random.Random) -> str:
    """Pick a tier whose interval at ``position`` contains ``depth``."""
    candidates = [
        t for t in TIERS if tier_depth_range(t, position, n)[0] <= depth <= tier_depth_range(t, position, n)[1]
    ]
    return rng.choice(candidates) if candidates else rng.choice(TIERS)
