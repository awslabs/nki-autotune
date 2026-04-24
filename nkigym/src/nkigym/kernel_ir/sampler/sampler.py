"""Sampler: draw one valid ``KernelIR`` from a seed IR + a rewrite registry.

Each ``sample_valid_ir`` call:

1. Samples a rewrite count ``k`` uniformly in ``[0, num_ops - 1]``.
2. Applies ``k`` rewrites from the unified ``REWRITES`` registry;
   at each step, draws one ``(pattern, instance)`` uniformly from
   the current match set across all patterns.
3. Rejection-samples ``dim_order`` / ``ltiles_per_block`` /
   ``buffer_placements`` for the resulting ir until ``validate``
   passes.

There is no outer/inner split. Every rewrite (loop fusion,
online fusion, DMA-transpose fusion) is a ``PatternRewrite`` in
one registry, sampled in one loop.
"""

import random
from dataclasses import replace

from nkigym.kernel_ir.fusion_group import BufferPlacement, FusionGroup
from nkigym.kernel_ir.ir import KernelIR, rebuild_edges
from nkigym.kernel_ir.rewrites.pattern_rewrite import PatternRewrite
from nkigym.kernel_ir.validate.emission import compute_staged_set
from nkigym.kernel_ir.validate.rules import validate
from nkigym.ops.base import NKIOp
from nkigym.ops.dma import NKILoad

_PLACEMENTS: tuple[BufferPlacement, ...] = (BufferPlacement.OUTER, BufferPlacement.MIDDLE, BufferPlacement.INNER)


def sample_valid_ir(
    ir: KernelIR, rng: random.Random, rewrites: list[PatternRewrite], max_tries: int = 1_000_000
) -> KernelIR:
    """Sample rewrites + per-group codegen state; return a valid ``KernelIR``."""
    k_max = max(0, len(ir.op_inputs) - 1)
    ir = _apply_stochastic_rewrites(ir, rewrites, rng, k_max)
    tensor_kinds = _tensor_buffers(ir)
    sbuf_tensors = {name for name, kinds in tensor_kinds.items() if "sbuf" in kinds}
    staged = compute_staged_set(ir)
    seed_ir = _assign_initial_state(ir, tensor_kinds, sbuf_tensors)
    rebuild_edges(seed_ir)
    forced_outer = _forced_outer_pairs(seed_ir)
    op_to_group = {id(op): gi for gi, group in enumerate(seed_ir.groups) for op in group.ops}
    divisor_choices = _enumerate_lpb_choices(seed_ir)
    for _ in range(max_tries):
        candidate = replace(seed_ir, ltiles_per_block={d: rng.choice(divisor_choices[d]) for d in divisor_choices})
        dim_orders = [_rand_blocking_inner_perm(o, b, rng) for o, b in _group_axis_splits(candidate)]
        new_groups: list[FusionGroup] = []
        for gi, old in enumerate(seed_ir.groups):
            new_groups.append(
                replace(
                    old,
                    dim_order=dim_orders[gi],
                    buffer_placements=_sample_group_placements(candidate, gi, forced_outer, rng),
                )
            )
        candidate = replace(candidate, groups=new_groups, edges=list(seed_ir.edges))
        if validate(candidate, op_to_group, staged):
            return candidate
    raise RuntimeError(f"No valid IR found after {max_tries} samples")


def _apply_stochastic_rewrites(
    ir: KernelIR, rewrites: list[PatternRewrite], rng: random.Random, k_max: int
) -> KernelIR:
    """Apply ``k`` rewrites where ``k`` is drawn uniformly in ``[0, k_max]``."""
    k = rng.randint(0, k_max)
    current = ir
    for _ in range(k):
        matches: list[tuple[PatternRewrite, object]] = []
        for pattern in rewrites:
            for instance in pattern.match(current):
                matches.append((pattern, instance))
        if not matches:
            break
        pattern, instance = rng.choice(matches)
        current = pattern.apply(current, instance)
    return current


def _enumerate_lpb_choices(ir: KernelIR) -> dict[str, list[int]]:
    """Per-dim sorted divisors of ``dim_size // logical_tile_size`` — the legal ``ltiles_per_block`` set."""
    return {d: _divisors(di.dim_size // di.logical_tile_size) for d, di in ir.dimensions.items()}


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


def _tensor_buffers(ir: KernelIR) -> dict[str, set[str]]:
    """Return ``{tensor_name: set_of_buffer_kinds}`` by walking the IR."""
    kinds: dict[str, set[str]] = {name: set() for name in ir.all_tensors()}
    for group in ir.groups:
        for op in group.ops:
            isa_loc = type(op).ISA_LOC
            for oname in ir.op_outputs.get(op, []):
                kinds.setdefault(oname, set()).add("psum" if isa_loc == "psum" else "sbuf")
    for group in ir.groups:
        for op in group.ops:
            input_locs = type(op).INPUT_LOCS
            for role, tname in ir.op_inputs.get(op, {}).items():
                if tname in kinds and input_locs.get(role, "sbuf") == "sbuf":
                    kinds[tname].add("sbuf")
    return kinds


def _assign_initial_state(ir: KernelIR, tensor_kinds: dict[str, set[str]], sbuf_tensors: set[str]) -> KernelIR:
    """Populate ``dim_order`` / ``buffer_degrees`` / ``buffer_placements`` on each group.

    ``buffer_placements`` only holds entries for **FG input
    buffers** — physical_buffers produced by an ``NKILoad`` in
    this group (i.e. buffers the FG reads data INTO from HBM).
    Output buffers and intermediates derive their placement via
    ``placement_semantics.effective_placement``.
    """
    _ = sbuf_tensors
    new_groups: list[FusionGroup] = []
    for group in ir.groups:
        touched = _group_touched_tensors(ir, group)
        new_groups.append(
            replace(
                group,
                dim_order=_group_dims(ir, group),
                buffer_degrees=_init_group_buffer_degrees(ir, group, touched, tensor_kinds),
                buffer_placements=_init_group_buffer_placements(ir, group),
            )
        )
    return replace(ir, groups=new_groups, edges=list(ir.edges))


def _group_touched_tensors(ir: KernelIR, group: FusionGroup) -> set[str]:
    """Return every tensor the group's ops touch (inputs + outputs)."""
    touched: set[str] = set()
    for op in group.ops:
        for name in ir.op_inputs.get(op, {}).values():
            if ir.has_tensor(name):
                touched.add(name)
        for name in ir.op_outputs.get(op, []):
            if ir.has_tensor(name):
                touched.add(name)
    return touched


def _group_dims(ir: KernelIR, group: FusionGroup) -> list[str]:
    """Return every dim any op in the group touches, sorted by dim_id."""
    dims: set[str] = set()
    for op in group.ops:
        for name in _op_touched_tensors(ir, op):
            if ir.has_tensor(name):
                dims.update(ir.tensor_info(name).dim_ids)
    return sorted(dims)


def _op_touched_tensors(ir: KernelIR, op: NKIOp) -> list[str]:
    """Inputs + outputs for one op."""
    return [*ir.op_inputs.get(op, {}).values(), *ir.op_outputs.get(op, [])]


def _init_group_buffer_degrees(
    ir: KernelIR, group: FusionGroup, touched: set[str], tensor_kinds: dict[str, set[str]]
) -> dict[tuple[str, str, str], int]:
    """Set every physical-buffer degree to 1 for one group."""
    psum_producers: set[str] = set()
    for op in group.ops:
        if type(op).ISA_LOC == "psum":
            psum_producers.update(ir.op_outputs.get(op, []))
    degrees: dict[tuple[str, str, str], int] = {}
    for tname in touched:
        kinds = tensor_kinds.get(tname, set())
        if "sbuf" in kinds:
            for dim_id in ir.tensor_info(tname).dim_ids:
                degrees[("sbuf", tname, dim_id)] = 1
        if "psum" in kinds and tname in psum_producers:
            for dim_id in ir.tensor_info(tname).dim_ids:
                degrees[("psum", tname, dim_id)] = 1
    return degrees


def _init_group_buffer_placements(ir: KernelIR, group: FusionGroup) -> dict[tuple[str, str], BufferPlacement]:
    """Seed placements for each Load-destination physical_buffer this group produces.

    An FG input buffer is a physical_buffer produced by an
    ``NKILoad`` op in this group — the buffer holds data the FG
    reads from HBM.
    """
    result: dict[tuple[str, str], BufferPlacement] = {}
    for op in group.ops:
        if not isinstance(op, NKILoad):
            continue
        for oname in ir.op_outputs.get(op, []):
            if oname in ir.physical_buffers:
                result[("sbuf", oname)] = BufferPlacement.INNER
    return result


def _group_axis_splits(ir: KernelIR) -> list[tuple[list[str], set[str]]]:
    """For each group return ``(ordered_dims, blocking_dims)``."""
    splits: list[tuple[list[str], set[str]]] = []
    for group in ir.groups:
        order = list(group.dim_order)
        blocking: set[str] = set()
        for op in group.ops:
            blocking |= ir.op_blocking_dims.get(op, set()) & set(order)
        splits.append((order, blocking))
    return splits


def _rand_blocking_inner_perm(order: list[str], blocking: set[str], rng: random.Random) -> list[str]:
    """Uniform random permutation over all dim orderings."""
    _ = blocking
    permuted = list(order)
    rng.shuffle(permuted)
    return permuted


def _forced_outer_pairs(ir: KernelIR) -> set[tuple[int, str]]:
    """Return ``(group_idx, tensor)`` pairs forced to ``OUTER`` by cross-group rules.

    A tensor touched by 2+ groups must be allocated at the kernel
    top so its data persists across the group boundary; every
    touching group must therefore have ``OUTER`` placement.
    """
    tensor_to_groups = _build_tensor_to_groups(ir)
    return {(gi, tname) for tname, gis in tensor_to_groups.items() if len(gis) >= 2 for gi in gis}


def _build_tensor_to_groups(ir: KernelIR) -> dict[str, set[int]]:
    """Map each tensor name to the set of group indices whose ops touch it."""
    result: dict[str, set[int]] = {}
    for gi, group in enumerate(ir.groups):
        for op in group.ops:
            for name in _op_touched_tensors(ir, op):
                if ir.has_tensor(name):
                    result.setdefault(name, set()).add(gi)
    return result


def _sample_group_placements(
    ir: KernelIR, group_idx: int, forced_outer: set[tuple[int, str]], rng: random.Random
) -> dict[tuple[str, str], BufferPlacement]:
    """Draw one group's ``buffer_placements`` — one ``BufferPlacement`` per FG input buffer.

    Only Load-destination physical_buffers get a sampled placement.
    Cross-FG input buffers are forced to ``OUTER`` so data survives
    the group boundary.
    """
    group = ir.groups[group_idx]
    placements: dict[tuple[str, str], BufferPlacement] = {}
    for key in group.buffer_placements:
        _kind, tname = key
        if (group_idx, tname) in forced_outer:
            placements[key] = BufferPlacement.OUTER
        else:
            placements[key] = rng.choice(_PLACEMENTS)
    return placements
