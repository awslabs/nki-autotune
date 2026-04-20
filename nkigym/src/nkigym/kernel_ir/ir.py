"""KernelIR: structured representation for lowering to NKI source."""

import random
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from nkigym.kernel_ir.dim_analysis import DimAnalysis, analyze_dims, op_blocking_dims
from nkigym.kernel_ir.op_graph import OpGraph, build_op_graph
from nkigym.kernel_ir.partition import compute_reachability, op_dims_of, sample_partition
from nkigym.kernel_ir.validate import tier_depth_range, validate

TIERS = ("per_tile", "per_block", "full")
_TIER_RANK = {"per_tile": 0, "per_block": 1, "full": 2}


@dataclass
class FusionGroup:
    """Per-fusion-group state.

    Everything in here describes ONE group's rendering decisions.
    Tensors appear here iff the group's ops actually touch them.

    Attributes:
        op_indices: Op indices (from ``op_graph.op_classes``)
            that execute inside this group's loop nest.
        dim_order: Complete ordered list of dim IDs this group
            loops over (outer-to-inner). Covers every dim the
            group's ops touch. An empty list marks a group with
            no loops (every op has trip count 1 on every dim it
            touches). Stored rather than derived so loop-
            reordering transforms can permute the order without
            mutating ``op_graph``.
        buffer_degrees: Per (buffer_kind, tensor_name, dim_id)
            multi-buffering degree for buffers allocated/used in
            this group. ``buffer_kind`` is ``"sbuf"`` or
            ``"psum"``. A tensor appears iff the group's ops
            produce or consume its buffer.
        tensor_placements: Per (buffer_kind, tensor_name, dim_id)
            tier (``per_tile`` / ``per_block`` / ``full``) — this
            group's own loop-scope decision for each physical
            buffer it touches. Today ``buffer_kind`` is always
            ``"sbuf"``; PSUM buffers carry no tier (allocated
            greedily at the narrowest scope, see
            ``render_psum_allocations``). The key mirrors
            ``buffer_degrees`` so logical tensors with multiple
            physical buffers remain unambiguous.
    """

    op_indices: list[int]
    dim_order: list[str]
    buffer_degrees: dict[tuple[str, str, str], int]
    tensor_placements: dict[tuple[str, str, str], str]


@dataclass
class KernelIR:
    """Complete representation for lowering to NKI source.

    Composes two independent analysis results with rendering
    parameters that control loop structure, buffer sizes, and
    DMA placement.

    Attributes:
        dim_analysis: Dimension IDs, tile sizes, tensor metadata.
            Global — derived from the math function alone.
        op_graph: Producer-consumer DAG. Global — derived from
            the math function alone.
        ltiles_per_block: Per-dimension tiling factor. Global;
            groups sharing a dim must agree on its tpb.
        fusion_groups: Which ops share a loop nest, plus per-
            group loop order, buffer degrees, and tier
            placements. Initially one singleton group per op;
            loop fusion merges groups.
    """

    dim_analysis: DimAnalysis
    op_graph: OpGraph
    ltiles_per_block: dict[str, int]
    fusion_groups: list[FusionGroup]

    def __post_init__(self) -> None:
        """Precompute cross-group effective-tier/degree maps used by codegen."""
        self._effective_tier: dict[tuple[str, str, str], str] = {}
        for group in self.fusion_groups:
            for (k, t, d), tier in group.tensor_placements.items():
                prev = self._effective_tier.get((k, t, d))
                if prev is None or _TIER_RANK[tier] > _TIER_RANK[prev]:
                    self._effective_tier[(k, t, d)] = tier
        self._effective_degree: dict[tuple[str, str, str], int] = {}
        for group in self.fusion_groups:
            for (k, t, d), deg in group.buffer_degrees.items():
                prev = self._effective_degree.get((k, t, d), 0)
                if deg > prev:
                    self._effective_degree[(k, t, d)] = deg

    def __repr__(self) -> str:
        """Show KernelIR with each field on its own line."""
        lines = [
            "KernelIR(",
            f"  dim_analysis={self.dim_analysis!r}",
            f"  op_graph={self.op_graph!r}",
            "  ltiles_per_block=",
            self._fmt_ltiles_per_block(),
            "  fusion_groups=",
            self._fmt_fusion_groups(),
            ")",
        ]
        return "\n".join(lines)

    def _fmt_ltiles_per_block(self) -> str:
        """Format ltiles_per_block as a dim → count table."""
        rows = [[dim_id, str(self.ltiles_per_block[dim_id])] for dim_id in sorted(self.ltiles_per_block)]
        return _fmt_table(["dim", "ltiles_per_block"], rows)

    def _fmt_fusion_groups(self) -> str:
        """Format each fusion group's state as a labeled block."""
        blocks: list[str] = []
        for gi, group in enumerate(self.fusion_groups):
            header = f"    Group {gi}: ops={group.op_indices}, dim_order={group.dim_order}"
            deg_rows = [
                [kind, tensor, dim_id, str(degree)]
                for (kind, tensor, dim_id), degree in sorted(group.buffer_degrees.items())
            ]
            plc_rows = [
                [kind, tensor, dim_id, tier] for (kind, tensor, dim_id), tier in sorted(group.tensor_placements.items())
            ]
            blocks.append(header)
            if deg_rows:
                blocks.append("      buffer_degrees=")
                blocks.append(_fmt_table(["kind", "tensor", "dim", "degree"], deg_rows))
            if plc_rows:
                blocks.append("      tensor_placements=")
                blocks.append(_fmt_table(["kind", "tensor", "dim", "placement"], plc_rows))
        return "\n".join(blocks)


def _fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a list of rows as an aligned text table."""
    col_widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    header_line = "  | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    data_lines = ["  | ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))) for r in rows]
    pad = "    "
    return "\n".join([f"{pad}{header_line}", f"{pad}{sep_line}"] + [f"{pad}{line}" for line in data_lines])


def get_tpb(ir: KernelIR, dim_id: str) -> int:
    """Return ltiles_per_block for a dimension.

    ``ltiles_per_block`` is per-dimension — the same block
    structure applies to every op and tensor touching that
    dim. Missing keys raise ``ValueError``.

    Args:
        ir: Complete kernel IR.
        dim_id: Dimension to look up.

    Returns:
        ltiles_per_block value for the dimension.

    Raises:
        ValueError: If the dim is not in ``ir.ltiles_per_block``.
    """
    if dim_id not in ir.ltiles_per_block:
        raise ValueError(f"No ltiles_per_block for dim {dim_id!r}")
    return ir.ltiles_per_block[dim_id]


def _init_group_buffer_degrees(
    da: DimAnalysis,
    graph: OpGraph,
    group_op_indices: list[int],
    group_tensors: set[str],
    tensor_kinds: dict[str, set[str]],
) -> dict[tuple[str, str, str], int]:
    """Set every physical-buffer degree to 1 for one group.

    One entry per ``(buffer_kind, tensor_name, dim_id)`` for
    every physical buffer materialized by the op graph on a
    tensor this group touches. Buffer kinds come from
    ``tensor_buffers``: a tensor has a PSUM entry in this group
    iff the group contains its producer; it has an SBUF entry
    iff any op in the group uses SBUF for it.
    """
    degrees: dict[tuple[str, str, str], int] = {}
    group_psum_producers = {
        oname
        for op_idx in group_op_indices
        if graph.op_classes[op_idx].ISA_LOC == "psum"
        for oname in graph.op_tensors[op_idx][1]
    }
    for tensor_name in group_tensors:
        kinds = tensor_kinds.get(tensor_name, set())
        if "sbuf" in kinds:
            for dim_id in da.tensors[tensor_name].dim_ids:
                degrees[("sbuf", tensor_name, dim_id)] = 1
        if "psum" in kinds and tensor_name in group_psum_producers:
            for dim_id in da.tensors[tensor_name].dim_ids:
                degrees[("psum", tensor_name, dim_id)] = 1
    return degrees


def build_tensor_to_groups(ir: "KernelIR") -> dict[str, set[int]]:
    """Map every tensor to the set of fusion-group indices whose ops touch it."""
    result: dict[str, set[int]] = {}
    da = ir.dim_analysis
    graph = ir.op_graph
    for gi, group in enumerate(ir.fusion_groups):
        for op_idx in group.op_indices:
            for name in graph.op_tensor_names(op_idx):
                if name in da.tensors:
                    result.setdefault(name, set()).add(gi)
    return result


def tensor_buffers(da: DimAnalysis, graph: OpGraph) -> dict[str, set[str]]:
    """Return ``{tensor_name: set_of_buffer_kinds}`` by walking the op graph.

    A tensor has a PSUM buffer iff its producer has ``ISA_LOC ==
    "psum"``. A tensor has an SBUF buffer iff any of the following
    holds: it is a kernel input, it is the return tensor, a
    consumer op reads it via ``INPUT_LOCS[role] == "sbuf"``, or
    it is a non-PSUM op's output. A PSUM-produced tensor that
    flows through any SBUF-reading consumer (or is the return
    tensor) also gets an SBUF buffer for staging.
    """
    kinds: dict[str, set[str]] = {name: set() for name in da.tensors}
    for name in da.param_names:
        kinds[name].add("sbuf")
    if da.return_name in kinds:
        kinds[da.return_name].add("sbuf")
    for op_idx, op_cls in enumerate(graph.op_classes):
        _, outputs = graph.op_tensors[op_idx]
        for oname in outputs:
            kinds[oname].add("psum" if op_cls.ISA_LOC == "psum" else "sbuf")
    for consumer_idx, (inputs, _outputs) in enumerate(graph.op_tensors):
        input_locs = graph.op_classes[consumer_idx].INPUT_LOCS
        for role, tname in inputs.items():
            if tname in kinds and input_locs.get(role, "sbuf") == "sbuf":
                kinds[tname].add("sbuf")
    return kinds


def _init_group_tensor_placements(
    da: DimAnalysis, group_tensors: set[str], sbuf_tensors: set[str]
) -> dict[tuple[str, str, str], str]:
    """Set every ``("sbuf", tensor, dim)`` placement to per_tile for one group.

    Placements key on ``(buffer_kind, tensor_name, dim_id)`` to
    mirror ``buffer_degrees``. Only ``sbuf`` entries are
    populated today — PSUM has no meaningful tier choice
    (allocated greedily at a group-local scope, see
    ``render_psum_allocations``). Each group gets independent
    tier entries for the SBUF-backed tensors its ops actually
    touch.
    """
    placements: dict[tuple[str, str, str], str] = {}
    for tensor_name in group_tensors & sbuf_tensors:
        for dim_id in da.tensors[tensor_name].dim_ids:
            placements[("sbuf", tensor_name, dim_id)] = "per_tile"
    return placements


def _group_touched_tensors(graph: OpGraph, da: DimAnalysis, op_indices: list[int]) -> set[str]:
    """Return every tensor any op in ``op_indices`` touches (inputs + outputs)."""
    touched: set[str] = set()
    for op_idx in op_indices:
        for name in graph.op_tensor_names(op_idx):
            if name in da.tensors:
                touched.add(name)
    return touched


def sample_valid_ir(ir: "KernelIR", rng: random.Random, max_tries: int = 1_000_000) -> "KernelIR":
    """Rejection-sample a full IR: fusion partition, ``dim_order``, and ``tensor_placements``.

    Per try: draw ``fusion_groups`` via stochastic pairwise merging
    (see ``partition.sample_partition``), then per-group draw
    ``dim_order`` (blocking dims innermost) and ``tensor_placements``
    (per-tensor emission depth feasible by construction; cross-group
    shared dims forced to ``full``). Accept jointly against
    ``validate``. PSUM buffers carry no tier — allocated greedily
    just before use. Raises ``RuntimeError`` if no valid combination
    is found within ``max_tries`` tries.
    """
    da = ir.dim_analysis
    graph = ir.op_graph
    reach = compute_reachability(graph)
    op_dims = [op_dims_of(graph, da, i) for i in range(len(graph.op_classes))]
    tensor_kinds = tensor_buffers(da, graph)
    sbuf_tensors = {name for name, kinds in tensor_kinds.items() if "sbuf" in kinds}
    for _ in range(max_tries):
        op_groups = sample_partition(graph, da, rng, reach=reach, op_dims=op_dims)
        seed_groups: list[FusionGroup] = []
        for op_indices in op_groups:
            touched = _group_touched_tensors(graph, da, op_indices)
            seed_groups.append(
                FusionGroup(
                    op_indices=op_indices,
                    dim_order=_group_dims(op_indices, da, graph),
                    buffer_degrees=_init_group_buffer_degrees(da, graph, op_indices, touched, tensor_kinds),
                    tensor_placements=_init_group_tensor_placements(da, touched, sbuf_tensors),
                )
            )
        seed = replace(ir, fusion_groups=seed_groups)
        tensor_to_groups = build_tensor_to_groups(seed)
        forced_full = _forced_full_pairs(seed, tensor_to_groups)
        op_to_group = {op_idx: gi for gi, group in enumerate(seed_groups) for op_idx in group.op_indices}
        dim_orders = [_rand_blocking_inner_perm(o, b, rng) for o, b in _group_axis_splits(seed, op_to_group)]
        new_groups = [
            replace(
                old,
                dim_order=dim_orders[gi],
                tensor_placements=_sample_group_placements(seed, gi, dim_orders[gi], forced_full, rng),
            )
            for gi, old in enumerate(seed_groups)
        ]
        candidate = replace(ir, fusion_groups=new_groups)
        if validate(candidate, tensor_to_groups):
            return candidate
    raise RuntimeError(f"No valid IR found after {max_tries} samples")


def _sample_group_placements(
    ir: "KernelIR", group_idx: int, dim_order: list[str], forced_full: set[tuple[int, str, str]], rng: random.Random
) -> dict[tuple[str, str, str], str]:
    """Draw one group's ``tensor_placements`` map.

    For each SBUF-backed tensor this group touches, pick a
    single emission depth uniformly in ``[0, 2n]`` and then pick
    each dim's tier from the set of tiers whose interval contains
    that depth. This guarantees per-tensor depth feasibility by
    construction. Entries forced to ``full`` by the cross-group
    rule are pinned up front.
    """
    da = ir.dim_analysis
    seed_group = ir.fusion_groups[group_idx]
    touched = {name for (_kind, name, _d) in seed_group.tensor_placements}
    pos = {d: i for i, d in enumerate(dim_order)}
    n = len(dim_order)
    placements: dict[tuple[str, str, str], str] = {}
    for name in touched:
        tinfo = da.tensors[name]
        depth = rng.randint(0, 2 * n) if n > 0 else 0
        for d in tinfo.dim_ids:
            key = ("sbuf", name, d)
            if (group_idx, name, d) in forced_full:
                placements[key] = "full"
                continue
            if d in pos:
                placements[key] = _pick_tier_containing(pos[d], n, depth, rng)
            else:
                placements[key] = rng.choice(TIERS)
    return placements


def _pick_tier_containing(position: int, n: int, depth: int, rng: random.Random) -> str:
    """Pick a tier uniformly from those whose interval at ``position`` contains ``depth``.

    Falls back to a uniform draw over all tiers if no tier's
    interval contains ``depth`` (cannot happen because per_tile,
    per_block, full together cover ``[0, 2n]``, but guards
    against future tier changes).
    """
    candidates = [
        t for t in TIERS if tier_depth_range(t, position, n)[0] <= depth <= tier_depth_range(t, position, n)[1]
    ]
    return rng.choice(candidates) if candidates else rng.choice(TIERS)


def _group_axis_splits(ir: "KernelIR", op_to_group: dict[int, int]) -> list[tuple[list[str], set[str]]]:
    """For each fusion group return ``(ordered_dims, blocking_dims)``.

    ``blocking_dims`` is the union over every op in the group of
    its concrete blocking dims (see ``op_blocking_dims``). The
    order list is the group's seed ``dim_order``; the sampler
    permutes only within the non-blocking and blocking halves.
    """
    da = ir.dim_analysis
    splits: list[tuple[list[str], set[str]]] = []
    for group in ir.fusion_groups:
        order = list(group.dim_order)
        blocking: set[str] = set()
        for op_idx in group.op_indices:
            op_cls = ir.op_graph.op_classes[op_idx]
            blocking |= op_blocking_dims(op_cls, da.per_op_axis_maps[op_idx]) & set(order)
        splits.append((order, blocking))
    return splits


def _rand_blocking_inner_perm(order: list[str], blocking: set[str], rng: random.Random) -> list[str]:
    """Uniform random permutation restricted to blocking-innermost orderings.

    Non-blocking dims are shuffled among themselves (outer
    positions), blocking dims among themselves (inner positions).
    """
    non_blocking = [d for d in order if d not in blocking]
    blocking_list = [d for d in order if d in blocking]
    rng.shuffle(non_blocking)
    rng.shuffle(blocking_list)
    return non_blocking + blocking_list


def _forced_full_pairs(ir: "KernelIR", tensor_to_groups: dict[str, set[int]]) -> set[tuple[int, str, str]]:
    """Return ``(group, tensor, dim)`` triples forced to ``full`` by cross-group rules.

    A tensor shared across multiple fusion groups (either
    producer→consumer edges or a kernel input feeding several
    consumer groups) must have ``tier=full`` in every group that
    touches it, on every dim the tensor carries that also appears
    in that group's ``dim_order``. Otherwise one group's iteration
    can overwrite slots another group hasn't read yet, or an
    under-loaded SBUF feeds downstream groups stale data.
    """
    da = ir.dim_analysis
    group_dim_sets = [set(group.dim_order) for group in ir.fusion_groups]
    forced: set[tuple[int, str, str]] = set()
    for tensor_name, groups in tensor_to_groups.items():
        if len(groups) < 2:
            continue
        tensor_dims = set(da.tensors[tensor_name].dim_ids)
        for gi in groups:
            for d in group_dim_sets[gi] & tensor_dims:
                forced.add((gi, tensor_name, d))
    return forced


def _group_dims(group: list[int], da: DimAnalysis, op_graph: OpGraph) -> list[str]:
    """Return every dim any op in the group touches, sorted by ``dim_id``."""
    dims: set[str] = set()
    for op_idx in group:
        for tensor_name in op_graph.op_tensor_names(op_idx):
            if tensor_name in da.tensors:
                dims.update(da.tensors[tensor_name].dim_ids)
    return sorted(dims)


def build_ir(
    func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]], seed: int | None = None
) -> KernelIR:
    """Construct the initial KernelIR from a math function.

    Runs dimension analysis and graph analysis, then sets all
    rendering parameters to their default (naive) values. The
    default is maximally unfused: one singleton fusion group per
    op, each owning a complete loop nest over every dim its op
    touches. Each group's ``dim_order`` and ``tensor_placements``
    are drawn via rejection sampling from a ``random.Random(seed)``
    RNG; pass an integer ``seed`` for reproducibility.

    Args:
        func: Math function using NKIOp classes.
        input_specs: ``{param_name: (shape, dtype)}``.
        seed: RNG seed for the sampler. ``None`` (default) uses a
            fresh unseeded RNG, so every call returns a different
            valid IR.

    Returns:
        KernelIR with default rendering parameters.
    """
    da = analyze_dims(func, input_specs)
    graph = build_op_graph(func, input_specs)

    num_ops = len(graph.op_classes)
    ltiles_per_block: dict[str, int] = {dim_id: 1 for dim_id in da.dims}
    tensor_kinds = tensor_buffers(da, graph)
    sbuf_tensors = {name for name, kinds in tensor_kinds.items() if "sbuf" in kinds}

    groups: list[FusionGroup] = []
    for op_idx in range(num_ops):
        op_indices = [op_idx]
        touched = _group_touched_tensors(graph, da, op_indices)
        groups.append(
            FusionGroup(
                op_indices=op_indices,
                dim_order=_group_dims(op_indices, da, graph),
                buffer_degrees=_init_group_buffer_degrees(da, graph, op_indices, touched, tensor_kinds),
                tensor_placements=_init_group_tensor_placements(da, touched, sbuf_tensors),
            )
        )

    ir = KernelIR(dim_analysis=da, op_graph=graph, ltiles_per_block=ltiles_per_block, fusion_groups=groups)
    return sample_valid_ir(ir, random.Random(seed))
