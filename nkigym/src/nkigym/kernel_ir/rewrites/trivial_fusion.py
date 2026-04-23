"""Trivial-fusion pattern: merge two groups whose shared dims are all parallel in the producer.

``A`` → ``B`` can be merged iff for every dim ``d`` shared
between the two groups, role(``d``, ``A``) is ``PARALLEL`` —
i.e. no op in ``A`` accumulates on ``d`` to produce its output.
This means the shared tensor carries ``d`` fully materialized
per iteration, so merging ``A`` and ``B`` into one ``d``-loop
is semantically identical to running them in sequence.

The rule covers cases (1) PAR × PAR, (2) PAR × SEQ, (3) PAR ×
ACC. Cases with non-PAR in ``A`` are left alone: SEQ/ACC in
``A`` means the shared tensor is mid-reduction during ``A``'s
loop, and a merged loop would have ``B`` read partial state.
Those cases are handled separately by ``OnlineFusionPattern``
when the accumulation structure matches.

Registered in ``REWRITES`` and sampled per-draw alongside the
other rewrites in a single stochastic loop.
"""

from collections import deque
from dataclasses import dataclass

from nkigym.kernel_ir.context.context import DimRole, KernelContext
from nkigym.kernel_ir.graph.fusion_group import FusionGroup
from nkigym.kernel_ir.graph.graph import KernelGraph, rebuild_edges
from nkigym.kernel_ir.sampler.partition import compute_reachability
from nkigym.ops.base import NKIOp


@dataclass(frozen=True)
class _Match:
    """One trivial-merge candidate: producer group index and consumer group index."""

    producer: int
    consumer: int


class TrivialFusion:
    """Merge a producer→consumer pair whose shared dims are all parallel in the producer."""

    name = "trivial_fusion"

    def match(self, context: KernelContext, graph: KernelGraph) -> list[_Match]:
        """Return every legal producer→consumer pair."""
        n = len(graph.groups)
        reach = compute_reachability(graph)
        matches: list[_Match] = []
        for producer_gi, consumer_gi in _producer_consumer_edges(context, graph):
            if not _shared_dims_all_parallel(context, graph, producer_gi, consumer_gi):
                continue
            if not _merge_preserves_convexity({producer_gi}, {consumer_gi}, n, reach):
                continue
            matches.append(_Match(producer=producer_gi, consumer=consumer_gi))
        return matches

    def apply(self, context: KernelContext, graph: KernelGraph, instance: _Match) -> tuple[KernelContext, KernelGraph]:
        """Merge the two groups, producer-before-consumer."""
        return context, _merge_two_groups(context, graph, instance.producer, instance.consumer)


def _producer_consumer_edges(context: KernelContext, graph: KernelGraph) -> list[tuple[int, int]]:
    """Return every ``(producer_gi, consumer_gi)`` pair connected by at least one tensor."""
    producer_of_tensor: dict[str, int] = {}
    for gi, group in enumerate(graph.groups):
        for op in group.ops:
            for name in context.op_outputs.get(op, []):
                producer_of_tensor[name] = gi
    seen: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []
    for gi, group in enumerate(graph.groups):
        for op in group.ops:
            for name in context.op_inputs.get(op, {}).values():
                src = producer_of_tensor.get(name)
                if src is None or src == gi:
                    continue
                key = (src, gi)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(key)
    return edges


def _shared_dims_all_parallel(context: KernelContext, graph: KernelGraph, producer_gi: int, consumer_gi: int) -> bool:
    """True iff every dim shared between producer group and consumer group is parallel in the producer."""
    producer_group = graph.groups[producer_gi]
    consumer_group = graph.groups[consumer_gi]
    shared = _group_touched_dims(context, producer_group) & _group_touched_dims(context, consumer_group)
    producer_blocking = _group_blocking_dims(context, producer_group)
    return not (shared & producer_blocking)


def _group_touched_dims(context: KernelContext, group: FusionGroup) -> set[str]:
    """Union of dim IDs across every tensor any op in the group touches."""
    result: set[str] = set()
    for op in group.ops:
        tensor_names = list(context.op_inputs.get(op, {}).values()) + list(context.op_outputs.get(op, []))
        for name in tensor_names:
            tinfo = context.logical_tensors.get(name)
            if tinfo is not None:
                result.update(tinfo.dim_ids)
    return result


def _group_blocking_dims(context: KernelContext, group: FusionGroup) -> set[str]:
    """Union of every op's SERIAL blocking dims in the group.

    ACCUMULATION dims do **not** block: by construction they
    produce valid incremental state per iteration, so a consumer
    in a downstream group can read the running buffer inside the
    same loop. Only SERIAL dims require the producer's reduction
    to complete before consumers may read.
    """
    result: set[str] = set()
    for op in group.ops:
        for dim_id in context.op_blocking_dims.get(op, set()):
            dim_info = context.dimensions.get(dim_id)
            if dim_info is None or dim_info.role is DimRole.SERIAL:
                result.add(dim_id)
    return result


def _merge_two_groups(context: KernelContext, graph: KernelGraph, gi: int, gj: int) -> KernelGraph:
    """Return a new ``KernelGraph`` with groups ``gi`` and ``gj`` merged, producer-before-consumer."""
    lo, hi = (gi, gj) if gi < gj else (gj, gi)
    raw_ops = list(graph.groups[lo].ops) + list(graph.groups[hi].ops)
    merged_ops = _toposort_ops(context, raw_ops)
    new_groups: list[FusionGroup] = []
    for idx, grp in enumerate(graph.groups):
        if idx == lo:
            new_groups.append(FusionGroup(ops=merged_ops))
        elif idx == hi:
            continue
        else:
            new_groups.append(FusionGroup(ops=list(grp.ops)))
    new_graph = KernelGraph(groups=new_groups)
    rebuild_edges(new_graph, context)
    return new_graph


def _merge_preserves_convexity(a: set[int], b: set[int], n: int, reach: list[set[int]]) -> bool:
    """R1: ``a ∪ b`` is convex in the DAG induced by ``reach``."""
    members = a | b
    ok = True
    for w in range(n):
        if w in members:
            continue
        if any(w in reach[u] for u in members) and any(m in reach[w] for m in members):
            ok = False
            break
    return ok


def _toposort_ops(context: KernelContext, ops: list[NKIOp]) -> list[NKIOp]:
    """Return ``ops`` reordered so each producer precedes its consumers (Kahn's algorithm).

    Ties break on original input order — preserves the sequence
    where there is no producer/consumer dependency.
    """
    idx_of = {id(op): i for i, op in enumerate(ops)}
    producer_of: dict[str, NKIOp] = {}
    for op in ops:
        for name in context.op_outputs.get(op, []):
            producer_of[name] = op
    in_degree: dict[int, int] = {id(op): 0 for op in ops}
    adjacency: dict[int, list[NKIOp]] = {id(op): [] for op in ops}
    for consumer in ops:
        for name in context.op_inputs.get(consumer, {}).values():
            producer = producer_of.get(name)
            if producer is None or producer is consumer:
                continue
            adjacency[id(producer)].append(consumer)
            in_degree[id(consumer)] += 1
    ready: deque[NKIOp] = deque(op for op in ops if in_degree[id(op)] == 0)
    ordered: list[NKIOp] = []
    while ready:
        current = _pop_lowest_idx(ready, idx_of)
        ordered.append(current)
        for successor in adjacency[id(current)]:
            in_degree[id(successor)] -= 1
            if in_degree[id(successor)] == 0:
                ready.append(successor)
    if len(ordered) != len(ops):
        raise ValueError("cycle detected while topologically sorting merged ops")
    return ordered


def _pop_lowest_idx(ready: deque[NKIOp], idx_of: dict[int, int]) -> NKIOp:
    """Remove and return the op in ``ready`` with the smallest original index."""
    best_pos = 0
    best_key = idx_of[id(ready[0])]
    for pos in range(1, len(ready)):
        candidate_key = idx_of[id(ready[pos])]
        if candidate_key < best_key:
            best_pos = pos
            best_key = candidate_key
    chosen = ready[best_pos]
    del ready[best_pos]
    return chosen
