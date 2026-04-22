"""``MergeComposites`` — merge two adjacent fusion groups into one.

Third graph rewrite (alongside ``LoadTransposePattern`` and
``OnlineFusionPattern``). Matches every pair of groups whose
merge preserves R1 (convexity — no external group is trapped on
a DAG path between the two) AND R2 (no reducer producer with a
multi-chunk reduction loop can share a group with a consumer of
its reduced output — the consumer would read a PARTIAL running
value). Each ``apply`` produces one merged graph with
``merged_ops`` topologically sorted so producers precede consumers.
"""

from collections import deque
from dataclasses import dataclass

from nkigym.kernel_ir.context.context import KernelContext
from nkigym.kernel_ir.graph.fusion_group import FusionGroup
from nkigym.kernel_ir.graph.graph import KernelGraph, rebuild_edges
from nkigym.kernel_ir.sampler.partition import compute_reachability
from nkigym.ops.base import NKIOp


@dataclass(frozen=True)
class _Match:
    """One merge candidate: the two group indices to combine."""

    lo: int
    hi: int


class MergeComposites:
    """Merge two convex-legal fusion groups into one."""

    name = "merge_composites"

    def match(self, context: KernelContext, graph: KernelGraph) -> list[_Match]:
        """Return every ``(lo, hi)`` pair whose merge preserves R1 + R2."""
        n = len(graph.groups)
        reach = compute_reachability(graph)
        matches: list[_Match] = []
        for i in range(n):
            for j in range(i + 1, n):
                if not _merge_preserves_convexity({i}, {j}, n, reach):
                    continue
                if _crosses_reduction_boundary(context, graph, i, j):
                    continue
                matches.append(_Match(lo=i, hi=j))
        return matches

    def apply(self, context: KernelContext, graph: KernelGraph, instance: _Match) -> tuple[KernelContext, KernelGraph]:
        """Merge groups ``lo`` and ``hi`` into one; keep all others unchanged."""
        raw_ops = list(graph.groups[instance.lo].ops) + list(graph.groups[instance.hi].ops)
        merged_ops = _toposort_ops(context, raw_ops)
        new_groups: list[FusionGroup] = []
        for gi, grp in enumerate(graph.groups):
            if gi == instance.lo:
                new_groups.append(FusionGroup(ops=merged_ops))
            elif gi == instance.hi:
                continue
            else:
                new_groups.append(FusionGroup(ops=list(grp.ops)))
        new_graph = KernelGraph(groups=new_groups)
        rebuild_edges(new_graph, context)
        return context, new_graph


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


def _crosses_reduction_boundary(context: KernelContext, graph: KernelGraph, gi: int, gj: int) -> bool:
    """True iff merging groups ``gi`` and ``gj`` would put a multi-chunk reducer + its consumer in one loop.

    A multi-chunk reducer is an op with a resolvable
    ``REDUCE_COMBINATOR`` for some output role whose reduction
    dim (blocking dim not in the output's dim_ids) has
    ``num_blocks > 1``. Merging such a reducer with a group
    containing a downstream op that reads the reduced output
    would interleave the running-state write with a consumer
    read of a PARTIAL value — NCC miscompilation / numerical
    drift.
    """
    crosses = _reducer_consumer_cross(context, graph.groups[gi], graph.groups[gj])
    if not crosses:
        crosses = _reducer_consumer_cross(context, graph.groups[gj], graph.groups[gi])
    return crosses


def _reducer_consumer_cross(context: KernelContext, producer_group: FusionGroup, consumer_group: FusionGroup) -> bool:
    """True iff some reducer in ``producer_group`` emits a multi-chunk output consumed by ``consumer_group``."""
    consumer_inputs: set[str] = set()
    for op in consumer_group.ops:
        consumer_inputs.update(context.op_inputs.get(op, {}).values())
    hit = False
    for op in producer_group.ops:
        for tname in _multichunk_reduced_outputs(context, op):
            if tname in consumer_inputs:
                hit = True
                break
        if hit:
            break
    return hit


def _multichunk_reduced_outputs(context: KernelContext, op: NKIOp) -> list[str]:
    """Tensors produced by ``op`` via a multi-chunk reduction (reduction dim's ``num_blocks > 1``)."""
    op_cls = type(op)
    kwargs = context.op_kwargs.get(op, {})
    axis_map = context.op_axis_map.get(op, {})
    blocking = context.op_blocking_dims.get(op, set())
    outputs = context.op_outputs.get(op, [])
    result: list[str] = []
    for role_idx, role in enumerate(op_cls.OUTPUT_AXES):
        if role_idx >= len(outputs):
            continue
        if op_cls.resolve_reduce_combinator(role, kwargs) is None:
            continue
        out_axes = op_cls.OUTPUT_AXES[role]
        out_dims = {axis_map.get(ax) for ax in out_axes if axis_map.get(ax) is not None}
        reduction_dims = blocking - out_dims
        if _any_multichunk(context, reduction_dims):
            result.append(outputs[role_idx])
    return result


def _any_multichunk(context: KernelContext, dim_ids: set[str]) -> bool:
    """True iff any dim's block-loop trip is > 1."""
    found = False
    for d in dim_ids:
        di = context.dimensions.get(d)
        if di is None:
            continue
        num_blocks = di.dim_size // (context.ltiles_per_block.get(d, 1) * di.logical_tile_size)
        if num_blocks > 1:
            found = True
            break
    return found


def _toposort_ops(context: KernelContext, ops: list[NKIOp]) -> list[NKIOp]:
    """Return ``ops`` reordered so each producer precedes its consumers (Kahn's algorithm).

    Ties break on input order — preserves the original sequence
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
