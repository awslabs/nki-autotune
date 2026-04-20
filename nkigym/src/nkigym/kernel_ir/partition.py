"""Fusion-partition sampling under R1 (convexity) + R2 (blocking barrier) + renderer-capability gate.

Draws a partition of ``op_graph.op_classes`` into fusion groups by
stochastic pairwise merging from the singleton partition, gated by:

* R1 (convexity): ``ops(M)`` must be a convex subset of the op
  DAG. For every op ``w`` outside ``M``, ``w`` must not lie on a
  DAG path between two members of ``M``.
* R2 (blocking barrier): for every producer→consumer edge
  ``u → v`` with both in ``M``,
  ``blocking_dims(u) ∩ dims(v) = ∅``. ``v``'s emission scope must
  sit outside ``u``'s blocking loops so ``u``'s store has fired
  before ``v`` reads it.
* Renderer-capability gate: reject intra-group edges where the
  producer is a blocking PSUM op.

Output is a list of op-index groups in topological order.
"""

import random

from nkigym.kernel_ir.dim_analysis import DimAnalysis, op_blocking_dims
from nkigym.kernel_ir.op_graph import OpGraph


def compute_reachability(graph: OpGraph) -> list[set[int]]:
    """Return transitive-closure reachability: ``reach[u]`` = ops reachable from ``u``.

    Includes ``u`` itself. Used for the R1 convexity check.
    """
    n = len(graph.op_classes)
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for producer, consumer, _tensor, _role in graph.edges:
        adjacency[producer].append(consumer)
    reach: list[set[int]] = [set() for _ in range(n)]
    for start in range(n):
        stack = [start]
        visited = reach[start]
        visited.add(start)
        while stack:
            u = stack.pop()
            for v in adjacency[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
    return reach


def op_dims_of(graph: OpGraph, da: DimAnalysis, op_idx: int) -> set[str]:
    """Return every dim the op touches via its input + output tensors."""
    dims: set[str] = set()
    for name in graph.op_tensor_names(op_idx):
        if name in da.tensors:
            dims.update(da.tensors[name].dim_ids)
    return dims


def _is_convex(ops: set[int], n: int, reach: list[set[int]]) -> bool:
    """Check R1: no external op lies on a DAG path between two members of ``ops``.

    An external op ``w`` violates convexity iff some member reaches
    ``w`` AND ``w`` reaches some member.
    """
    external_on_path = (
        any(w in reach[u] for u in ops) and any(m in reach[w] for m in ops) for w in range(n) if w not in ops
    )
    return not any(external_on_path)


def _violates_blocking(ops: set[int], graph: OpGraph, da: DimAnalysis, op_dims: list[set[str]]) -> bool:
    """Check R2 + renderer-capability: any intra-group edge ``u → v`` where either
    ``blocking_dims(u) ∩ dims(v) ≠ ∅`` (math-invalid), or ``u`` is a PSUM producer
    with non-empty ``BLOCKING_AXES`` (renderer can't hoist ``v`` out of ``u``'s blocking scope).
    """
    return any(
        _edge_blocks(graph, da, producer, consumer, op_dims)
        for producer, consumer, _tensor, _role in graph.edges
        if producer in ops and consumer in ops
    )


def _edge_blocks(graph: OpGraph, da: DimAnalysis, producer: int, consumer: int, op_dims: list[set[str]]) -> bool:
    """True iff edge ``producer → consumer`` violates R2 or the renderer-capability gate."""
    producer_cls = graph.op_classes[producer]
    blocking = op_blocking_dims(producer_cls, da.per_op_axis_maps[producer])
    r2_violation = bool(blocking & op_dims[consumer])
    capability_violation = producer_cls.ISA_LOC == "psum" and bool(blocking)
    return r2_violation or capability_violation


def _legal_merge(
    a: set[int], b: set[int], graph: OpGraph, da: DimAnalysis, reach: list[set[int]], op_dims: list[set[str]], n: int
) -> bool:
    """Test whether merging ``a`` and ``b`` preserves R1 + R2 + renderer capability."""
    merged = a | b
    return _is_convex(merged, n, reach) and not _violates_blocking(merged, graph, da, op_dims)


def sample_partition(
    graph: OpGraph,
    da: DimAnalysis,
    rng: random.Random,
    p_merge: float = 0.7,
    reach: list[set[int]] | None = None,
    op_dims: list[set[str]] | None = None,
) -> list[list[int]]:
    """Draw a partition of the op set via stochastic pairwise merging.

    Starts from the singleton partition and repeatedly selects a
    random legal merge (under R1 + R2) with probability ``p_merge``
    per step. Stops when no legal merges remain or the Bernoulli
    draw breaks. Groups are returned in topological order with ops
    inside each group sorted by index. ``reach`` and ``op_dims``
    depend only on ``(graph, da)`` — callers in hot paths should
    compute them once and pass them in.
    """
    n = len(graph.op_classes)
    reach_final: list[set[int]] = compute_reachability(graph) if reach is None else reach
    op_dims_final: list[set[str]] = (
        [op_dims_of(graph, da, op_idx) for op_idx in range(n)] if op_dims is None else op_dims
    )
    groups: list[set[int]] = [{i} for i in range(n)]

    while True:
        legal_pairs = [
            (i, j)
            for i in range(len(groups))
            for j in range(i + 1, len(groups))
            if _legal_merge(groups[i], groups[j], graph, da, reach_final, op_dims_final, n)
        ]
        if not legal_pairs:
            break
        if rng.random() >= p_merge:
            break
        i, j = rng.choice(legal_pairs)
        merged = groups[i] | groups[j]
        groups = [g for k, g in enumerate(groups) if k not in (i, j)]
        groups.append(merged)

    ordered_op_lists = [sorted(g) for g in groups]
    topo_order = graph.toposort_groups(ordered_op_lists)
    return [ordered_op_lists[gi] for gi in topo_order]
