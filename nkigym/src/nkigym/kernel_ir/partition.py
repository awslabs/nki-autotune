"""Fusion-partition sampling under R1 (convexity) only.

Draws a partition of ``op_graph.op_classes`` into fusion groups by
stochastic pairwise merging from the singleton partition. The one
partition-time rule is R1 (convexity): ``ops(M)`` must be a convex
subset of the op DAG — for every op ``w`` outside ``M``, ``w``
must not lie on a DAG path between two members of ``M``.

R2 (blocking barrier) and renderer-capability are trip-count-
dependent and draw-specific (they depend on the sampled
``dim_order``, ``ltiles_per_block``, and ``tensor_placements``).
They are enforced at IR-validation time via
``validate._check_emission_feasibility`` rather than here.

Output is a list of op-index groups in topological order.
"""

import random

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


def _is_convex(ops: set[int], n: int, reach: list[set[int]]) -> bool:
    """Check R1: no external op lies on a DAG path between two members of ``ops``.

    An external op ``w`` violates convexity iff some member reaches
    ``w`` AND ``w`` reaches some member.
    """
    external_on_path = (
        any(w in reach[u] for u in ops) and any(m in reach[w] for m in ops) for w in range(n) if w not in ops
    )
    return not any(external_on_path)


def _legal_merge(a: set[int], b: set[int], reach: list[set[int]], n: int) -> bool:
    """Test whether merging ``a`` and ``b`` preserves R1 (convexity).

    Trip-count-dependent rules (R2, emission-slot feasibility) are
    enforced at IR-validation time against the full drawn state.
    """
    return _is_convex(a | b, n, reach)


def sample_partition(
    graph: OpGraph, rng: random.Random, p_merge: float = 0.7, reach: list[set[int]] | None = None
) -> list[list[int]]:
    """Draw a partition of the op set via stochastic pairwise merging.

    Starts from the singleton partition and repeatedly selects a
    random legal merge (under R1) with probability ``p_merge`` per
    step. Stops when no legal merges remain or the Bernoulli draw
    breaks. Groups are returned in topological order with ops inside
    each group sorted by index. ``reach`` depends only on ``graph``
    — callers in hot paths should compute it once and pass it in.
    """
    n = len(graph.op_classes)
    reach_final: list[set[int]] = compute_reachability(graph) if reach is None else reach
    groups: list[set[int]] = [{i} for i in range(n)]

    while True:
        legal_pairs = [
            (i, j)
            for i in range(len(groups))
            for j in range(i + 1, len(groups))
            if _legal_merge(groups[i], groups[j], reach_final, n)
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
