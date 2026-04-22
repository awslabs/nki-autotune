"""Graph reachability helper for fusion-group merge legality.

The old stochastic merge sampler has been retired; merges are
now enumerated at graph-variant time by the ``MergeComposites``
``PatternRewrite`` in ``rewrites/merge_composites.py``. This
module keeps only the small reachability primitive the rewrite
uses to check R1 convexity.
"""

from nkigym.kernel_ir.graph.graph import KernelGraph


def compute_reachability(graph: KernelGraph) -> list[set[int]]:
    """Return transitive-closure forward reachability over groups.

    ``reach[u]`` is the set of group indices reachable from ``u``
    via ``graph.edges`` (inclusive of ``u``). Used by
    ``MergeComposites`` to test whether merging two groups
    preserves R1 — no external group is trapped on a DAG path
    between them.
    """
    n = len(graph.groups)
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
