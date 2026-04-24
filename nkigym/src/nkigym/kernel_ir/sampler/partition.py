"""Graph reachability helper for fusion-group merge legality.

Small primitive used by fusion rewrites (``LoopFusion``) to
check R1 convexity when merging two groups.
"""

from nkigym.kernel_ir.ir import KernelIR


def compute_reachability(ir: KernelIR) -> list[set[int]]:
    """Return transitive-closure forward reachability over groups.

    ``reach[u]`` is the set of group indices reachable from ``u``
    via ``ir.edges`` (inclusive of ``u``). Used by
    ``LoopFusion`` to test whether merging two groups preserves
    R1 — no external group is trapped on a DAG path between them.
    """
    n = len(ir.groups)
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for producer, consumer, _tensor, _role in ir.edges:
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
