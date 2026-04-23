"""Generic pattern-rewrite driver for ``(KernelContext, KernelGraph)``."""

from typing import Protocol, runtime_checkable

from nkigym.kernel_ir.context.context import KernelContext
from nkigym.kernel_ir.graph.graph import KernelGraph


@runtime_checkable
class MatchInstance(Protocol):
    """Marker type for pattern match instances."""


@runtime_checkable
class PatternRewrite(Protocol):
    """Protocol for one kind of graph rewrite.

    Patterns operate on the ``(context, graph)`` pair and
    return a fresh pair on apply. The driver is pure: it
    threads the pair through ``match`` / ``apply`` without
    mutation.
    """

    name: str

    def match(self, context: KernelContext, graph: KernelGraph) -> list[MatchInstance]:
        """Return zero or more ``MatchInstance`` describing applications of this pattern."""
        ...

    def apply(
        self, context: KernelContext, graph: KernelGraph, instance: MatchInstance
    ) -> tuple[KernelContext, KernelGraph]:
        """Rewrite ``(context, graph)`` for one match; return the mutated pair."""
        ...


def apply_rewrites_until_fixpoint(
    context: KernelContext, graph: KernelGraph, patterns: list[PatternRewrite], max_iterations: int = 64
) -> tuple[KernelContext, KernelGraph]:
    """Run every pattern until a full pass over all patterns yields zero matches."""
    current_ctx, current_graph = context, graph
    for _ in range(max_iterations):
        any_applied = False
        for pattern in patterns:
            while True:
                instances = pattern.match(current_ctx, current_graph)
                if not instances:
                    break
                current_ctx, current_graph = pattern.apply(current_ctx, current_graph, instances[0])
                any_applied = True
        if not any_applied:
            return current_ctx, current_graph
    raise RuntimeError(
        f"pattern-rewrite driver did not reach fixpoint after {max_iterations} passes — "
        "check for oscillating patterns"
    )
