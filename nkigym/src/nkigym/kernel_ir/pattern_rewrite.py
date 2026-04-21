"""Generic pattern-rewrite driver for ``(DimAnalysis, OpGraph)``.

Transformations that modify the op graph structure (e.g., online
fusion, future Load+Transpose → DMATranspose peephole) implement
the ``PatternRewrite`` protocol. The driver applies every pattern
until no further matches are available — a fixpoint — so rewrites
whose output enables further rewrites compose without per-pattern
coordination.

A pattern exposes two operations:

* ``match(da, graph)`` — return a list of ``MatchInstance`` objects
  describing every independent application of the pattern. Each
  instance carries whatever fields the pattern's ``apply`` needs
  to identify the match (typically op indices plus
  pattern-specific metadata).
* ``apply(da, graph, instance)`` — produce the rewritten
  ``(DimAnalysis, OpGraph)`` for ONE match. The driver re-invokes
  ``match`` after each ``apply`` call so newly-exposed matches
  (e.g. a composite that itself qualifies for another pattern)
  are picked up on the next iteration.

Matches across the same pattern are applied one at a time —
applying one may invalidate op indices in others. Between
patterns, the driver iterates until a full pass over every
pattern yields zero matches. The iteration is bounded by
``max_iterations`` to surface oscillating rewrites at test time.
"""

from typing import Protocol, runtime_checkable

from nkigym.kernel_ir.dim_analysis import DimAnalysis
from nkigym.kernel_ir.op_graph import OpGraph


@runtime_checkable
class MatchInstance(Protocol):
    """Marker type — a pattern's ``match`` returns these and its ``apply`` consumes them.

    Concrete ``MatchInstance`` types are pattern-specific (e.g.
    ``OnlineFusionMatch``). The driver treats them opaquely.
    """


@runtime_checkable
class PatternRewrite(Protocol):
    """Protocol for one kind of graph rewrite.

    Implementations must provide:

    * ``name``: human-readable label used in logs and cycle-detection.
    * ``match(da, graph)``: find every application point.
    * ``apply(da, graph, instance)``: rewrite one match.
    """

    name: str

    def match(self, da: DimAnalysis, graph: OpGraph) -> list[MatchInstance]:
        """Return zero or more ``MatchInstance`` describing applications of this pattern."""
        ...

    def apply(self, da: DimAnalysis, graph: OpGraph, instance: MatchInstance) -> tuple[DimAnalysis, OpGraph]:
        """Rewrite ``(da, graph)`` for one match; return the mutated pair."""
        ...


def apply_rewrites_until_fixpoint(
    da: DimAnalysis, graph: OpGraph, patterns: list[PatternRewrite], max_iterations: int = 64
) -> tuple[DimAnalysis, OpGraph]:
    """Run every pattern until a full pass over all patterns yields zero matches.

    Each iteration visits the patterns in order, and for each
    pattern drains its current match list one-by-one (re-running
    ``match`` after every ``apply`` so newly-exposed instances are
    included). If ANY pattern applies at least one rewrite during
    the pass, another pass runs. Stops cleanly when a full pass
    completes without any rewrites.

    Args:
        da: Starting dim analysis.
        graph: Starting op graph.
        patterns: Ordered list of patterns to apply. Order matters
            when patterns can cascade (a later pattern exposing a
            match for an earlier one): the driver re-enters the
            outer loop after any successful rewrite, so earlier
            patterns get another shot.
        max_iterations: Safety bound on total passes. An
            oscillating pair of patterns (add/undo) would otherwise
            loop forever; raising past this bound signals a bug in
            one pattern's match/apply.

    Raises:
        RuntimeError: If ``max_iterations`` passes complete without
            reaching a fixpoint.
    """
    current_da, current_graph = da, graph
    for _ in range(max_iterations):
        any_applied = False
        for pattern in patterns:
            while True:
                instances = pattern.match(current_da, current_graph)
                if not instances:
                    break
                current_da, current_graph = pattern.apply(current_da, current_graph, instances[0])
                any_applied = True
        if not any_applied:
            return current_da, current_graph
    raise RuntimeError(
        f"pattern-rewrite driver did not reach fixpoint after {max_iterations} passes — "
        "check for oscillating patterns"
    )
