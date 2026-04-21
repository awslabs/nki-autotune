"""Online-fusion as a ``PatternRewrite``.

Wraps the detector + rewrite into the protocol the pattern driver
expects. ``match`` returns every un-rewritten online-fusion
candidate; ``apply`` rewrites one candidate into a composite node.
The driver re-runs ``match`` after each ``apply``, so chains where
a post-rewrite composite enables another online fusion fire
naturally on the next iteration.
"""

from dataclasses import dataclass

from nkigym.kernel_ir.dim_analysis import DimAnalysis
from nkigym.kernel_ir.online_fusion_detect import OnlineFusionCandidate, detect_online_fusion
from nkigym.kernel_ir.online_fusion_rewrite import rewrite_one_candidate
from nkigym.kernel_ir.op_graph import OpGraph
from nkigym.ops.online_fusion_chain import NKIOnlineFusionChain


@dataclass(frozen=True)
class _Match:
    """Internal match instance — just wraps an ``OnlineFusionCandidate``."""

    candidate: OnlineFusionCandidate


class OnlineFusionPattern:
    """PatternRewrite implementation for online-fusion rewrites.

    The detector runs on ``(da, graph)`` and returns every
    (X, accumulators) grouping. Matches on composite nodes are
    filtered out so the driver doesn't re-fuse an already-fused
    chain into itself. Supported scale roles are delegated to
    ``rewrite_one_candidate``; unsupported roles are skipped.
    """

    name = "online_fusion"

    def match(self, da: DimAnalysis, graph: OpGraph) -> list[_Match]:
        """Return every supported candidate visible in the current graph.

        Composite nodes produced by earlier rewrites are excluded:
        their class-level ``BLOCKING_AXES`` is empty by
        construction, so the detector wouldn't flag them as X
        anyway, but filtering explicitly guards against future
        composite-produces-blocking-dim cases.
        """
        candidates = detect_online_fusion(da, graph)
        supported = {"rsqrt_then_mul", "exp_bias"}
        return [
            _Match(candidate=c)
            for c in candidates
            if c.scale_role in supported and not self._touches_composite(graph, c)
        ]

    def apply(self, da: DimAnalysis, graph: OpGraph, instance: _Match) -> tuple[DimAnalysis, OpGraph]:
        """Apply one candidate's rewrite."""
        return rewrite_one_candidate(da, graph, instance.candidate)

    @staticmethod
    def _touches_composite(graph: OpGraph, candidate: OnlineFusionCandidate) -> bool:
        """True iff X or any accumulator is already a composite node."""
        touched_idx = {candidate.x_op_idx, *candidate.accumulator_op_indices}
        return any(issubclass(graph.op_classes[i], NKIOnlineFusionChain) for i in touched_idx)
