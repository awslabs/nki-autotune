"""Online-fusion as a ``PatternRewrite``."""

from dataclasses import dataclass

from nkigym.kernel_ir.context.context import KernelContext
from nkigym.kernel_ir.graph.graph import KernelGraph
from nkigym.kernel_ir.rewrites.online_fusion_detect import OnlineFusionCandidate, detect_online_fusion
from nkigym.kernel_ir.rewrites.online_fusion_rewrite import rewrite_one_candidate
from nkigym.ops.online_fusion_chain import NKIOnlineFusionChain


@dataclass(frozen=True)
class _Match:
    """Internal match instance — wraps an ``OnlineFusionCandidate``."""

    candidate: OnlineFusionCandidate


class OnlineFusionPattern:
    """PatternRewrite implementation for online-fusion rewrites."""

    name = "online_fusion"

    def match(self, context: KernelContext, graph: KernelGraph) -> list[_Match]:
        """Return every supported candidate visible in the current graph."""
        candidates = detect_online_fusion(context, graph)
        supported = {"rsqrt_then_mul", "exp_bias"}
        return [_Match(candidate=c) for c in candidates if c.scale_role in supported and not self._touches_composite(c)]

    def apply(self, context: KernelContext, graph: KernelGraph, instance: _Match) -> tuple[KernelContext, KernelGraph]:
        """Apply one candidate's rewrite."""
        return rewrite_one_candidate(context, graph, instance.candidate)

    @staticmethod
    def _touches_composite(candidate: OnlineFusionCandidate) -> bool:
        """True iff X or any accumulator is already a composite node."""
        all_ops = (candidate.x_op, *candidate.accumulator_ops)
        return any(isinstance(op, NKIOnlineFusionChain) for op in all_ops)
