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
        skip_ops = _collect_skip_op_ids(graph)
        return [
            _Match(candidate=c)
            for c in candidates
            if c.scale_role in supported and not self._touches_composite(c) and not _candidate_hits_skip(c, skip_ops)
        ]

    def apply(self, context: KernelContext, graph: KernelGraph, instance: _Match) -> tuple[KernelContext, KernelGraph]:
        """Apply one candidate's rewrite."""
        return rewrite_one_candidate(context, graph, instance.candidate)

    @staticmethod
    def _touches_composite(candidate: OnlineFusionCandidate) -> bool:
        """True iff X or any accumulator is already a composite node."""
        all_ops = (candidate.x_op, *candidate.accumulator_ops)
        return any(isinstance(op, NKIOnlineFusionChain) for op in all_ops)


def _collect_skip_op_ids(graph: KernelGraph) -> set[int]:
    """Return ``id(op)`` for every op living inside a ``skip_spec``-annotated group."""
    result: set[int] = set()
    for group in graph.groups:
        if group.skip_spec is None:
            continue
        for op in group.ops:
            result.add(id(op))
    return result


def _candidate_hits_skip(candidate: OnlineFusionCandidate, skip_ops: set[int]) -> bool:
    """True iff any op in the candidate lives inside a skip-annotated group — don't rewrite it."""
    return any(id(op) in skip_ops for op in (candidate.x_op, *candidate.accumulator_ops))
