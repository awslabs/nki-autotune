"""Online-fusion as a ``PatternRewrite``."""

from dataclasses import dataclass

from nkigym.kernel_ir.ir import KernelIR
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

    def match(self, ir: KernelIR) -> list[_Match]:
        """Return every supported candidate visible in the current ir."""
        candidates = detect_online_fusion(ir)
        supported = {"rsqrt_then_mul", "exp_bias"}
        extend_supported = supported | {"passthrough_mul"}
        return [
            _Match(candidate=c)
            for c in candidates
            if self._role_ok(c, supported, extend_supported) and self._is_valid_shape(c)
        ]

    @staticmethod
    def _role_ok(candidate: OnlineFusionCandidate, create_set: frozenset | set, extend_set: frozenset | set) -> bool:
        """Supported-role gate differs for create vs extend: extend inherits SCALE_SPEC, accepting passthrough_mul."""
        return candidate.scale_role in (extend_set if candidate.mode == "extend" else create_set)

    def apply(self, ir: KernelIR, instance: _Match) -> tuple[KernelIR, KernelIR]:
        """Apply one candidate's rewrite."""
        return rewrite_one_candidate(ir, instance.candidate)

    @staticmethod
    def _is_valid_shape(candidate: OnlineFusionCandidate) -> bool:
        """Create mode: neither X nor accs are composites. Extend mode: X is a composite, accs aren't."""
        acc_touches_composite = any(isinstance(op, NKIOnlineFusionChain) for op in candidate.accumulator_ops)
        x_is_composite = isinstance(candidate.x_op, NKIOnlineFusionChain)
        return not acc_touches_composite and (candidate.mode == "extend") == x_is_composite
