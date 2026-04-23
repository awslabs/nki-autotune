"""Pattern-rewrite driver + every registered pattern."""

from nkigym.kernel_ir.rewrites.load_transpose_pattern import LoadTransposePattern
from nkigym.kernel_ir.rewrites.online_fusion_pattern import OnlineFusionPattern
from nkigym.kernel_ir.rewrites.pattern_rewrite import MatchInstance, PatternRewrite, apply_rewrites_until_fixpoint
from nkigym.kernel_ir.rewrites.trivial_fusion import TrivialFusion

__all__ = [
    "LoadTransposePattern",
    "MatchInstance",
    "OnlineFusionPattern",
    "PatternRewrite",
    "TrivialFusion",
    "apply_rewrites_until_fixpoint",
]
