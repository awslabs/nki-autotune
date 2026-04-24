"""Pattern-rewrite driver + every registered pattern."""

from nkigym.kernel_ir.rewrites.load_transpose_pattern import LoadTransposePattern
from nkigym.kernel_ir.rewrites.loop_fusion import LoopFusion
from nkigym.kernel_ir.rewrites.online_fusion_pattern import OnlineFusionPattern
from nkigym.kernel_ir.rewrites.pattern_rewrite import MatchInstance, PatternRewrite, apply_rewrites_until_fixpoint

__all__ = [
    "LoadTransposePattern",
    "LoopFusion",
    "MatchInstance",
    "OnlineFusionPattern",
    "PatternRewrite",
    "apply_rewrites_until_fixpoint",
]
