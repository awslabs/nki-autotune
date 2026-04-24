"""Inner-layer sampler + ir-reachability helper used by LoopFusion."""

from nkigym.kernel_ir.sampler.partition import compute_reachability
from nkigym.kernel_ir.sampler.sampler import sample_valid_ir

__all__ = ["compute_reachability", "sample_valid_ir"]
