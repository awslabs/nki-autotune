"""Deterministic programmatic synthesis."""

from nkigym.synthesis.artifact import SynthesizedKernel
from nkigym.synthesis.torch_to_nkigym import synthesize_torch_to_nkigym

__all__ = ["SynthesizedKernel", "synthesize_torch_to_nkigym"]
