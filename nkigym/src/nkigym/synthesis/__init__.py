"""Synthesis: translate numpy references into nkigym source via LLM calls."""

from nkigym.synthesis.numpy_to_nkigym import compile_numpy_to_nkigym
from nkigym.synthesis.simulate_nki import simulate_fp32

__all__ = ["compile_numpy_to_nkigym", "simulate_fp32"]
