"""CPU simulation and SSH profiling for NKI kernels."""

from nkigym.profile.api import profile
from nkigym.profile.simulate_nki import simulate_fp32

__all__ = ["profile", "simulate_fp32"]
