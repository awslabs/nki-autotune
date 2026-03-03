"""NKI Gym IR: GymProgram and GymStatement types."""

from nkigym.ir.tensor import TensorRef, full_slices, ref_name
from nkigym.ir.types import GymProgram, GymStatement

__all__ = ["GymProgram", "GymStatement", "TensorRef", "full_slices", "ref_name"]
