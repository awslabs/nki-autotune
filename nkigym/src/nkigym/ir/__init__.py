"""NKI Gym IR: GymProgram and GymStatement types with source↔program translation."""

from nkigym.ir.parse import program_to_source, source_to_program
from nkigym.ir.tensor import TensorRef, full_slices, ref_name
from nkigym.ir.types import GymProgram, GymStatement

__all__ = [
    "GymProgram",
    "GymStatement",
    "TensorRef",
    "full_slices",
    "program_to_source",
    "ref_name",
    "source_to_program",
]
