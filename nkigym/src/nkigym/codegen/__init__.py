"""Code generation for NKI Gym.

Available codegen passes:
- roll_loops: Detects repeating patterns and rolls them into for loops.
- lower_to_nki: Lowers GymProgram IR to NKI kernel source code.
"""

from nkigym.codegen.context import _LoweringContext, get_kwarg
from nkigym.codegen.gym_to_nki import lower_to_nki
from nkigym.codegen.loop_rolling import roll_loops

__all__ = ["_LoweringContext", "get_kwarg", "lower_to_nki", "roll_loops"]
