"""Code generation for NKI Gym.

Available codegen passes:
- roll_loops: Detects repeating patterns and rolls them into for loops.
- lower_to_nki: Lowers GymProgram IR to NKI kernel source code.
"""

from nkigym.program_to_nki.gym_to_nki import lower_to_nki
from nkigym.program_to_nki.loop_rolling import roll_loops

__all__ = ["lower_to_nki", "roll_loops"]
