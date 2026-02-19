"""Code generation for NKI Gym.

Available codegen passes:
- roll_loops: Detects repeating patterns and rolls them into for loops.
"""

from nkigym.codegen.loop_rolling import roll_loops

__all__ = ["roll_loops"]
