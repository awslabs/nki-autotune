"""Base class for NKI Gym IR transforms.

All transforms follow the analyze-then-transform pattern on GymProgram:

1. ``analyze_ir(program)`` — inspect a GymProgram and return a list of
   individual transform opportunities.
2. ``transform_ir(program, option)`` — apply a single opportunity, returning
   a new GymProgram.

The autotuner calls ``analyze_ir()`` to discover opportunities, selects which
to apply, and calls ``transform_ir()`` for each.
"""

from abc import ABC, abstractmethod
from typing import Any

from nkigym.ir import GymProgram


class Transform(ABC):
    """Base class for nkigym IR transforms.

    Subclasses implement ``analyze_ir`` and ``transform_ir`` which operate
    on GymProgram directly.

    Attributes:
        name: Human-readable name for logging and diagnostics.
    """

    name: str

    @abstractmethod
    def analyze_ir(self, program: GymProgram) -> list[Any]:
        """Find optimization opportunities on a GymProgram.

        Args:
            program: A tiled GymProgram.

        Returns:
            List of transform opportunities. Each element is a single option
            that can be passed to ``transform_ir()``.
        """

    @abstractmethod
    def transform_ir(self, program: GymProgram, option: Any) -> GymProgram:
        """Apply one opportunity, return new GymProgram.

        Args:
            program: GymProgram to transform.
            option: A single opportunity from ``analyze_ir()``.

        Returns:
            New GymProgram with the optimization applied.
        """
