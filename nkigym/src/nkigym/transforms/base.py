"""Base class for NKI Gym IR transforms.

All transforms follow the analyze-then-transform pattern on the tuple IR:

1. ``analyze_ir(program)`` — inspect a program tuple and return a list of
   individual transform opportunities.
2. ``transform_ir(program, option)`` — apply a single opportunity, returning
   a new program tuple.

The autotuner calls ``analyze_ir()`` to discover opportunities, selects which
to apply, and calls ``transform_ir()`` for each.
"""

from abc import ABC, abstractmethod
from typing import Any

from nkigym.ir import Program


class Transform(ABC):
    """Base class for nkigym IR transforms.

    Subclasses implement ``analyze_ir`` and ``transform_ir`` which operate
    on the tuple-based program IR directly.

    Attributes:
        name: Human-readable name for logging and diagnostics.
    """

    name: str

    @abstractmethod
    def analyze_ir(self, program: Program) -> list[Any]:
        """Find optimization opportunities on the tuple-based program.

        Args:
            program: Program tuple (name, params, stmts, return_var).

        Returns:
            List of transform opportunities. Each element is a single option
            that can be passed to ``transform_ir()``.
        """

    @abstractmethod
    def transform_ir(self, program: Program, option: Any) -> Program:
        """Apply one opportunity, return new program tuple.

        Args:
            program: Program tuple to transform.
            option: A single opportunity from ``analyze_ir()``.

        Returns:
            New program tuple with the optimization applied.
        """
