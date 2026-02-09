"""Base class for NKI Gym IR transforms.

All transforms follow the analyze-then-transform pattern:

1. ``analyze(func)`` — inspect a tiled function and return a list of
   individual transform opportunities.
2. ``transform(func, option)`` — apply a single opportunity, returning
   a new callable.

The autotuner calls ``analyze()`` to discover opportunities, selects which
to apply, and calls ``transform()`` for each.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np


class Transform(ABC):
    """Base class for nkigym IR transforms.

    Each transform operates on the tiled IR (a callable with ``__source__``)
    produced by the tiling pass. ``analyze()`` returns a list of transform
    opportunities; ``transform()`` applies one at a time.

    Attributes:
        name: Human-readable name for logging and diagnostics.
    """

    name: str

    @abstractmethod
    def analyze(self, func: Callable) -> list[Any]:
        """Analyze a tiled function to find optimization opportunities.

        Args:
            func: A tiled function (with __source__ attribute) to analyze.

        Returns:
            List of transform opportunities. Each element is a single option
            that can be passed to ``transform()``.
        """

    @abstractmethod
    def transform(self, func: Callable, option: Any) -> Callable[..., np.ndarray]:
        """Apply a single transform opportunity.

        Args:
            func: A tiled function to transform.
            option: A single opportunity from ``analyze()``.

        Returns:
            New callable with the optimization applied.
        """
