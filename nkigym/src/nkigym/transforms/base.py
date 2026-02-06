"""Base class for NKI Gym IR transforms.

All transforms follow the analyze-then-transform pattern:
1. analyze(): Inspect a tiled function to find optimization opportunities
2. transform(): Apply the optimization, returning a new callable

Subclasses define the analysis result type and transform logic.
The __call__ convenience method runs both phases in sequence.

To add a new transform:
1. Subclass Transform
2. Implement analyze() returning your analysis type
3. Implement transform() applying optimizations from analysis
4. Register the transform in transforms/__init__.py
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np


class Transform(ABC):
    """Base class for nkigym IR transforms.

    Each transform operates on the tiled IR (a callable with __source__)
    produced by the tiling pass, and returns a new callable with the
    optimization applied.

    Attributes:
        name: Human-readable name for logging and diagnostics.
    """

    name: str

    @abstractmethod
    def analyze(self, func: Callable) -> Any:
        """Analyze a tiled function to find optimization opportunities.

        Args:
            func: A tiled function (with __source__ attribute) to analyze.

        Returns:
            Transform-specific analysis result describing what can be optimized.
        """

    @abstractmethod
    def transform(self, func: Callable, analysis: Any) -> Callable[..., np.ndarray]:
        """Apply the transform based on analysis results.

        Args:
            func: A tiled function to transform.
            analysis: Analysis result from analyze().

        Returns:
            New callable with the optimization applied.
        """

    def __call__(self, func: Callable) -> Callable[..., np.ndarray]:
        """Convenience: analyze then transform in one step.

        Args:
            func: A tiled function to optimize.

        Returns:
            New callable with all optimizations from analysis applied.
        """
        analysis = self.analyze(func)
        return self.transform(func, analysis)
