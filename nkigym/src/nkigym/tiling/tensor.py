"""TracedTensor class for symbolic tensor tracing.

This module provides TracedTensor, a symbolic tensor that tracks
dimension information during tracing. Operations are handled by
NKIOp classes in nkigym.ops.
"""

from nkigym.tiling.dim_tracker import _DimTracker


class TracedTensor:
    """Symbolic tensor that tracks dimension information during tracing.

    Attributes:
        name: Tensor name (e.g., "a", "b", or generated for intermediates).
        shape: Shape tuple of the tensor.
        dims: List of dimension IDs for each axis.
        tracker: Shared dimension tracker.
    """

    def __init__(self, name: str, shape: tuple[int, ...], dims: list[str], tracker: _DimTracker) -> None:
        """Initialize a TracedTensor.

        Args:
            name: Tensor name (e.g., "a", "b", or generated for intermediates).
            shape: Shape tuple of the tensor.
            dims: List of dimension IDs for each axis.
            tracker: Shared dimension tracker.
        """
        self.name = name
        self.shape = shape
        self.dims = dims
        self.tracker = tracker

    def __repr__(self) -> str:
        """Return formatted string representation."""
        return f"TracedTensor({self.name}, shape={self.shape}, dims={self.dims})"
