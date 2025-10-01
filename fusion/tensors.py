"""Tensor wrapper for fusion operations."""

from typing import Any, Tuple, Union

import numpy as np


class Tensor:
    """Numpy array wrapper with fusion-specific helpers."""

    def __init__(self, data: Union[np.ndarray, list, float]):
        """Initialize a Tensor from various input types."""
        # Handle numpy scalar types
        if isinstance(data, (np.floating, np.integer, np.complexfloating)):
            data = np.array(data)
        elif isinstance(data, (list, float, int, complex)):
            data = np.array(data)
        elif not isinstance(data, np.ndarray):
            raise TypeError(f"Unsupported type for Tensor: {type(data)}")

        self.data = data
        self.shape = data.shape
        self.ndim = data.ndim
        self.dtype = data.dtype
        self.size = data.size

    def __getitem__(self, idx: Union[int, slice, Tuple[Union[int, slice], ...]]) -> Any:
        """Support indexing for fusion axis iteration."""
        result = self.data[idx]
        # Return scalar or numpy array directly for compatibility
        return result

    def __len__(self) -> int:
        """Length along first dimension (fusion axis)."""
        return self.shape[0] if self.ndim > 0 else 1

    def __repr__(self) -> str:
        """String representation."""
        return f"Tensor(shape={self.shape}, dtype={self.dtype})"

    def __str__(self) -> str:
        """String conversion."""
        return str(self.data)
