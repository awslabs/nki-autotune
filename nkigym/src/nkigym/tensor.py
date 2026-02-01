"""TracedTensor class for symbolic tensor tracing.

This module provides TracedTensor, which intercepts NumPy operations
during symbolic execution to track dimension information.
"""

from collections.abc import Callable

import numpy as np

from nkigym.dim_tracker import _DimTracker
from nkigym.numpy_ops import HANDLED_FUNCTIONS


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

    def __array_function__(self, func: Callable, _types: tuple, args: tuple, kwargs: dict) -> "TracedTensor":
        """Intercept NumPy function calls on TracedTensor.

        Args:
            func: NumPy function being called.
            _types: Types of arguments (unused).
            args: Positional arguments to the function.
            kwargs: Keyword arguments to the function.

        Returns:
            TracedTensor result from the registered handler.

        Raises:
            NotImplementedError: If the function is not implemented.
        """
        if func not in HANDLED_FUNCTIONS:
            raise NotImplementedError(f"TracedTensor does not implement {func.__name__}")
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **_kwargs) -> "TracedTensor":
        """Intercept NumPy ufunc calls on TracedTensor.

        Args:
            ufunc: NumPy ufunc being called.
            method: Ufunc method (e.g., "__call__").
            *inputs: Input tensors.
            **_kwargs: Additional keyword arguments (unused).

        Returns:
            TracedTensor result from the handler.

        Raises:
            NotImplementedError: If the ufunc is not implemented.
        """
        if ufunc == np.matmul and method == "__call__":
            if np.matmul not in HANDLED_FUNCTIONS:
                raise NotImplementedError(f"TracedTensor does not implement ufunc {ufunc.__name__}")
            return HANDLED_FUNCTIONS[np.matmul](inputs[0], inputs[1])
        raise NotImplementedError(f"TracedTensor does not implement ufunc {ufunc.__name__}")

    def __matmul__(self, other: "TracedTensor") -> "TracedTensor":
        """Handle matrix multiplication: C[m,n] = A[m,k] @ B[k,n].

        Args:
            other: Right-hand side tensor.

        Returns:
            TracedTensor result of the matrix multiplication.

        Raises:
            NotImplementedError: If matmul is not registered.
        """
        if np.matmul not in HANDLED_FUNCTIONS:
            raise NotImplementedError("TracedTensor does not implement matmul")
        return HANDLED_FUNCTIONS[np.matmul](self, other)
