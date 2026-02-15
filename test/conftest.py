"""Shared test utilities and fixtures for pytest."""

from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import strategies as st

import nkigym


@pytest.fixture
def matmul_func() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Fixture providing a standard matmul function for testing.

    Returns a function that computes matrix multiplication using nkigym.nc_matmul.
    This fixture reduces code duplication across test files by providing
    a consistent matmul implementation.

    Returns:
        A callable that takes two numpy arrays and returns their matrix product.
    """

    def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute matrix multiplication.

        Args:
            a: First input matrix of shape (K, M).
            b: Second input matrix of shape (K, N).

        Returns:
            Matrix product of shape (M, N).
        """
        return nkigym.nc_matmul(a, b)

    return matmul


@pytest.fixture
def double_matmul_func() -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Fixture providing a standard double matmul function for testing.

    Returns a function that computes double matrix multiplication using nkigym.nc_matmul.
    This fixture reduces code duplication across test files by providing
    a consistent double matmul implementation.

    Returns:
        A callable that takes three numpy arrays and returns their chained matrix product.
    """

    def double_matmul(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Compute double matrix multiplication.

        Args:
            a: First input matrix of shape (K1, M).
            b: Second input matrix of shape (K1, K2).
            c: Third input matrix of shape (K2, N).

        Returns:
            Matrix product of shape (M, N).
        """
        return nkigym.nc_matmul(nkigym.nc_matmul(a, b), c)

    return double_matmul


def assert_arrays_close(actual: np.ndarray, expected: np.ndarray, rtol: float = 1e-4, atol: float = 1e-4) -> None:
    """Assert two arrays are numerically close with informative error messages.

    Uses relaxed tolerances (rtol=1e-4, atol=1e-4) to account for floating-point
    differences introduced by reduction tiling, where the order of operations
    changes due to tiling the reduction dimension.

    Args:
        actual: Result from tiled function.
        expected: Result from original function.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Raises:
        AssertionError: If arrays differ beyond tolerance.
    """
    assert actual.shape == expected.shape, f"Shape mismatch: actual {actual.shape} vs expected {expected.shape}"
    assert actual.dtype == expected.dtype, f"Dtype mismatch: actual {actual.dtype} vs expected {expected.dtype}"
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=atol, err_msg="Tiled function output differs from original"
    )


def normalize_source(source: str) -> str:
    """Normalize source code for comparison.

    Strips leading/trailing whitespace from each line, removes blank lines,
    and joins with single newlines.

    Args:
        source: Source code string.

    Returns:
        Normalized source string.
    """
    lines = [line.strip() for line in source.strip().splitlines()]
    return "\n".join(line for line in lines if line)


def make_random_array(shape: tuple[int, ...], seed: int, dtype: np.dtype = np.float32) -> np.ndarray:
    """Generate a deterministic random array for testing.

    Args:
        shape: Shape of the array to generate.
        seed: Random seed for reproducibility.
        dtype: Data type for the array.

    Returns:
        Random array with values in [-1, 1] range.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=shape).astype(dtype)


TILE_SIZE = 128


@st.composite
def matmul_input_shapes(draw: st.DrawFn) -> dict[str, tuple[int, int]]:
    """Generate valid input shapes for nc_matmul operation.

    Generates compatible shapes for nc_matmul: C[m, n] = A[k, m].T @ B[k, n]
    where m, k, n are multiples of 128.

    Args:
        draw: Hypothesis draw function for generating values.

    Returns:
        Dictionary with keys 'a' and 'b' mapping to shape tuples:
        - 'a': (k, m) shape for first matrix
        - 'b': (k, n) shape for second matrix
    """
    m = draw(st.integers(min_value=1, max_value=4)) * TILE_SIZE
    k = draw(st.integers(min_value=1, max_value=4)) * TILE_SIZE
    n = draw(st.integers(min_value=1, max_value=4)) * TILE_SIZE
    return {"a": (k, m), "b": (k, n)}
