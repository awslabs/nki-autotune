"""Shared test utilities and fixtures for pytest."""

from collections.abc import Callable

import numpy as np
import pytest

import nkigym
from nkigym.ir import GymProgram


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


def make_random_array(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Generate a deterministic random float32 array for testing.

    Args:
        shape: Shape of the array to generate.
        seed: Random seed for reproducibility.

    Returns:
        Random float32 array with values in [-1, 1] range.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def assert_programs_numerically_equal(before: GymProgram, after: GymProgram, rtol: float, atol: float) -> None:
    """Assert two GymPrograms produce identical outputs for random inputs.

    Executes both programs via GymProgram.__call__(), generates
    deterministic random inputs from the before program's input_shapes,
    and compares outputs with assert_allclose.

    Args:
        before: Reference program.
        after: Program to compare against.
        rtol: Relative tolerance for assert_allclose.
        atol: Absolute tolerance for assert_allclose.
    """
    input_shapes = dict(before.input_shapes)
    kwargs = {p: make_random_array(input_shapes[p], seed=42 + i) for i, p in enumerate(sorted(input_shapes))}
    expected = before(**kwargs)
    actual = after(**kwargs)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
