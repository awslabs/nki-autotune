"""Shared test utilities and fixtures for pytest."""

from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import strategies as st

import nkigym
from nkigym.tiling.analysis import DimensionAnalysis


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


def shape_id(shapes: tuple[tuple[int, int], ...]) -> str:
    """Generate a test ID from shape tuples.

    Args:
        shapes: Tuple of shape tuples.

    Returns:
        String like "256x128_128x256" for test identification.
    """
    return "_".join(f"{s[0]}x{s[1]}" for s in shapes)


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
def valid_shape_multiple_of_128(draw: st.DrawFn) -> tuple[int, int]:
    """Generate a valid 2D shape where both dimensions are multiples of 128.

    Shapes are constrained to reasonable sizes for testing (128 to 512).

    Args:
        draw: Hypothesis draw function for generating values.

    Returns:
        A tuple (rows, cols) where both are multiples of 128.
    """
    rows = draw(st.integers(min_value=1, max_value=4)) * TILE_SIZE
    cols = draw(st.integers(min_value=1, max_value=4)) * TILE_SIZE
    return (rows, cols)


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


valid_dim_size = st.integers(min_value=1, max_value=8).map(lambda x: x * TILE_SIZE)
"""Strategy for generating valid dimension sizes as multiples of TILE_SIZE."""


@st.composite
def matmul_shapes(draw: st.DrawFn) -> tuple[tuple[int, int], tuple[int, int]]:
    """Generate valid nc_matmul shapes where all dimensions are multiples of TILE_SIZE.

    This strategy generates shapes for nc_matmul: C[m, n] = A[k, m].T @ B[k, n]
    with a wider range of sizes (up to 8 tiles) compared to matmul_input_shapes.

    Args:
        draw: Hypothesis draw function for generating values.

    Returns:
        Tuple of (a_shape, b_shape) for A[k, m] @ B[k, n].
    """
    m = draw(valid_dim_size)
    k = draw(valid_dim_size)
    n = draw(valid_dim_size)
    return ((k, m), (k, n))


@st.composite
def matmul_shapes_with_reduction_tile_index(draw: st.DrawFn) -> tuple[tuple[int, int], tuple[int, int], int]:
    """Generate valid nc_matmul shapes with a valid reduction tile index.

    This strategy is useful for testing reduction tiling where we need to
    verify behavior at specific reduction tile positions.

    Args:
        draw: Hypothesis draw function for generating values.

    Returns:
        Tuple of (a_shape, b_shape, reduction_tile_index) where:
        - a_shape: Shape of first matrix (k, m)
        - b_shape: Shape of second matrix (k, n)
        - reduction_tile_index: Valid index into reduction tiles (0 to k//TILE_SIZE - 1)
    """
    m = draw(valid_dim_size)
    k = draw(valid_dim_size)
    n = draw(valid_dim_size)
    num_reduction_tiles = k // TILE_SIZE
    reduction_tile_index = draw(st.integers(min_value=0, max_value=num_reduction_tiles - 1))
    return ((k, m), (k, n), reduction_tile_index)


def assert_dimension_analysis_equal(actual: DimensionAnalysis, expected: DimensionAnalysis) -> None:
    """Assert two DimensionAnalysis objects are equal with detailed error messages.

    Args:
        actual: The actual DimensionAnalysis result.
        expected: The expected golden DimensionAnalysis.

    Raises:
        AssertionError: If any field differs between actual and expected.
    """
    assert (
        actual.dim_order == expected.dim_order
    ), f"dim_order mismatch:\n  actual: {actual.dim_order}\n  expected: {expected.dim_order}"
    assert (
        actual.dim_info == expected.dim_info
    ), f"dim_info mismatch:\n  actual: {actual.dim_info}\n  expected: {expected.dim_info}"
    assert (
        actual.tensor_dims == expected.tensor_dims
    ), f"tensor_dims mismatch:\n  actual: {actual.tensor_dims}\n  expected: {expected.tensor_dims}"
    assert (
        actual.tensor_shapes == expected.tensor_shapes
    ), f"tensor_shapes mismatch:\n  actual: {actual.tensor_shapes}\n  expected: {expected.tensor_shapes}"
    assert (
        actual.tile_counts == expected.tile_counts
    ), f"tile_counts mismatch:\n  actual: {actual.tile_counts}\n  expected: {expected.tile_counts}"
    assert (
        actual.num_subgraphs == expected.num_subgraphs
    ), f"num_subgraphs mismatch:\n  actual: {actual.num_subgraphs}\n  expected: {expected.num_subgraphs}"

    actual_positions = list(actual.iter_tile_positions())
    expected_positions = list(expected.iter_tile_positions())
    assert (
        actual_positions == expected_positions
    ), f"iter_tile_positions mismatch:\n  actual: {actual_positions}\n  expected: {expected_positions}"

    assert (
        actual.slice_params == expected.slice_params
    ), f"slice_params mismatch:\n  actual: {actual.slice_params}\n  expected: {expected.slice_params}"

    assert (
        actual.output == expected.output
    ), f"output mismatch:\n  actual: {actual.output}\n  expected: {expected.output}"

    assert (
        actual.reduction_tile_counts == expected.reduction_tile_counts
    ), f"reduction_tile_counts mismatch:\n  actual: {actual.reduction_tile_counts}\n  expected: {expected.reduction_tile_counts}"

    actual_reduction_positions = list(actual.iter_reduction_tile_positions())
    expected_reduction_positions = list(expected.iter_reduction_tile_positions())
    assert (
        actual_reduction_positions == expected_reduction_positions
    ), f"iter_reduction_tile_positions mismatch:\n  actual: {actual_reduction_positions}\n  expected: {expected_reduction_positions}"
