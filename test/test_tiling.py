"""Tests for the tiling pass.

Verifies that for each input shape configuration:
- Generated tiled program matches a hardcoded golden GymProgram
- Generated tiled source matches a hardcoded golden string
- Tiled function produces numerically correct results

Uses hardcoded before/after user function source strings with minimal
pipeline dependency. Only imports source_to_program (to create the
GymProgram input) and tile_program (the function under test).
"""

import numpy as np
import pytest
from conftest import make_random_array, normalize_source
from golden.tiling import (
    GOLDEN_DOUBLE_MATMUL_PROGRAM,
    GOLDEN_DOUBLE_MATMUL_SOURCE,
    GOLDEN_SINGLE_MATMUL_PROGRAM,
    GOLDEN_SINGLE_MATMUL_SOURCE,
)

from nkigym.function_to_program import program_to_source, source_to_program, tile_program

SINGLE_MATMUL_SOURCE = """\
import nkigym

def matmul(a, b):
    return nkigym.nc_matmul(a, b)
"""

DOUBLE_MATMUL_SOURCE = """\
import nkigym

def double_matmul(a, b, c):
    return nkigym.nc_matmul(nkigym.nc_matmul(a, b), c)
"""


def _kw(shapes: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    """Create zero-filled float32 kernel kwargs from shapes."""
    return {k: np.zeros(v, dtype=np.float32) for k, v in shapes.items()}


def _shape_id(shapes: tuple[tuple[int, int], ...]) -> str:
    """Generate a test ID from shape tuples.

    Args:
        shapes: Tuple of shape tuples.

    Returns:
        String like "256x128_128x256" for test identification.
    """
    return "_".join(f"{s[0]}x{s[1]}" for s in shapes)


class TestSingleMatmulTiling:
    """Tests that tiled single matmul matches golden source and is numerically correct."""

    @pytest.mark.parametrize(
        "a_shape,b_shape",
        list(GOLDEN_SINGLE_MATMUL_SOURCE.keys()),
        ids=[_shape_id(k) for k in GOLDEN_SINGLE_MATMUL_SOURCE.keys()],
    )
    def test_golden_source_and_numerical(self, a_shape: tuple[int, int], b_shape: tuple[int, int]) -> None:
        """Verify tiled source matches golden string and output is numerically correct.

        Args:
            a_shape: Shape of the first input matrix.
            b_shape: Shape of the second input matrix.
        """
        kwargs = _kw({"a": a_shape, "b": b_shape})
        program = source_to_program(SINGLE_MATMUL_SOURCE, kwargs)
        tiled = tile_program(program)

        actual_source = program_to_source(tiled)
        expected_source = GOLDEN_SINGLE_MATMUL_SOURCE[(a_shape, b_shape)]
        assert normalize_source(actual_source) == normalize_source(expected_source)

        expected_program = GOLDEN_SINGLE_MATMUL_PROGRAM[(a_shape, b_shape)]
        assert tiled == expected_program

        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)
        expected = a.T @ b
        actual = tiled(a=a, b=b)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


class TestDoubleMatmulTiling:
    """Tests that tiled double matmul matches golden source and is numerically correct."""

    @pytest.mark.parametrize(
        "a_shape,b_shape,c_shape",
        list(GOLDEN_DOUBLE_MATMUL_SOURCE.keys()),
        ids=[_shape_id(k) for k in GOLDEN_DOUBLE_MATMUL_SOURCE.keys()],
    )
    def test_golden_source_and_numerical(
        self, a_shape: tuple[int, int], b_shape: tuple[int, int], c_shape: tuple[int, int]
    ) -> None:
        """Verify tiled source matches golden string and output is numerically correct.

        Args:
            a_shape: Shape of the first input matrix.
            b_shape: Shape of the second input matrix.
            c_shape: Shape of the third input matrix.
        """
        kwargs = _kw({"a": a_shape, "b": b_shape, "c": c_shape})
        program = source_to_program(DOUBLE_MATMUL_SOURCE, kwargs)
        tiled = tile_program(program)

        actual_source = program_to_source(tiled)
        expected_source = GOLDEN_DOUBLE_MATMUL_SOURCE[(a_shape, b_shape, c_shape)]
        assert normalize_source(actual_source) == normalize_source(expected_source)

        expected_program = GOLDEN_DOUBLE_MATMUL_PROGRAM[(a_shape, b_shape, c_shape)]
        assert tiled == expected_program

        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)
        c = make_random_array(c_shape, seed=44)
        expected = (a.T @ b).T @ c
        actual = tiled(a=a, b=b, c=c)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
