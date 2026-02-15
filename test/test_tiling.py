"""Tests for the tiling pass.

Verifies that for each input shape configuration:
- Generated tiled source matches a hardcoded golden string
- Tiled function produces numerically correct results
"""

import numpy as np
import pytest
from conftest import assert_arrays_close, make_random_array, normalize_source
from tiling_golden import GOLDEN_DOUBLE_MATMUL_SOURCE, GOLDEN_SINGLE_MATMUL_SOURCE

from nkigym.ir import func_to_program, program_to_func, program_to_source
from nkigym.tiling import tile_program


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
    def test_golden_source_and_numerical(self, a_shape: tuple[int, int], b_shape: tuple[int, int], matmul_func) -> None:
        """Verify tiled source matches golden string and output is numerically correct.

        Args:
            a_shape: Shape of the first input matrix.
            b_shape: Shape of the second input matrix.
            matmul_func: Fixture providing the matmul function.
        """
        program = func_to_program(matmul_func, {"a": a_shape, "b": b_shape}, np.float32)
        tiled = tile_program(program)

        actual_source = program_to_source(tiled)
        expected_source = GOLDEN_SINGLE_MATMUL_SOURCE[(a_shape, b_shape)]
        assert normalize_source(actual_source) == normalize_source(expected_source)

        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)
        expected = matmul_func(a, b)
        actual = program_to_func(tiled)(a, b)
        assert_arrays_close(actual, expected)


class TestDoubleMatmulTiling:
    """Tests that tiled double matmul matches golden source and is numerically correct."""

    @pytest.mark.parametrize(
        "a_shape,b_shape,c_shape",
        list(GOLDEN_DOUBLE_MATMUL_SOURCE.keys()),
        ids=[_shape_id(k) for k in GOLDEN_DOUBLE_MATMUL_SOURCE.keys()],
    )
    def test_golden_source_and_numerical(
        self, a_shape: tuple[int, int], b_shape: tuple[int, int], c_shape: tuple[int, int], double_matmul_func
    ) -> None:
        """Verify tiled source matches golden string and output is numerically correct.

        Args:
            a_shape: Shape of the first input matrix.
            b_shape: Shape of the second input matrix.
            c_shape: Shape of the third input matrix.
            double_matmul_func: Fixture providing the double matmul function.
        """
        program = func_to_program(double_matmul_func, {"a": a_shape, "b": b_shape, "c": c_shape}, np.float32)
        tiled = tile_program(program)

        actual_source = program_to_source(tiled)
        expected_source = GOLDEN_DOUBLE_MATMUL_SOURCE[(a_shape, b_shape, c_shape)]
        assert normalize_source(actual_source) == normalize_source(expected_source)

        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)
        c = make_random_array(c_shape, seed=44)
        expected = double_matmul_func(a, b, c)
        actual = program_to_func(tiled)(a, b, c)
        assert_arrays_close(actual, expected)
