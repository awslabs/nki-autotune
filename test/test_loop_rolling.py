"""Tests for loop rolling codegen pass.

Test classes:
- TestRollOnce: single-pass _roll_once golden source verification
- TestRollFull: full roll_loops() golden source + numerical verification

Run with: pytest test/test_loop_rolling.py -v
"""

import numpy as np
import pytest
from conftest import make_random_array, normalize_source
from loop_rolling_golden import GOLDEN, ROLL_ONCE, _make_tiled

from nkigym.lower.loop_rolling import _roll_once, roll_loops
from nkigym.utils.source import exec_source_to_func, get_source

ROLL_ONCE_CASES = [(name, *vals) for name, vals in ROLL_ONCE.items()]
GOLDEN_CASES = [(name, *vals) for name, vals in GOLDEN.items()]


class TestRollOnce:
    """Tests for single-pass rolling detection and application.

    Verifies that _roll_once produces the expected source after one
    rolling pass.
    """

    @pytest.mark.parametrize(
        "name,a_shape,b_shape,expected_source", ROLL_ONCE_CASES, ids=[c[0] for c in ROLL_ONCE_CASES]
    )
    def test_roll_once_golden(
        self, name: str, a_shape: tuple[int, int], b_shape: tuple[int, int], expected_source: str
    ):
        """Verify _roll_once output matches golden source.

        Args:
            name: Test case identifier.
            a_shape: Shape of the first input matrix.
            b_shape: Shape of the second input matrix.
            expected_source: Expected source after one pass.
        """
        tiled = _make_tiled(a_shape, b_shape)
        result = _roll_once(get_source(tiled))
        assert normalize_source(result) == normalize_source(expected_source)

        rolled = exec_source_to_func(result, "tiled_matmul")
        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)
        np.testing.assert_allclose(rolled(a, b), tiled(a, b), rtol=1e-4, atol=1e-4)


class TestRollFull:
    """Tests for full roll_loops() convergence.

    For each shape configuration, verifies that:
    1. The rolled source matches the golden expected string.
    2. The rolled function produces numerically identical output.
    """

    @pytest.mark.parametrize("name,a_shape,b_shape,expected_source", GOLDEN_CASES, ids=[c[0] for c in GOLDEN_CASES])
    def test_rolled_golden(self, name: str, a_shape: tuple[int, int], b_shape: tuple[int, int], expected_source: str):
        """Verify rolled source matches golden and numerical output is correct.

        Args:
            name: Test case identifier.
            a_shape: Shape of the first input matrix.
            b_shape: Shape of the second input matrix.
            expected_source: Expected rolled source string.
        """
        tiled = _make_tiled(a_shape, b_shape)
        rolled = roll_loops(tiled)

        assert normalize_source(get_source(rolled)) == normalize_source(expected_source)

        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)
        np.testing.assert_allclose(rolled(a, b), tiled(a, b), rtol=1e-4, atol=1e-4)
