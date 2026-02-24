"""Tests for loop rolling codegen pass.

Run with: pytest test/test_loop_rolling.py -v
"""

import numpy as np
import pytest
from conftest import make_random_array, normalize_source
from golden.loop_rolling import CASES, to_source

from nkigym.codegen.loop_rolling import _roll_once, roll_loops
from nkigym.utils import source_to_callable


def _run_source(source: str, a_shape: tuple[int, ...], b_shape: tuple[int, ...]) -> np.ndarray:
    """Execute source with deterministic random inputs and return the output.

    Args:
        source: Python source code string containing a 'matmul' function.
        a_shape: Shape of the first input array.
        b_shape: Shape of the second input array.

    Returns:
        Output array from executing the function.
    """
    func = source_to_callable(source, "matmul")
    a = make_random_array(a_shape, seed=42)
    b = make_random_array(b_shape, seed=43)
    return func(a, b)


@pytest.mark.parametrize("name", list(CASES.keys()))
def test_roll(name: str) -> None:
    """Verify rolling produces expected structure and preserves numerics."""
    case = CASES[name]
    before = to_source(case.before)
    expected_after = to_source(case.after)

    actual_after = before
    for _ in range(case.num_rolls):
        actual_after = _roll_once(actual_after)
    assert normalize_source(actual_after) == normalize_source(expected_after)

    if _roll_once(actual_after) == actual_after:
        assert normalize_source(roll_loops(before)) == normalize_source(expected_after)

    expected = _run_source(before, case.a_shape, case.b_shape)
    actual = _run_source(actual_after, case.a_shape, case.b_shape)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
