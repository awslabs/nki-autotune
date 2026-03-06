"""Tests for loop rolling codegen pass.

Run with: pytest test/test_loop_rolling.py -v
"""

import ast
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pytest
from conftest import make_random_array, normalize_source
from golden.loop_rolling_flat import (
    flat_1x1,
    flat_1x4,
    flat_1x5,
    flat_2x1,
    flat_2x2,
    flat_3x1,
    flat_3x5,
    flat_4x1,
    flat_4x4,
)
from golden.loop_rolling_flat_red import flat_2x2_red2, flat_2x3_red3, flat_3x5_red2, flat_red2, flat_red4, flat_red8
from golden.loop_rolling_rolled import (
    roll1_1x4,
    roll1_1x5,
    roll1_2x1,
    roll1_2x2,
    roll1_2x2_red2,
    roll1_3x1,
    roll1_4x1,
    roll1_red4,
    roll1_red8,
    roll2_2x2,
    roll2_2x2_red2,
    roll2_3x5,
    roll2_3x5_red2,
    roll2_4x4,
    roll3_2x3_red3,
)

from nkigym.program_to_nki.loop_rolling import _roll_once, roll_loops
from nkigym.utils import callable_to_source, source_to_callable

_HEADER = "import numpy as np\nimport nkigym\n"


class Case(NamedTuple):
    """A loop rolling test case with parallel before/after functions.

    Attributes:
        before: Unrolled golden function (input to rolling pass).
        after: Expected golden function (output of rolling pass).
        num_rolls: Number of _roll_once passes to reach after from before.
        a_shape: Shape of the first input array for numerical verification.
        b_shape: Shape of the second input array for numerical verification.
    """

    before: Callable
    after: Callable
    num_rolls: int
    a_shape: tuple[int, ...]
    b_shape: tuple[int, ...]


def _to_source(func: Callable) -> str:
    """Convert a golden function to rollable source with standard name and imports.

    Extracts source via callable_to_source, strips docstrings and type
    annotations added for style compliance, renames the function to
    'matmul', and prepends the import header.

    Args:
        func: A golden function from the companion modules.

    Returns:
        Complete Python source string ready for the loop rolling pass.
    """
    raw = callable_to_source(func)
    tree = ast.parse(raw)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.name = "matmul"
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
            node.returns = None
            for arg in node.args.args:
                arg.annotation = None
            break
    ast.fix_missing_locations(tree)
    return _HEADER + ast.unparse(tree)


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


"""                before             after              rolls  a_shape        b_shape       """
CASES: dict[str, Case] = {
    "1x1": Case(flat_1x1, flat_1x1, 0, (128, 128), (128, 128)),
    "2x1": Case(flat_2x1, roll1_2x1, 1, (128, 128), (128, 256)),
    "3x1": Case(flat_3x1, roll1_3x1, 1, (128, 128), (128, 384)),
    "4x1": Case(flat_4x1, roll1_4x1, 1, (128, 128), (128, 512)),
    "1x4": Case(flat_1x4, roll1_1x4, 1, (128, 512), (128, 128)),
    "1x5": Case(flat_1x5, roll1_1x5, 1, (128, 640), (128, 128)),
    "2x2_roll1": Case(flat_2x2, roll1_2x2, 1, (128, 256), (128, 256)),
    "2x2": Case(flat_2x2, roll2_2x2, 2, (128, 256), (128, 256)),
    "3x5": Case(flat_3x5, roll2_3x5, 2, (128, 640), (128, 384)),
    "4x4": Case(flat_4x4, roll2_4x4, 2, (128, 512), (128, 512)),
    "red2": Case(flat_red2, flat_red2, 0, (256, 128), (256, 128)),
    "red4": Case(flat_red4, roll1_red4, 1, (512, 128), (512, 128)),
    "red8": Case(flat_red8, roll1_red8, 1, (1024, 128), (1024, 128)),
    "2x2_red2_roll1": Case(flat_2x2_red2, roll1_2x2_red2, 1, (256, 256), (256, 256)),
    "2x2_red2": Case(flat_2x2_red2, roll2_2x2_red2, 2, (256, 256), (256, 256)),
    "3x5_red2": Case(flat_3x5_red2, roll2_3x5_red2, 2, (256, 640), (256, 384)),
    "2x3_red3": Case(flat_2x3_red3, roll3_2x3_red3, 3, (384, 256), (384, 384)),
}


@pytest.mark.parametrize("name", list(CASES.keys()))
def test_roll(name: str) -> None:
    """Verify rolling produces expected structure, converges, and preserves numerics."""
    case = CASES[name]
    before = _to_source(case.before)
    expected_after = _to_source(case.after)

    actual_after = before
    for _ in range(case.num_rolls):
        actual_after = _roll_once(actual_after)
    assert normalize_source(actual_after) == normalize_source(expected_after)

    if _roll_once(actual_after) == actual_after:
        assert normalize_source(roll_loops(before)) == normalize_source(expected_after)

    expected = _run_source(before, case.a_shape, case.b_shape)
    actual = _run_source(actual_after, case.a_shape, case.b_shape)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
