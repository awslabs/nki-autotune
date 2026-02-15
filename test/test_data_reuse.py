"""Tests for data reuse analysis and transform.

Each case specifies a starting tiled function, expected reuse pairs,
merge steps to apply, expected post-merge source, and input shapes
for numerical equivalence verification.

Run with: pytest test/test_data_reuse.py -v
"""

import numpy as np
import pytest
from conftest import make_random_array, normalize_source
from data_reuse_golden import CASES

from nkigym.ir import callable_to_ir, ir_to_callable
from nkigym.transforms import DataReuseTransform, normalize_reuse_groups
from nkigym.utils import callable_to_source

_reuse = DataReuseTransform()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_data_reuse(case):
    """Verify data reuse analysis, merge source, and numerical correctness.

    For each case:
    1. Convert the starting function to IR and verify the expected reuse pairs.
    2. If merges are specified, apply them and verify the merged source matches.
    3. Verify the merged function produces numerically equivalent output.
    """
    program = callable_to_ir(case.func)
    pairs = _reuse.analyze_ir(program)
    assert normalize_reuse_groups(pairs) == normalize_reuse_groups(case.expected_pairs)

    if not case.merges:
        return

    merged_program = program
    for pair in case.merges:
        merged_program = _reuse.transform_ir(merged_program, pair)
    merged_func = ir_to_callable(merged_program)

    assert normalize_source(callable_to_source(merged_func)) == normalize_source(case.expected_source)

    inputs = [make_random_array(shape, seed=42 + i) for i, shape in enumerate(case.input_shapes)]
    expected = case.func(*inputs)
    actual = merged_func(*inputs)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
