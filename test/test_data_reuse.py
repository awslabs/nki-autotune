"""Tests for data reuse analysis and transform.

Each case specifies a pre-tiled before golden, expected reuse pairs,
a merge count, and a post-merge after golden. The single parametrized
test validates analysis correctness, IR transformation, and numerical
equivalence.

Run with: pytest test/test_data_reuse.py -v
"""

import numpy as np
import pytest
from conftest import make_random_array
from golden.data_reuse import CASES, DataReuseCase

from nkigym.ir import program_to_source, source_to_program
from nkigym.transforms import DataReuseTransform
from nkigym.utils import callable_to_source, source_to_callable

_reuse = DataReuseTransform()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_data_reuse(case: DataReuseCase) -> None:
    """Verify data reuse analysis, IR transformation, and numerical equivalence.

    For each case:
    1. Parse before golden into a GymProgram.
    2. Assert analyze_ir returns exact expected pairs.
    3. Apply merge_count iterative merges.
    4. Parse after golden into a GymProgram.
    5. Assert transformed stmts match after golden stmts.
    6. Verify numerical equivalence between before and after callables.

    Args:
        case: DataReuseCase with before/after goldens, expected pairs, and merge count.
    """
    before_program = source_to_program(callable_to_source(case.before), case.input_shapes, case.output_dtype)

    pairs = _reuse.analyze_ir(before_program)
    assert pairs == case.expected_pairs

    merged_program = before_program
    for _ in range(case.merge_count):
        pairs = _reuse.analyze_ir(merged_program)
        assert len(pairs) > 0
        merged_program = _reuse.transform_ir(merged_program, pairs[0])

    after_program = source_to_program(callable_to_source(case.after), case.input_shapes, case.output_dtype)
    assert merged_program.stmts == after_program.stmts

    before_func = source_to_callable(program_to_source(before_program), before_program.name)
    after_func = source_to_callable(program_to_source(merged_program), merged_program.name)

    inputs = [make_random_array(case.input_shapes[p], seed=42 + i) for i, p in enumerate(sorted(case.input_shapes))]
    expected = before_func(*inputs)
    actual = after_func(*inputs)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
