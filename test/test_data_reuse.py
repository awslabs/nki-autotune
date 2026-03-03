"""Tests for data reuse analysis and transform.

Each case specifies a hardcoded before GymProgram, expected reuse pairs,
a merge count, and a hardcoded after GymProgram. The single parametrized
test validates analysis correctness, IR transformation, and numerical
equivalence.

Run with: pytest test/test_data_reuse.py -v
"""

import pytest
from conftest import assert_programs_numerically_equal
from golden.data_reuse import CASES, DataReuseCase

from nkigym.transforms import DataReuseTransform

_reuse = DataReuseTransform()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_data_reuse(case: DataReuseCase) -> None:
    """Verify data reuse analysis, IR transformation, and numerical equivalence.

    For each case:
    1. Assert analyze_ir returns exact expected pairs on the before GymProgram.
    2. Apply merge_count iterative merges.
    3. Assert transformed stmts match after golden stmts.
    4. Verify numerical equivalence between before and merged programs.

    Args:
        case: DataReuseCase with before/after GymPrograms, expected pairs, and merge count.
    """
    pairs = _reuse.analyze_ir(case.before)
    assert pairs == case.expected_pairs

    merged_program = case.before
    for _ in range(case.merge_count):
        pairs = _reuse.analyze_ir(merged_program)
        assert len(pairs) > 0
        merged_program = _reuse.transform_ir(merged_program, pairs[0])

    assert merged_program.stmts == case.after.stmts

    assert_programs_numerically_equal(case.before, merged_program, rtol=1e-5, atol=1e-8)
