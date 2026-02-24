"""Tests for operand merge analysis and transform.

Each case specifies a pre-tiled before golden, a merge count, and a post-merge
after golden. The single parametrized test validates IR transformation and
numerical equivalence.

Run with: pytest test/test_operand_merge.py -v
"""

import pytest
from conftest import assert_programs_numerically_equal
from operand_merge_golden import CASES, CORNER_CASES, OperandMergeCase

from nkigym.transforms.operand_merge import OperandMergeTransform

_merge = OperandMergeTransform()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_operand_merge(case: OperandMergeCase) -> None:
    """Verify operand merge IR transformation and numerical equivalence.

    For each case:
    1. Apply merge_count iterative merges to the before GymProgram.
    2. Assert transformed stmts match after golden stmts.
    3. Verify numerical equivalence between before and after callables.

    Args:
        case: OperandMergeCase with before/after GymProgram and merge count.
    """
    merged_program = case.before
    for _ in range(case.merge_count):
        opps = _merge.analyze_ir(merged_program)
        assert len(opps) > 0
        merged_program = _merge.transform_ir(merged_program, opps[0])

    assert merged_program.stmts == case.after.stmts

    assert_programs_numerically_equal(case.before, merged_program, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("case", CORNER_CASES, ids=lambda c: c.id)
def test_operand_merge_no_opportunity(case: OperandMergeCase) -> None:
    """Verify that analyze_ir finds no merge opportunities for corner cases.

    These cases have output gaps from np.empty that make numerical comparison
    unreliable, so only the IR structural assertion is checked.

    Args:
        case: OperandMergeCase with merge_count=0 and identical before/after goldens.
    """
    opps = _merge.analyze_ir(case.before)
    assert len(opps) == 0
