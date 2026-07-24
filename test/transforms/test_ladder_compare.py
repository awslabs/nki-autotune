"""Tests for the strict manual-ladder comparison oracle."""

from __future__ import annotations

from test.transforms._ladder_compare import assert_matches_render_ordered

import pytest

_INTERLEAVED = """
def generated(lhs, rhs):
    a = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    nisa.memset(dst=a[0:128, 0:512], value=0.0)
    b = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=b[0:128, 0:512], src=a[0:128, 0:512])
"""

_HOISTED = """
def hand_written(lhs, rhs):
    a = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    b = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=a[0:128, 0:512], value=0.0)
    nisa.tensor_copy(dst=b[0:128, 0:512], src=a[0:128, 0:512])
"""


def test_oracle_rejects_declaration_order_difference() -> None:
    """Buffer declaration position is part of the compared program."""
    with pytest.raises(AssertionError):
        assert_matches_render_ordered(_INTERLEAVED, _HOISTED)


def test_oracle_rejects_missing_assertion() -> None:
    """Shape assertions cannot disappear from one side of the comparison."""
    with_assert = "def generated(x):\n    assert x.shape == (128, 128)\n    return x\n"
    without_assert = "def hand_written(x):\n    return x\n"
    with pytest.raises(AssertionError):
        assert_matches_render_ordered(with_assert, without_assert)


def test_oracle_rejects_argument_order_difference() -> None:
    """Operation argument order survives normalization."""
    left = "def generated(x):\n    return nisa.tensor_copy(src=x, dst=x)\n"
    right = "def hand_written(x):\n    return nisa.tensor_copy(dst=x, src=x)\n"
    with pytest.raises(AssertionError):
        assert_matches_render_ordered(left, right)


def test_oracle_accepts_function_name_and_affine_spelling() -> None:
    """Only function names and equivalent integer affine syntax are normalized."""
    generated = "def generated(x):\n    return x[0:0 + 128, (i * 4 + j) * 128:(i * 4 + j) * 128 + 128]\n"
    hand_written = "def kernel_0(x):\n    return x[0:128, i * 512 + j * 128:i * 512 + j * 128 + 128]\n"
    assert_matches_render_ordered(generated, hand_written)
