"""Tests for the order-significant variant of the AST-canonical compare oracle."""

from __future__ import annotations

import pytest

from test.transforms._ladder_compare import assert_matches_render, assert_matches_render_ordered

_INTERLEAVED = """
def k(lhs_T, rhs):
    a = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]
    nisa.memset(dst=a[0][0:128, 0, 0:512], value=0.0)
    b = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    nisa.tensor_copy(dst=b[0][0:128, 0, 0:512], src=a[0][0:128, 0, 0:512])
"""

_HOISTED = """
def k(lhs_T, rhs):
    a = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]
    b = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    nisa.memset(dst=a[0][0:128, 0, 0:512], value=0.0)
    nisa.tensor_copy(dst=b[0][0:128, 0, 0:512], src=a[0][0:128, 0, 0:512])
"""


def test_hoisting_compare_ignores_decl_order():
    """The existing hoisting oracle treats interleaved and hoisted decls as equal."""
    assert_matches_render(_INTERLEAVED, _HOISTED)


def test_ordered_compare_rejects_decl_order_difference():
    """The ordered oracle rejects a kernel whose decls sit in a different position."""
    with pytest.raises(AssertionError):
        assert_matches_render_ordered(_INTERLEAVED, _HOISTED)


def test_ordered_compare_accepts_identical_order():
    """The ordered oracle accepts two sources with identical statement order."""
    assert_matches_render_ordered(_INTERLEAVED, _INTERLEAVED)
