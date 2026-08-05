"""Tests for the REDUCE_COMBINATOR reducer declaration on NKIOp."""

from __future__ import annotations

from nkigym.ops.base import NKIOp, ReduceCombinator
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_copy import NKITensorCopy


def test_reduce_combinator_contract() -> None:
    """Reduction operations declare a reducer and other operations default to none."""
    rc = NKIMatmul.REDUCE_COMBINATOR
    assert isinstance(rc, ReduceCombinator)
    assert rc.combiner == "add"
    assert rc.identity == 0.0
    assert NKITensorCopy.REDUCE_COMBINATOR is None
    assert NKIOp.REDUCE_COMBINATOR is None
