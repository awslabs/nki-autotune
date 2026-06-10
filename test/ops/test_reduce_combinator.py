"""Tests for the REDUCE_COMBINATOR reducer declaration on NKIOp."""

from __future__ import annotations

from nkigym.ops.base import NKIOp, ReduceCombinator
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_copy import NKITensorCopy


def test_matmul_declares_sum_reducer() -> None:
    """NKIMatmul exposes the sum reducer (combiner='add', identity=0.0)."""
    rc = NKIMatmul.REDUCE_COMBINATOR
    assert isinstance(rc, ReduceCombinator)
    assert rc.combiner == "add"
    assert rc.identity == 0.0


def test_non_reduction_op_has_no_reducer() -> None:
    """An op with no reduction (tensor_copy) declares REDUCE_COMBINATOR = None."""
    assert NKITensorCopy.REDUCE_COMBINATOR is None


def test_base_default_is_none() -> None:
    """The base NKIOp default is None."""
    assert NKIOp.REDUCE_COMBINATOR is None
