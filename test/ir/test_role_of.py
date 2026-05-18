"""Unit tests for :func:`nkigym.ir.tree.role_of`."""

from __future__ import annotations

import pytest

from nkigym.ir.tree import ISANode, role_of
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.base import AxisRole
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul


def _make_leaf(op_cls: type, axis_map: dict[str, str]) -> ISANode:
    """Build a minimal :class:`ISANode` for role lookup tests.

    Only ``op_cls`` and ``axis_map`` are consulted by :func:`role_of`,
    so the other payload fields are left at their defaults.
    """
    return ISANode(op_cls=op_cls, axis_map=axis_map)


def test_role_of_load_returns_parallel_for_both_axes() -> None:
    """``NKILoad`` does not declare any non-PARALLEL axes."""
    leaf = _make_leaf(NKILoad, {"P": "d0", "F": "d1"})
    assert role_of(leaf, "d0") == AxisRole.PARALLEL
    assert role_of(leaf, "d1") == AxisRole.PARALLEL


def test_role_of_matmul_k_is_accumulation() -> None:
    """``NKIMatmul`` declares ``K`` as ``ACCUMULATION``; ``M`` and ``N`` are ``PARALLEL``."""
    leaf = _make_leaf(NKIMatmul, {"K": "d0", "M": "d1", "N": "d2"})
    assert role_of(leaf, "d0") == AxisRole.ACCUMULATION
    assert role_of(leaf, "d1") == AxisRole.PARALLEL
    assert role_of(leaf, "d2") == AxisRole.PARALLEL


def test_role_of_activation_reduce_f_is_accumulation() -> None:
    """``NKIActivationReduce`` declares ``F`` as ``ACCUMULATION``; ``P`` is ``PARALLEL``."""
    leaf = _make_leaf(NKIActivationReduce, {"P": "d0", "F": "d1"})
    assert role_of(leaf, "d0") == AxisRole.PARALLEL
    assert role_of(leaf, "d1") == AxisRole.ACCUMULATION


def test_role_of_unmapped_dim_raises() -> None:
    """Asking about a dim the leaf does not touch is a loud failure."""
    leaf = _make_leaf(NKILoad, {"P": "d0", "F": "d1"})
    with pytest.raises(KeyError, match="no axis mapping"):
        role_of(leaf, "d99")
