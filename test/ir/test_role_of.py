"""Unit tests for :func:`nkigym.ir.tree.role_of`."""

from __future__ import annotations

import pytest

from nkigym.ir.tree import BlockNode, IterVar, role_of
from nkigym.ops.base import AxisRole


def _make_block(roles: dict[str, AxisRole]) -> BlockNode:
    """Build a minimal :class:`BlockNode` with iter_vars whose roles come from ``roles``."""
    iter_vars = tuple(IterVar(axis=name, dom=(0, 128), role=role) for name, role in roles.items())
    return BlockNode(iter_vars=iter_vars, iter_values=(), reads=(), writes=())


def test_role_of_returns_parallel_when_declared() -> None:
    block = _make_block({"P": AxisRole.PARALLEL, "F": AxisRole.PARALLEL})
    assert role_of(block, "P") == AxisRole.PARALLEL
    assert role_of(block, "F") == AxisRole.PARALLEL


def test_role_of_returns_accumulation_when_declared() -> None:
    block = _make_block({"K": AxisRole.ACCUMULATION, "M": AxisRole.PARALLEL, "N": AxisRole.PARALLEL})
    assert role_of(block, "K") == AxisRole.ACCUMULATION
    assert role_of(block, "M") == AxisRole.PARALLEL


def test_role_of_unknown_axis_raises() -> None:
    block = _make_block({"P": AxisRole.PARALLEL})
    with pytest.raises(KeyError, match="does not declare axis"):
        role_of(block, "d99")
