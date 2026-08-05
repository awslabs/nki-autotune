"""Canonical IR contains no trip-1 ForNodes (the 'no trip-1 anywhere' rule)."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

from nkigym.ir.tree import ForNode


def test_canonical_has_no_trip1_loops():
    """Every ForNode in canonical IR has extent > 1; trip-1 axes are loopless
    (pure tensorize_size on the access)."""
    ir = build_canonical_ir()
    trip1 = [
        ir.tree.loop(n).loop_var
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ForNode) and ir.tree.loop(n).extent == 1
    ]
    assert trip1 == [], f"canonical still emits trip-1 loops: {trip1}"
