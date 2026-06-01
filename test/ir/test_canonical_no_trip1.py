"""Canonical IR contains no trip-1 ForNodes (the 'no trip-1 anywhere' rule)."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

from nkigym.ir.tree import ForNode


def test_canonical_has_no_trip1_loops():
    """Every ForNode in canonical IR has extent > 1; trip-1 axes are loopless
    (pure tensorize_size on the access)."""
    ir = build_canonical_ir()
    trip1 = [
        ir.tree.data(n).loop_var
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ForNode) and ir.tree.data(n).extent == 1
    ]
    assert trip1 == [], f"canonical still emits trip-1 loops: {trip1}"


def test_canonical_load_d1_is_loopless_full_width():
    """The lhs_T load's d1 (M) axis is trip-1 -> no loop; its sbuf_lhs_T write spans
    the full 2048 free width in one access."""
    from nkigym.ir.tree import ISANode

    ir = build_canonical_ir()
    load = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKILoad"
    )
    """Only the d0 (K) loop encloses the load leaf; no d1 loop."""
    loops = [ir.tree.data(a).loop_var for a in ir.tree.ancestors(load) if isinstance(ir.tree.data(a), ForNode)]
    assert loops == ["i_d0_0"], loops
