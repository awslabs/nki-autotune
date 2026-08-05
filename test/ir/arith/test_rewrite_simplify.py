"""Tests for RewriteSimplifier.

These assert the simplifier's output against literal :class:`Const` / :class:`Var`
forms, including constant folding, identity elimination, aligned division and
modulo, add-chain flattening, and bound-predicate proofs.
"""

from nkigym.ir.arith.expr import LT, Add, Const, FloorDiv, Mod, Mul, Var
from nkigym.ir.arith.rewrite_simplify import RewriteSimplifier


def test_simplify_algebraic_and_aligned_forms():
    """The simplifier folds constants, identities, aligned arithmetic, and add chains."""
    rs = RewriteSimplifier()
    assert rs.simplify(Add(left=Const(value=2), right=Const(value=3))) == Const(value=5)
    assert rs.simplify(Add(left=Var(name="x"), right=Const(value=0))) == Var(name="x")
    assert rs.simplify(Mul(left=Var(name="x"), right=Const(value=1))) == Var(name="x")
    expr = Mod(
        left=Add(left=Mul(left=Var(name="x"), right=Const(value=512)), right=Const(value=3)), right=Const(value=512)
    )
    assert rs.simplify(expr) == Const(value=3)
    expr = Mod(left=Mul(left=Var(name="x"), right=Const(value=4)), right=Const(value=4))
    assert rs.simplify(expr) == Const(value=0)
    expr = FloorDiv(left=Mul(left=Var(name="x"), right=Const(value=512)), right=Const(value=512))
    assert rs.simplify(expr) == Var(name="x")
    expr = Add(left=Add(left=Var(name="x"), right=Const(value=2)), right=Const(value=3))
    assert rs.simplify(expr) == Add(left=Var(name="x"), right=Const(value=5))


def test_bound_proofs_accept_implied_and_reject_unproven_predicates():
    """Bound reasoning proves implied split predicates and rejects weaker bounds."""
    rs = RewriteSimplifier()
    rs.bind("i0", 0, 2)
    rs.bind("i1", 0, 2)
    pred = LT(left=Add(left=Mul(left=Var(name="i0"), right=Const(value=2)), right=Var(name="i1")), right=Const(value=4))
    assert rs.can_prove(pred) is True
    rs = RewriteSimplifier()
    rs.bind("i0", 0, 4)
    pred = LT(left=Var(name="i0"), right=Const(value=2))
    assert rs.can_prove(pred) is False
