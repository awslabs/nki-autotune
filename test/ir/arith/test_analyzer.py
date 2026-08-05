"""Tests for the Analyzer facade.

These exercise the :class:`Analyzer` facade (bound binding + ``simplify`` +
``can_prove_equal``) against literal expectations.
"""

from nkigym.ir.arith.analyzer import Analyzer
from nkigym.ir.arith.expr import Add, Const, Mod, Mul, Var


def test_analyzer_simplifies_and_proves_affine_equalities():
    """The facade simplifies aligned modulo and proves reordered affine sums."""
    a = Analyzer()
    lhs = Add(left=Var(name="x"), right=Var(name="y"))
    rhs = Add(left=Var(name="y"), right=Var(name="x"))
    assert a.can_prove_equal(lhs, rhs) is True
    a.bind("x", 0, 128)
    e = Mod(
        left=Add(left=Mul(left=Var(name="x"), right=Const(value=512)), right=Const(value=3)), right=Const(value=512)
    )
    assert a.simplify(e) == Const(value=3)
    lhs = Add(left=Mul(left=Var(name="i"), right=Const(value=4)), right=Var(name="j"))
    rhs = Add(left=Var(name="j"), right=Mul(left=Var(name="i"), right=Const(value=4)))
    assert a.can_prove_equal(lhs, rhs) is True
