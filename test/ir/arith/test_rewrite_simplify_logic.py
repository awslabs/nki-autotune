"""TVM-independent RewriteSimplifier tests (mirrors TVM ``src/arith/rewrite_simplify.cc``).

These assert the simplifier's output against literal :class:`Const` / :class:`Var`
forms, so none of them needs the TVM oracle and they run in a TVM-less
environment. The corpus mirrors the cases the oracle cross-checks in
``test_rewrite_simplify.py`` (constant folding, identity elimination, ``Mod`` /
``FloorDiv`` over an aligned ``Mul``, add-chain flattening) plus the two
bound-predicate ``can_prove`` checks (Split predicate elision). The matching TVM
cross-checks live in ``test_rewrite_simplify.py`` behind its ``importorskip``.
"""

from nkigym.ir.arith.expr import LT, Add, Const, FloorDiv, Mod, Mul, Var
from nkigym.ir.arith.rewrite_simplify import RewriteSimplifier


def test_simplify_constant_fold():
    """``2 + 3`` folds to the literal ``5``."""
    rs = RewriteSimplifier()
    assert rs.simplify(Add(left=Const(value=2), right=Const(value=3))) == Const(value=5)


def test_simplify_add_zero_identity():
    """``x + 0`` simplifies to ``x``."""
    rs = RewriteSimplifier()
    assert rs.simplify(Add(left=Var(name="x"), right=Const(value=0))) == Var(name="x")


def test_simplify_mul_one_identity():
    """``x * 1`` simplifies to ``x``."""
    rs = RewriteSimplifier()
    assert rs.simplify(Mul(left=Var(name="x"), right=Const(value=1))) == Var(name="x")


def test_simplify_mod_aligned_offset():
    """``(x*512 + 3) % 512`` simplifies to the literal ``3`` (aligned term drops)."""
    rs = RewriteSimplifier()
    expr = Mod(
        left=Add(left=Mul(left=Var(name="x"), right=Const(value=512)), right=Const(value=3)), right=Const(value=512)
    )
    assert rs.simplify(expr) == Const(value=3)


def test_simplify_mod_aligned_to_zero():
    """``(x*4) % 4`` simplifies to the literal ``0`` (fully aligned)."""
    rs = RewriteSimplifier()
    expr = Mod(left=Mul(left=Var(name="x"), right=Const(value=4)), right=Const(value=4))
    assert rs.simplify(expr) == Const(value=0)


def test_simplify_floordiv_aligned():
    """``(x*512) // 512`` simplifies to ``x``."""
    rs = RewriteSimplifier()
    expr = FloorDiv(left=Mul(left=Var(name="x"), right=Const(value=512)), right=Const(value=512))
    assert rs.simplify(expr) == Var(name="x")


def test_simplify_add_chain_flattens():
    """``(x + 2) + 3`` flattens to ``x + 5`` (nested const operands combine)."""
    rs = RewriteSimplifier()
    expr = Add(left=Add(left=Var(name="x"), right=Const(value=2)), right=Const(value=3))
    assert rs.simplify(expr) == Add(left=Var(name="x"), right=Const(value=5))


def test_can_prove_lt_with_bounds():
    """(i0*2 + i1) < 4 with i0 in [0,2), i1 in [0,2) -> provable True (Split predicate elision)."""
    rs = RewriteSimplifier()
    rs.bind("i0", 0, 2)
    rs.bind("i1", 0, 2)
    pred = LT(left=Add(left=Mul(left=Var(name="i0"), right=Const(value=2)), right=Var(name="i1")), right=Const(value=4))
    assert rs.can_prove(pred) is True


def test_cannot_prove_false_bound():
    """``i0 < 2`` with i0 in [0,4) is NOT provable (the bound does not imply it)."""
    rs = RewriteSimplifier()
    rs.bind("i0", 0, 4)
    pred = LT(left=Var(name="i0"), right=Const(value=2))
    assert rs.can_prove(pred) is False
