"""TVM-independent Analyzer tests (mirrors TVM ``src/arith/analyzer.cc``).

These exercise the :class:`Analyzer` facade (bound binding + ``simplify`` +
``can_prove_equal``) against literal expectations, so none of them needs the TVM
oracle and they run in a TVM-less environment. The matching TVM cross-checks
(``simplify`` against ``tvm.arith.Analyzer.simplify``; ``can_prove_equal``
against TVM) live in ``test_analyzer.py`` behind its ``importorskip``.
"""

from nkigym.ir.arith.analyzer import Analyzer
from nkigym.ir.arith.expr import Add, Const, Mod, Mul, Var


def test_can_prove_equal():
    """``x + y`` and ``y + x`` are provably equal (commutativity through simplify)."""
    a = Analyzer()
    lhs = Add(left=Var(name="x"), right=Var(name="y"))
    rhs = Add(left=Var(name="y"), right=Var(name="x"))
    assert a.can_prove_equal(lhs, rhs) is True


def test_simplify_mod_aligned_offset():
    """``(x*512 + 3) % 512`` simplifies to the literal ``3`` with x bound to [0,128)."""
    a = Analyzer()
    a.bind("x", 0, 128)
    e = Mod(
        left=Add(left=Mul(left=Var(name="x"), right=Const(value=512)), right=Const(value=3)), right=Const(value=512)
    )
    assert a.simplify(e) == Const(value=3)


def test_can_prove_equal_affine_reorder():
    """``i*4 + j`` and ``j + i*4`` are provably equal (operand reorder)."""
    a = Analyzer()
    lhs = Add(left=Mul(left=Var(name="i"), right=Const(value=4)), right=Var(name="j"))
    rhs = Add(left=Var(name="j"), right=Mul(left=Var(name="i"), right=Const(value=4)))
    assert a.can_prove_equal(lhs, rhs) is True
