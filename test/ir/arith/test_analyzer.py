import pytest

tvm = pytest.importorskip("tvm")
from test.ir.arith._tvm_bridge import from_tvm, to_tvm

from tvm import arith as tarith

from nkigym.ir.arith.analyzer import Analyzer
from nkigym.ir.arith.expr import Add, Const, Mod, Mul, Var


def test_simplify_matches_tvm():
    a = Analyzer()
    a.bind("x", 0, 128)
    e = Mod(
        left=Add(left=Mul(left=Var(name="x"), right=Const(value=512)), right=Const(value=3)), right=Const(value=512)
    )
    ours = a.simplify(e)
    ta = tarith.Analyzer()
    expected = from_tvm(ta.simplify(to_tvm(e)))
    assert ours == expected == Const(value=3)


def test_can_prove_equal_matches_tvm():
    a = Analyzer()
    lhs = Add(left=Mul(left=Var(name="i"), right=Const(value=4)), right=Var(name="j"))
    rhs = Add(left=Var(name="j"), right=Mul(left=Var(name="i"), right=Const(value=4)))
    ours = a.can_prove_equal(lhs, rhs)
    ta = tarith.Analyzer()
    env: dict = {}
    tvm_res = bool(ta.can_prove_equal(to_tvm(lhs, env), to_tvm(rhs, env)))
    assert ours == tvm_res is True
