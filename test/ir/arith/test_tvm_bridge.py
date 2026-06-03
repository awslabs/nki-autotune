import pytest

tvm = pytest.importorskip("tvm")
from test.ir.arith._tvm_bridge import from_tvm, to_tvm

from nkigym.ir.arith.expr import Add, Const, Mod, Mul, Var


def test_roundtrip_affine():
    e = Add(left=Mul(left=Var(name="x"), right=Const(value=512)), right=Var(name="j"))
    assert from_tvm(to_tvm(e)) == e


def test_roundtrip_div_mod():
    e = Mod(
        left=Add(left=Mul(left=Var(name="x"), right=Const(value=512)), right=Const(value=3)), right=Const(value=512)
    )
    assert from_tvm(to_tvm(e)) == e
