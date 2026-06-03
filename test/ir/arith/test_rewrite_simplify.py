import pytest

tvm = pytest.importorskip("tvm")
from test.ir.arith._tvm_bridge import from_tvm, to_tvm

from tvm import arith as tarith

from nkigym.ir.arith.expr import LE, LT, Add, Const, FloorDiv, Mod, Mul, Var
from nkigym.ir.arith.rewrite_simplify import RewriteSimplifier

CASES = [
    Add(left=Const(value=2), right=Const(value=3)),
    Add(left=Var(name="x"), right=Const(value=0)),
    Mul(left=Var(name="x"), right=Const(value=1)),
    Mod(left=Add(left=Mul(left=Var(name="x"), right=Const(value=512)), right=Const(value=3)), right=Const(value=512)),
    FloorDiv(left=Mul(left=Var(name="x"), right=Const(value=512)), right=Const(value=512)),
    Add(left=Add(left=Var(name="x"), right=Const(value=2)), right=Const(value=3)),
    Add(left=Add(left=Mul(left=Var(name="i"), right=Const(value=512)), right=Var(name="j")), right=Const(value=5)),
    Mod(left=Mul(left=Var(name="x"), right=Const(value=4)), right=Const(value=4)),
    Mod(left=Mul(left=Var(name="x"), right=Const(value=512)), right=Const(value=256)),
]


@pytest.mark.parametrize("expr", CASES)
def test_simplify_matches_tvm(expr):
    ours = RewriteSimplifier().simplify(expr)
    a = tarith.Analyzer()
    tvm_simplified = from_tvm(a.simplify(to_tvm(expr)))
    assert ours == tvm_simplified, f"{ours} != {tvm_simplified}"


def test_can_prove_matches_tvm_oracle():
    """Cross-check can_prove against tvm.arith on a few bound predicates."""
    import pytest

    pytest.importorskip("tvm")
    from test.ir.arith._tvm_bridge import to_tvm

    import tvm.tirx as T
    from tvm import arith as tarith
    from tvm import ir as tir_ir

    specs = [
        (
            {"i0": (0, 2), "i1": (0, 2)},
            LT(
                left=Add(left=Mul(left=Var(name="i0"), right=Const(value=2)), right=Var(name="i1")),
                right=Const(value=4),
            ),
            True,
        ),
        ({"i0": (0, 4)}, LT(left=Var(name="i0"), right=Const(value=2)), False),
        ({"x": (0, 128)}, LT(left=Var(name="x"), right=Const(value=128)), True),
        ({"x": (0, 128)}, LE(left=Var(name="x"), right=Const(value=127)), True),
    ]
    for ranges, pred, _expected in specs:
        rs = RewriteSimplifier()
        for nm, (lo, hi) in ranges.items():
            rs.bind(nm, lo, hi)
        ours = rs.can_prove(pred)
        env = {}
        a = tarith.Analyzer()
        for nm, (lo, hi) in ranges.items():
            v = T.Var(nm, "int32")
            env[nm] = v
            a.bind(v, tir_ir.Range(lo, hi))
        tvm_res = bool(a.can_prove(to_tvm(pred, env)))
        assert ours == tvm_res, f"{pred}: ours={ours} tvm={tvm_res}"
