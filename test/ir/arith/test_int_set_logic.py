"""TVM-independent IntSet interval-math tests (mirrors TVM ``src/arith/int_set.cc``).

These exercise the pure-nkigym halves of the IntSet skeleton: :meth:`IntSet.eval`
over a const-bounded affine expression and :meth:`IntSet.union` /
:meth:`IntSet.intersect` interval algebra. Every assertion compares against a
literal :class:`Const` endpoint, so none of them needs the TVM oracle and they
run in a TVM-less environment. The matching TVM cross-checks live in
``test_int_set.py`` behind its ``importorskip``.
"""

from nkigym.ir.arith.analyzer import Analyzer
from nkigym.ir.arith.expr import Add, Const, Mul, Sub, Var
from nkigym.ir.arith.int_set import IntSet


def test_eval_affine_over_bound_var():
    """EvalSet(x*512 + j) with x in [0,4), j in [0,512) -> interval [0, 2047]."""
    a = Analyzer()
    a.bind("x", 0, 4)
    a.bind("j", 0, 512)
    e = Add(left=Mul(left=Var(name="x"), right=Const(value=512)), right=Var(name="j"))
    s = IntSet.eval(e, a)
    assert s.min_value == Const(value=0)
    assert s.max_value == Const(value=2047)


def test_union_intersect():
    """Union of [0,10] and [5,20] -> [0,20]; Intersect -> [5,10]."""
    a = IntSet.interval(Const(value=0), Const(value=10))
    b = IntSet.interval(Const(value=5), Const(value=20))
    u = IntSet.union(a, b)
    i = IntSet.intersect(a, b)
    assert u.min_value == Const(value=0)
    assert u.max_value == Const(value=20)
    assert i.min_value == Const(value=5)
    assert i.max_value == Const(value=10)


def test_eval_affine_two_vars_literal():
    """EvalSet(x*512 + j) over x in [0,4), j in [0,512) -> literal [0, 2047]."""
    a = Analyzer()
    a.bind("x", 0, 4)
    a.bind("j", 0, 512)
    e = Add(left=Mul(left=Var(name="x"), right=Const(value=512)), right=Var(name="j"))
    ours = IntSet.eval(e, a)
    assert ours.min_value == Const(value=0)
    assert ours.max_value == Const(value=2047)


def test_eval_sign_cases_literal():
    """Sign-sensitive EvalSet endpoints match literal expectations.

    A negative ``Mul`` factor swaps the interval endpoints, and ``Sub(c, x)``
    reflects ``x``'s interval about ``c``; the two-term affine case sums the
    per-term ranges. Each binds half-open ranges and asserts the propagated
    ``[min, max]`` against literal :class:`Const` endpoints (no TVM oracle).
    """
    cases = [
        (Mul(left=Var(name="i"), right=Const(value=-2)), {"i": (0, 4)}, -6, 0),
        (Sub(left=Const(value=5), right=Var(name="x")), {"x": (2, 10)}, -4, 3),
        (
            Add(
                left=Mul(left=Var(name="a"), right=Const(value=4)),
                right=Mul(left=Var(name="b"), right=Const(value=128)),
            ),
            {"a": (0, 2), "b": (0, 4)},
            0,
            388,
        ),
    ]
    for expr, ranges, expected_min, expected_max in cases:
        a = Analyzer()
        for name, (lo, hi) in ranges.items():
            a.bind(name, lo, hi)
        ours = IntSet.eval(expr, a)
        assert ours.min_value == Const(value=expected_min)
        assert ours.max_value == Const(value=expected_max)
