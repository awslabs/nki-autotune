"""Oracle-gated tests for the IntSet skeleton (mirrors TVM ``src/arith/int_set.cc``).

The cross-check evaluates the same affine expression and variable domain through
TVM's ``Analyzer.int_set`` (the Python entry point onto ``IntervalSetEvaluator``)
and asserts our ``[min, max]`` interval equals TVM's. TVM's ``dom_map`` carries
inclusive ``[min, max]`` intervals, so a half-open ``Analyzer.bind(name, lo, hi)``
maps to TVM's ``IntervalSet(lo, hi - 1)``.

The TVM-independent halves of these checks (literal interval-math assertions)
live in ``test_int_set_logic.py`` and run unconditionally.
"""

import pytest

tvm = pytest.importorskip("tvm")

from test.ir.arith._tvm_bridge import to_tvm

import tvm.tirx as T
from tvm import arith as tarith

from nkigym.ir.arith.analyzer import Analyzer
from nkigym.ir.arith.expr import Add, Const, Mul, Sub, Var
from nkigym.ir.arith.int_set import IntSet


def test_eval_matches_tvm():
    """Our EvalSet interval equals TVM's ``Analyzer.int_set`` over the same domain.

    Direct TVM IntSet eval: ``Analyzer.int_set(expr, dom_map)`` runs the
    ``IntervalSetEvaluator`` and returns an ``IntervalSet`` whose simplified
    ``min_value`` / ``max_value`` are the propagated interval endpoints.
    """
    a = Analyzer()
    a.bind("x", 0, 4)
    a.bind("j", 0, 512)
    e = Add(left=Mul(left=Var(name="x"), right=Const(value=512)), right=Var(name="j"))
    ours = IntSet.eval(e, a)

    ta = tarith.Analyzer()
    env: dict[str, "T.Var"] = {}
    tvm_expr = to_tvm(e, env)
    dom_map = {
        env["x"]: tarith.IntervalSet(T.IntImm("int32", 0), T.IntImm("int32", 3)),
        env["j"]: tarith.IntervalSet(T.IntImm("int32", 0), T.IntImm("int32", 511)),
    }
    tvm_set = ta.int_set(tvm_expr, dom_map)
    tvm_min = int(ta.simplify(tvm_set.min_value).value)
    tvm_max = int(ta.simplify(tvm_set.max_value).value)

    assert ours.min_value == Const(value=tvm_min)
    assert ours.max_value == Const(value=tvm_max)


def test_eval_sign_cases_match_tvm():
    """Sign-sensitive EvalSet endpoints match TVM (Mul-by-negative, Sub-endpoint-swap).

    These are the classic IntSet sign bugs: a negative ``Mul`` factor swaps the
    interval endpoints, and ``Sub(c, x)`` reflects ``x``'s interval about ``c``.
    Each case binds half-open ranges (TVM's inclusive ``IntervalSet(lo, hi - 1)``)
    and cross-checks our ``[min, max]`` against TVM's ``Analyzer.int_set``.
    """
    cases = [
        (Mul(left=Var(name="i"), right=Const(value=-2)), {"i": (0, 4)}),
        (Sub(left=Const(value=5), right=Var(name="x")), {"x": (2, 10)}),
        (
            Add(
                left=Mul(left=Var(name="a"), right=Const(value=4)),
                right=Mul(left=Var(name="b"), right=Const(value=128)),
            ),
            {"a": (0, 2), "b": (0, 4)},
        ),
    ]
    for expr, ranges in cases:
        a = Analyzer()
        for name, (lo, hi) in ranges.items():
            a.bind(name, lo, hi)
        ours = IntSet.eval(expr, a)

        ta = tarith.Analyzer()
        env: dict[str, "T.Var"] = {}
        tvm_expr = to_tvm(expr, env)
        dom_map = {
            env[name]: tarith.IntervalSet(T.IntImm("int32", lo), T.IntImm("int32", hi - 1))
            for name, (lo, hi) in ranges.items()
        }
        tvm_set = ta.int_set(tvm_expr, dom_map)
        tvm_min = int(ta.simplify(tvm_set.min_value).value)
        tvm_max = int(ta.simplify(tvm_set.max_value).value)
        assert (ours.min_value, ours.max_value) == (Const(value=tvm_min), Const(value=tvm_max))
