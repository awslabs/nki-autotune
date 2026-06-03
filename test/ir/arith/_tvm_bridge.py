"""Test-only bridge between our Expr AST and TVM PrimExpr (for the arith oracle).

Never imported by shipped ``nkigym/`` code — only by Layer-A oracle tests.
``int32`` throughout (our extents/indices are all int32-range).
"""

from __future__ import annotations

import tvm.tirx as T

from nkigym.ir.arith.expr import EQ, LE, LT, Add, Const, Expr, FloorDiv, Max, Min, Mod, Mul, Sub, Var

_I32 = "int32"


def to_tvm(expr: Expr, env: dict[str, "T.Var"] | None = None) -> "T.PrimExpr":
    """Lower our Expr to a tvm.tirx PrimExpr. ``env`` interns Var names -> T.Var."""
    env = {} if env is None else env

    def rec(e: Expr):
        if isinstance(e, Const):
            return T.IntImm(_I32, e.value)
        if isinstance(e, Var):
            if e.name not in env:
                env[e.name] = T.Var(e.name, _I32)
            return env[e.name]
        if isinstance(e, Add):
            return T.Add(rec(e.left), rec(e.right))
        if isinstance(e, Sub):
            return T.Sub(rec(e.left), rec(e.right))
        if isinstance(e, Mul):
            return T.Mul(rec(e.left), rec(e.right))
        if isinstance(e, FloorDiv):
            return T.FloorDiv(rec(e.left), rec(e.right))
        if isinstance(e, Mod):
            return T.FloorMod(rec(e.left), rec(e.right))
        if isinstance(e, Min):
            return T.Min(rec(e.left), rec(e.right))
        if isinstance(e, Max):
            return T.Max(rec(e.left), rec(e.right))
        if isinstance(e, LT):
            return T.LT(rec(e.left), rec(e.right))
        if isinstance(e, LE):
            return T.LE(rec(e.left), rec(e.right))
        if isinstance(e, EQ):
            return T.EQ(rec(e.left), rec(e.right))
        raise TypeError(f"to_tvm: unknown node {type(e).__name__}")

    return rec(expr)


def from_tvm(pe: "T.PrimExpr") -> Expr:
    """Lift a tvm.tirx PrimExpr back to our Expr (inverse of to_tvm for the supported subset)."""
    if isinstance(pe, T.IntImm):
        return Const(value=int(pe.value))
    if isinstance(pe, T.Var):
        return Var(name=pe.name)
    binops = {
        T.Add: Add,
        T.Sub: Sub,
        T.Mul: Mul,
        T.FloorDiv: FloorDiv,
        T.FloorMod: Mod,
        T.Min: Min,
        T.Max: Max,
        T.LT: LT,
        T.LE: LE,
        T.EQ: EQ,
    }
    result: Expr | None = None
    for tvm_cls, our_cls in binops.items():
        if isinstance(pe, tvm_cls):
            result = our_cls(left=from_tvm(pe.a), right=from_tvm(pe.b))
            break
    if result is None:
        raise TypeError(f"from_tvm: unsupported PrimExpr {type(pe).__name__}")
    return result
