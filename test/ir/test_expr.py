"""Tests for the legacy affine-expression compatibility helpers."""

from __future__ import annotations

import pytest

from nkigym.ir.arith.expr import (
    Add,
    Const,
    Mod,
    Mul,
    NonAffineError,
    Var,
    format_expr,
    from_affine,
    substitute,
    to_affine,
)


def test_to_affine_contract() -> None:
    """Affine conversion handles constants and sums and rejects nonlinear forms."""
    assert to_affine(Const(value=7)) == {None: 7}
    assert to_affine(Var(name="i")) == {"i": 1}
    expr = Add(left=Mul(left=Var(name="i"), right=Const(value=8)), right=Var(name="j"))
    assert to_affine(expr) == {"i": 8, "j": 1}
    zero_term = Add(left=Mul(left=Const(value=0), right=Var(name="i")), right=Const(value=5))
    assert to_affine(zero_term) == {None: 5}
    with pytest.raises(NonAffineError):
        to_affine(Mul(left=Var(name="i"), right=Var(name="j")))
    with pytest.raises(NonAffineError):
        to_affine(Mod(left=Var(name="i"), right=Var(name="j")))


def test_from_affine_contract() -> None:
    """Affine reconstruction handles canonical sums, constants, variables, and zero."""
    expr = Add(left=Mul(left=Var(name="i"), right=Const(value=8)), right=Var(name="j"))
    coefficients = to_affine(expr)
    assert to_affine(from_affine(coefficients)) == coefficients
    assert from_affine({None: 5}) == Const(value=5)
    assert from_affine({"i": 1}) == Var(name="i")
    assert from_affine({"i": 3}) == Mul(left=Var(name="i"), right=Const(value=3))
    assert from_affine({}) == Const(value=0)


def test_substitution_contract() -> None:
    """Substitution recursively replaces selected variables and preserves others."""
    assert substitute(Var(name="i"), {"i": Const(value=7)}) == Const(value=7)
    expr = Add(left=Var(name="i"), right=Var(name="j"))
    assert substitute(expr, {"i": Const(value=7)}) == Add(left=Const(value=7), right=Var(name="j"))

    compound = Add(left=Mul(left=Var(name="i"), right=Const(value=128)), right=Var(name="j"))
    replacement = Add(left=Mul(left=Var(name="i_outer"), right=Const(value=8)), right=Var(name="i_inner"))
    assert to_affine(substitute(compound, {"i": replacement})) == {"i_outer": 1024, "i_inner": 128, "j": 1}
    assert substitute(Const(value=5), {"i": Const(value=7)}) == Const(value=5)


def test_format_contract() -> None:
    """Formatting covers scalar nodes, affine sums, and negative constants."""
    assert format_expr(Const(value=5)) == "5"
    assert format_expr(Var(name="i")) == "i"
    affine = Add(left=Mul(left=Var(name="i"), right=Const(value=8)), right=Var(name="j"))
    assert format_expr(affine) == "i * 8 + j"
    negative = Add(left=Var(name="i"), right=Const(value=-3))
    assert format_expr(negative) == "i + -3"
