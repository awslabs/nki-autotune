"""Tests for nkigym.ir.expr."""

from __future__ import annotations

import pytest

from nkigym.ir.expr import Add, Const, Mul, NonAffineError, Var, from_affine, substitute, to_affine


def test_const_and_var_construction():
    """Const and Var are frozen dataclasses with structural equality."""
    assert Const(value=3) == Const(value=3)
    assert Const(value=3) != Const(value=4)
    assert Var(name="i") == Var(name="i")
    assert Var(name="i") != Var(name="j")


def test_compound_expression_construction():
    """Compound expressions compose Add/Mul/FloorDiv/Mod recursively."""
    expr = Add(left=Mul(left=Var(name="i"), right=Const(value=8)), right=Var(name="j"))
    assert isinstance(expr, Add)
    assert isinstance(expr.left, Mul)
    assert expr.right == Var(name="j")


def test_expr_is_hashable():
    """Frozen dataclasses are hashable; equal exprs hash equal."""
    e1 = Add(left=Var(name="i"), right=Const(value=1))
    e2 = Add(left=Var(name="i"), right=Const(value=1))
    assert hash(e1) == hash(e2)
    assert {e1: 1, e2: 2} == {e1: 2}


def test_to_affine_const_only():
    """A bare Const collapses to {None: value}."""
    assert to_affine(Const(value=7)) == {None: 7}


def test_to_affine_var_only():
    """A bare Var collapses to {name: 1}."""
    assert to_affine(Var(name="i")) == {"i": 1}


def test_to_affine_linear_combination():
    """i*8 + j collapses to {'i': 8, 'j': 1}."""
    expr = Add(left=Mul(left=Var(name="i"), right=Const(value=8)), right=Var(name="j"))
    assert to_affine(expr) == {"i": 8, "j": 1}


def test_to_affine_zero_coefficients_dropped():
    """0*i + 5 collapses to {None: 5} (zero coefficients are pruned)."""
    expr = Add(left=Mul(left=Const(value=0), right=Var(name="i")), right=Const(value=5))
    assert to_affine(expr) == {None: 5}


def test_to_affine_rejects_var_times_var():
    """Var * Var is non-affine."""
    with pytest.raises(NonAffineError):
        to_affine(Mul(left=Var(name="i"), right=Var(name="j")))


def test_to_affine_rejects_var_in_mod_divisor():
    """Mod with a non-Const divisor is non-affine."""
    from nkigym.ir.expr import Mod

    with pytest.raises(NonAffineError):
        to_affine(Mod(left=Var(name="i"), right=Var(name="j")))


def test_from_affine_round_trip():
    """from_affine(to_affine(e)) is structurally equal to a canonical form of e."""
    expr = Add(left=Mul(left=Var(name="i"), right=Const(value=8)), right=Var(name="j"))
    coeffs = to_affine(expr)
    rebuilt = from_affine(coeffs)
    assert to_affine(rebuilt) == coeffs


def test_from_affine_constant_only():
    """from_affine({None: 5}) is Const(5)."""
    assert from_affine({None: 5}) == Const(value=5)


def test_from_affine_single_var():
    """from_affine({'i': 1}) is Var(i); from_affine({'i': 3}) is Mul(Var(i), Const(3))."""
    assert from_affine({"i": 1}) == Var(name="i")
    assert from_affine({"i": 3}) == Mul(left=Var(name="i"), right=Const(value=3))


def test_from_affine_empty_is_zero():
    """from_affine({}) == Const(0) (empty sum)."""
    assert from_affine({}) == Const(value=0)


def test_substitute_simple_var():
    """substitute({'i': Const(7)}) into Var('i') returns Const(7)."""
    assert substitute(Var(name="i"), {"i": Const(value=7)}) == Const(value=7)


def test_substitute_passes_through_other_vars():
    """substitute leaves non-substituted Vars alone."""
    expr = Add(left=Var(name="i"), right=Var(name="j"))
    result = substitute(expr, {"i": Const(value=7)})
    assert result == Add(left=Const(value=7), right=Var(name="j"))


def test_substitute_into_compound_expression():
    """Substituting i with i_outer*8 + i_inner inside i*128 + j gives the expected affine form."""
    expr = Add(left=Mul(left=Var(name="i"), right=Const(value=128)), right=Var(name="j"))
    sub = Add(left=Mul(left=Var(name="i_outer"), right=Const(value=8)), right=Var(name="i_inner"))
    result = substitute(expr, {"i": sub})
    """The result, when normalised, should equal i_outer*1024 + i_inner*128 + j."""
    expected_coeffs = {"i_outer": 1024, "i_inner": 128, "j": 1}
    assert to_affine(result) == expected_coeffs


def test_substitute_unaffected_passes_through():
    """substitute on a Const returns the same Const."""
    assert substitute(Const(value=5), {"i": Const(value=7)}) == Const(value=5)


def test_format_const():
    """Const(5) formats as '5'."""
    from nkigym.ir.expr import format_expr

    assert format_expr(Const(value=5)) == "5"


def test_format_var():
    """Var('i') formats as 'i'."""
    from nkigym.ir.expr import format_expr

    assert format_expr(Var(name="i")) == "i"


def test_format_affine_combination():
    """An affine combination formats with terms in sorted-name order, then constant."""
    from nkigym.ir.expr import format_expr

    expr = Add(left=Mul(left=Var(name="i"), right=Const(value=8)), right=Var(name="j"))
    assert format_expr(expr) == "i * 8 + j"


def test_format_negative_constant():
    """Negative constants surface inline (not normalised away)."""
    from nkigym.ir.expr import format_expr

    expr = Add(left=Var(name="i"), right=Const(value=-3))
    assert format_expr(expr) == "i + -3"
