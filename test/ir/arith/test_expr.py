from nkigym.ir.arith.expr import Add, Const, Mul, Var, affine_terms


def test_affine_terms_preserve_affine_and_opaque_terms():
    """Affine terms decompose while nonlinear products remain opaque."""
    e = Add(left=Mul(left=Var(name="i"), right=Const(value=512)), right=Var(name="j"))
    assert affine_terms(e) == {Var(name="i"): 512, Var(name="j"): 1}
    e = Mul(left=Var(name="x"), right=Var(name="y"))
    assert affine_terms(e) == {Mul(left=Var(name="x"), right=Var(name="y")): 1}
