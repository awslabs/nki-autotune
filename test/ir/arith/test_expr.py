from nkigym.ir.arith.expr import EQ, LE, LT, Add, Const, Max, Min, Mul, Sub, Var, affine_terms, substitute


def test_new_nodes_construct_and_substitute():
    e = Sub(left=Var(name="x"), right=Const(value=1))
    assert isinstance(e, Sub)
    out = substitute(e, {"x": Const(value=5)})
    assert out == Sub(left=Const(value=5), right=Const(value=1))


def test_min_max_predicate_nodes():
    assert isinstance(Min(left=Var(name="a"), right=Var(name="b")), Min)
    assert isinstance(Max(left=Var(name="a"), right=Var(name="b")), Max)
    assert isinstance(LT(left=Var(name="i"), right=Const(value=128)), LT)
    assert isinstance(LE(left=Var(name="i"), right=Const(value=128)), LE)
    assert isinstance(EQ(left=Var(name="i"), right=Const(value=0)), EQ)


def test_predicate_substitute_recurses():
    p = LT(left=Add(left=Mul(left=Var(name="i"), right=Const(value=512)), right=Var(name="j")), right=Const(value=2048))
    out = substitute(p, {"i": Const(value=3)})
    assert out == LT(
        left=Add(left=Mul(left=Const(value=3), right=Const(value=512)), right=Var(name="j")), right=Const(value=2048)
    )


def test_affine_terms_decomposes_affine():
    """An affine expression decomposes to Var-keyed coefficients."""
    e = Add(left=Mul(left=Var(name="i"), right=Const(value=512)), right=Var(name="j"))
    assert affine_terms(e) == {Var(name="i"): 512, Var(name="j"): 1}


def test_affine_terms_carries_var_times_var_opaque():
    """A non-affine Var * Var becomes a single opaque key with coefficient 1."""
    e = Mul(left=Var(name="x"), right=Var(name="y"))
    assert affine_terms(e) == {Mul(left=Var(name="x"), right=Var(name="y")): 1}
