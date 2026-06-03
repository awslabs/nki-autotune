"""TVM-independent ``iter_map`` structural tests (mirrors TVM ``src/arith/iter_affine_map.cc``).

``iter_map_simplify`` returns its detected iterator list without the TVM oracle,
so the two structural demand-driven corpus behaviours (Split recombine collapses
to one affine iter; Fuse is the inverse of Split, recovering two iters) are
asserted here against literal lengths and run in a TVM-less environment. The
oracle cross-check (detection succeeds exactly when ``tvm.arith.detect_iter_map``
succeeds and normalizes to the same PrimExpr) lives in ``test_iter_map.py``
behind its ``importorskip``.
"""

from nkigym.ir.arith.expr import Add, Const, FloorDiv, Mod, Mul, Var
from nkigym.ir.arith.iter_map import iter_map_simplify

SPLIT_BINDING = Add(left=Mul(left=Var(name="i0"), right=Const(value=4)), right=Var(name="i1"))
SPLIT_RANGES: dict[str, tuple[int, int]] = {"i0": (0, 4), "i1": (0, 4)}

FUSE_HI = FloorDiv(left=Var(name="fused"), right=Const(value=4))
FUSE_LO = Mod(left=Var(name="fused"), right=Const(value=4))
FUSE_RANGES: dict[str, tuple[int, int]] = {"fused": (0, 16)}


def test_split_recombine_collapses() -> None:
    """Split: i0*4 + i1 over i0 in [0,4), i1 in [0,4) is one affine iter of extent 16."""
    out = iter_map_simplify([SPLIT_BINDING], SPLIT_RANGES)
    assert out is not None and len(out) == 1


def test_fuse_split_inverse() -> None:
    """Fuse: (fused//4, fused%4) over fused in [0,16) recovers two iters of extent 4."""
    out = iter_map_simplify([FUSE_HI, FUSE_LO], FUSE_RANGES)
    assert out is not None and len(out) == 2
