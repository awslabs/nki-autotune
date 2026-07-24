"""Structural tests for ``iter_map``.

The demand-driven Split/Fuse cases assert that Split recombination collapses to
one affine iterator and Fuse recovers the component iterators.
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
