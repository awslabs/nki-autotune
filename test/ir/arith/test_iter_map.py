"""Oracle-gated tests for the ported ``iter_map`` (``src/arith/iter_affine_map.cc``).

TVM cross-check: detection succeeds exactly when ``tvm.arith.detect_iter_map``
succeeds, and ``iter_map_simplify`` lowers to the same PrimExpr that
``tvm.arith.normalize_iter_map_to_expr`` produces (compared through the bridge).

The TVM-independent structural behaviours (the demand-driven corpus Split/Fuse
rely on: Split recombine collapses to one iter; Fuse is its inverse) live in
``test_iter_map_logic.py`` and run unconditionally.
"""

import pytest

tvm = pytest.importorskip("tvm")

from test.ir.arith._tvm_bridge import from_tvm, to_tvm

import tvm.tirx as T
from tvm import arith as tarith
from tvm import ir as tir_ir

from nkigym.ir.arith.expr import Add, Const, Expr, FloorDiv, Mod, Mul, Var
from nkigym.ir.arith.iter_map import detect_iter_map, iter_map_simplify

SPLIT_BINDING = Add(left=Mul(left=Var(name="i0"), right=Const(value=4)), right=Var(name="i1"))
SPLIT_RANGES: dict[str, tuple[int, int]] = {"i0": (0, 4), "i1": (0, 4)}

FUSE_HI = FloorDiv(left=Var(name="fused"), right=Const(value=4))
FUSE_LO = Mod(left=Var(name="fused"), right=Const(value=4))
FUSE_RANGES: dict[str, tuple[int, int]] = {"fused": (0, 16)}


def _tvm_detect(indices: list[Expr], ranges: dict[str, tuple[int, int]]) -> tuple[int, list["T.PrimExpr"]]:
    """Run TVM's ``detect_iter_map`` + ``normalize_iter_map_to_expr`` on the same input.

    Returns the number of detected indices and the list of normalized PrimExprs
    (empty when TVM fails to detect), bridging our :class:`Expr` to TVM and back
    through a shared variable environment so the ranges key the same ``T.Var``s.
    """
    env: dict[str, "T.Var"] = {}
    tvm_indices = [to_tvm(idx, env) for idx in indices]
    for name, (lo, hi) in ranges.items():
        if name not in env:
            env[name] = T.Var(name, "int32")
    dom = {env[name]: tir_ir.Range(lo, hi) for name, (lo, hi) in ranges.items()}
    res = tarith.detect_iter_map(tvm_indices, dom)
    normalized = [tarith.normalize_iter_map_to_expr(idx) for idx in res.indices]
    return len(res.indices), normalized


@pytest.mark.parametrize(
    "indices,ranges",
    [
        ([SPLIT_BINDING], SPLIT_RANGES),
        ([FUSE_HI, FUSE_LO], FUSE_RANGES),
        ([Var(name="i0"), Var(name="i0")], {"i0": (0, 4)}),
    ],
)
def test_iter_map_matches_tvm(indices: list[Expr], ranges: dict[str, tuple[int, int]]) -> None:
    """detect/simplify agree with TVM on success/failure and on normalized PrimExprs."""
    tvm_count, tvm_normalized = _tvm_detect(indices, ranges)
    ours = detect_iter_map(indices, ranges)
    if tvm_count == 0:
        assert ours is None
    else:
        assert ours is not None and len(ours) == tvm_count
        simplified = iter_map_simplify(indices, ranges)
        assert simplified is not None
        expected = [from_tvm(pe) for pe in tvm_normalized]
        assert simplified == expected, f"ours={simplified} tvm={expected}"
