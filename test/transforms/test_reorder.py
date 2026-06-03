"""Tests for nkigym.transforms.Reorder under BlockNode IR."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

import pytest

from nkigym.ir.tree import ForNode
from nkigym.transforms import Reorder, ReorderOption, TransformLegalityError


def _first_two_adjacent_fors(ir):
    """Return (outer_nid, inner_nid) for the first parent-child ForNode pair."""
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if not isinstance(data, ForNode):
            continue
        kids = ir.tree.children(nid)
        if len(kids) != 1:
            continue
        kid_data = ir.tree.data(kids[0])
        if isinstance(kid_data, ForNode):
            return nid, kids[0]
    raise AssertionError("no adjacent ForNode pair")


def test_reorder_swaps_payloads():
    """Apply swaps the two ForNode payloads while keeping nids stable."""
    ir = build_canonical_ir()
    outer, inner = _first_two_adjacent_fors(ir)
    outer_data = ir.tree.data(outer)
    inner_data = ir.tree.data(inner)
    new_ir = Reorder().apply(ir, ReorderOption(outer_nid=outer, inner_nid=inner))
    assert new_ir.tree.data(outer) == inner_data
    assert new_ir.tree.data(inner) == outer_data


def test_reorder_self_inverse():
    """Apply twice returns the original payload."""
    ir = build_canonical_ir()
    outer, inner = _first_two_adjacent_fors(ir)
    opt = ReorderOption(outer_nid=outer, inner_nid=inner)
    new_ir = Reorder().apply(Reorder().apply(ir, opt), opt)
    assert new_ir.tree.data(outer) == ir.tree.data(outer)
    assert new_ir.tree.data(inner) == ir.tree.data(inner)


def test_reorder_rejects_sequential_role():
    """Reorder rejects a swap on a dim whose enclosing block declares SEQUENTIAL role."""
    from test.transforms._seq_fixture import build_seq_ir

    ir, outer, inner, _ = build_seq_ir()
    with pytest.raises(TransformLegalityError, match="SEQUENTIAL"):
        Reorder().apply(ir, ReorderOption(outer_nid=outer, inner_nid=inner))


def test_reorder_matches_tvm_structure():
    """Layer-B guard: our Reorder's resulting loop order matches TVM's own ``schedule.reorder``.

    Reorder is a PURE PAYLOAD-SWAP (no arith): ``apply`` swaps the two adjacent
    ForNode payloads, leaving every binding intact (loop vars travel with their
    payloads), so only the tree loop ORDER changes. This guard confronts that
    resulting outer->inner order against TVM's own TensorIR ``schedule.reorder``
    on the equivalent perfect nest (the Layer-B structural oracle).

    The canonical matmul block nests ``i_d0_0`` (K=16), ``i_d1_0`` (M=16),
    ``i_d2_0`` (N=4) outer->inner. We swap the adjacent ``d1``/``d2`` pair — both
    PARALLEL, so unambiguously legal — yielding ``[d0, d2, d1]`` with extents
    ``[16, 4, 16]``. TVM's ``reorder`` of the equivalent ``[16, 16, 4]`` nest with
    ``order=[0, 2, 1]`` must reproduce the same extents. Orthogonal to the
    byte-exact ladder gate.
    """
    pytest.importorskip("tvm")
    from test.transforms._oracle_helpers import enclosing_for_nids
    from test.transforms._tvm_struct_oracle import tvm_reorder_loopnest

    from nkigym.ir.tree import ISANode

    ir = build_canonical_ir()
    mm = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKIMatmul"
    )
    enclosing = enclosing_for_nids(ir, mm, "i_d")
    before_order = [(ir.tree.data(n).loop_var, ir.tree.data(n).extent) for n in enclosing]
    assert before_order == [("i_d0_0", 16), ("i_d1_0", 16), ("i_d2_0", 4)]

    """Swap the adjacent d1/d2 pair (both PARALLEL); d1 is the outer of the pair."""
    d1 = next(n for n in enclosing if ir.tree.data(n).loop_var == "i_d1_0")
    d2 = next(n for n in enclosing if ir.tree.data(n).loop_var == "i_d2_0")
    out = Reorder().apply(ir, ReorderOption(outer_nid=d1, inner_nid=d2))

    out_mm = next(
        n
        for n in out.tree.preorder()
        if isinstance(out.tree.data(n), ISANode) and out.tree.data(n).op_cls.__name__ == "NKIMatmul"
    )
    our_extents = [out.tree.data(n).extent for n in enclosing_for_nids(out, out_mm, "i_d")]

    """TVM reorders the equivalent [16, 16, 4] nest, placing original loop 2 (d2)
    before original loop 1 (d1): new outer->inner order = [d0, d2, d1]."""
    source_extents = [ext for _lv, ext in before_order]
    nest = tvm_reorder_loopnest(extents=source_extents, order=[0, 2, 1])
    assert our_extents == nest.extents == [16, 4, 16]
