"""Tests for nkigym.transforms.Split under BlockNode IR."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir, build_ladder_state
from test.transforms._helpers import block_for_op, first_for_in

import pytest

from nkigym.ir.arith.expr import Var, to_affine
from nkigym.ir.tree import BlockNode, ForNode, ISANode
from nkigym.transforms import Split, SplitOption, TransformLegalityError


def test_split_outer_trip_replaces_for_with_chain():
    """Splitting a ForNode trip 16 by factors=(4, 4) gives a 4 -> 4 chain."""
    ir = build_canonical_ir()
    matmul_block_nid = block_for_op(ir, "NKIMatmul")
    target = first_for_in(ir, matmul_block_nid)
    target_extent = ir.tree.loop(target).extent

    new_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(4, target_extent // 4)))

    """Old IR untouched."""
    assert ir.tree.loop(target).extent == target_extent

    """New IR: parent's child slot now contains a fresh ForNode of extent 4 with one ForNode child."""
    parent = ir.tree.parent(target)
    assert parent is not None
    new_kid = new_ir.tree.children(parent)[0]
    new_kid_data = new_ir.tree.data(new_kid)
    assert isinstance(new_kid_data, ForNode)
    assert new_kid_data.extent == 4
    inner = new_ir.tree.children(new_kid)[0]
    assert isinstance(new_ir.tree.data(inner), ForNode)
    assert new_ir.tree.loop(inner).extent == target_extent // 4


def test_split_outer_trip_rewrites_iter_value_for_bound_axis():
    """The enclosing block's iter_value for the split iter_var becomes a sum of new loop_vars * strides."""
    ir = build_canonical_ir()
    matmul_block_nid = block_for_op(ir, "NKIMatmul")
    matmul_block = ir.tree.block(matmul_block_nid)
    target = first_for_in(ir, matmul_block_nid)
    target_for = ir.tree.loop(target)
    target_loop_var = target_for.loop_var
    target_extent = target_for.extent

    """Identify which iter_var was bound by the original loop_var."""
    bound_axis_index = None
    for i, value in enumerate(matmul_block.iter_values):
        if isinstance(value, Var) and value.name == target_loop_var:
            bound_axis_index = i
            break
    assert bound_axis_index is not None, "could not locate the iter_value bound by the target ForNode"

    new_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(4, target_extent // 4)))
    new_block = new_ir.tree.block(matmul_block_nid)
    new_value = new_block.iter_values[bound_axis_index]
    coeffs = to_affine(new_value)
    """The new value is a 2-term affine combination summing two loop_vars."""
    var_terms = {k: v for k, v in coeffs.items() if k is not None}
    assert len(var_terms) == 2
    """Coefficients match outer * inner_extent + inner."""
    assert sorted(var_terms.values()) == [1, target_extent // 4]


def test_split_apply_preserves_input_ir():
    """``apply`` must not mutate its input IR."""
    ir = build_canonical_ir()
    matmul_block_nid = block_for_op(ir, "NKIMatmul")
    target = first_for_in(ir, matmul_block_nid)
    target_extent = ir.tree.loop(target).extent
    snapshot_num_nodes = ir.tree.num_nodes
    Split().apply(ir, SplitOption(target_nid=target, factors=(4, target_extent // 4)))
    assert ir.tree.num_nodes == snapshot_num_nodes


def test_split_rejects_factor_product_mismatch():
    ir = build_canonical_ir()
    matmul_block_nid = block_for_op(ir, "NKIMatmul")
    target = first_for_in(ir, matmul_block_nid)
    with pytest.raises(TransformLegalityError):
        Split().apply(ir, SplitOption(target_nid=target, factors=(3, 5)))


def test_split_rejects_outer_trip_on_shared_post_computeat_loop():
    """Outer-trip Split of a loop a prior CodeMotion made shared across two blocks
    is rejected (would rewrite only the enclosing block, leaving the nested
    sibling's index stale -> sim OOB / wrong accumulation).

    ``build_ladder_state(2)`` sinks the lhs_T load under the matmul's loop nest,
    so loop ``i_d0_0`` (the matmul K loop) now encloses both the matmul leaf and
    the nested load leaf. Splitting it must raise, and ``analyze`` must not offer
    it. The legal route is to Split the per-op loop BEFORE the CodeMotion.
    """
    ir = build_ladder_state(2)

    def owners_under(loop_nid):
        return {
            next(a for a in reversed(ir.tree.ancestors(d)) if isinstance(ir.tree.data(a), BlockNode))
            for d in ir.tree.descendants(loop_nid)
            if isinstance(ir.tree.data(d), ISANode)
        }

    """The shared loop is whichever ForNode encloses ISA leaves of 2+ blocks."""
    shared = next(
        nid for nid in ir.tree.preorder() if isinstance(ir.tree.data(nid), ForNode) and len(owners_under(nid)) >= 2
    )
    assert len(owners_under(shared)) >= 2, owners_under(shared)

    extent = ir.tree.loop(shared).extent
    with pytest.raises(TransformLegalityError, match="shared across multiple blocks"):
        Split().apply(ir, SplitOption(target_nid=shared, factors=(2, extent // 2)))
    assert not any(o.target_nid == shared and o.target_axis is None for o in Split().analyze(ir))


def test_split_tensorize_preserves_sibling_order_of_co_located_block():
    """Tensorizing a leaf that precedes a co-located sibling block must keep it
    BEFORE that sibling (sibling order is dataflow).

    ``build_ladder_state(7)`` co-locates the memset and matmul blocks under a
    shared loop, memset first. Splitting the memset leaf inserts loops above it;
    a naive splice appends the new chain to the parent's child list, moving the
    memset AFTER the matmul (zeroing the accumulator post-compute -> all-zeros
    output). The memset's ISA leaf must still pre-order BEFORE the matmul leaf.
    """

    def first_index(tree, op_name):
        return next(
            i
            for i, n in enumerate(tree.preorder())
            if isinstance(tree.data(n), ISANode) and tree.isa(n).op_cls.__name__ == op_name
        )

    ir = build_ladder_state(7)
    memset_leaf = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.isa(n).op_cls.__name__ == "NKIMemset"
    )
    assert first_index(ir.tree, "NKIMemset") < first_index(ir.tree, "NKIMatmul"), "precondition: memset before matmul"

    new_ir = Split().apply(ir, SplitOption(target_nid=memset_leaf, factors=(2, 256), target_axis="d2"))
    assert first_index(new_ir.tree, "NKIMemset") < first_index(
        new_ir.tree, "NKIMatmul"
    ), "memset must stay before matmul after tensorize split"


def test_split_analyze_offers_tensorize_on_load():
    """Split.analyze must offer a tensorize-flavor (target_axis set) option on the load leaf,
    whose d1 free-axis tile is width-2048 and factorizable to 16x128. Regression: the
    concrete(d1)-vs-abstract(F) axis-name mismatch made _current_tensorize_width return None."""
    ir = build_canonical_ir()
    load_leaf = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.isa(n).op_cls.__name__ == "NKILoad"
    )
    tensorize_opts = [o for o in Split().analyze(ir) if o.target_nid == load_leaf and o.target_axis is not None]
    assert tensorize_opts, "Split.analyze offered no tensorize option on the load leaf"
    """The d1 (concrete) free axis has width 2048 → factorizations include (16, 128)."""
    assert any(o.factors == (16, 128) for o in tensorize_opts), [o.factors for o in tensorize_opts]


def test_split_tensorize_below_min_tile_rejected():
    """A tensorize-split whose innermost factor < the axis MIN_TILE_SIZE is illegal.

    The matmul M axis (d1) is the PSUM partition axis at tile 128 with
    MIN_TILE_SIZE 128, so factors=(16,8) (final tile 8) must be rejected by
    apply's legality check — it would otherwise render a sub-128 partition tile.
    """
    ir = build_canonical_ir()
    mm = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.isa(n).op_cls.__name__ == "NKIMatmul"
    )
    with pytest.raises(TransformLegalityError, match="MIN_TILE_SIZE"):
        Split().apply(ir, SplitOption(target_nid=mm, factors=(16, 8), target_axis="d1"))


def test_split_analyze_omits_below_min_tensorize_splits():
    """analyze never offers a tensorize-split whose final factor < the axis MIN_TILE_SIZE.

    The matmul K (d0) and M (d1) axes are at tile 128 = MIN, so they admit no
    tensorize-split at all; N (d2) at tile 512 admits only finals >= 128
    (i.e. (4,128); (2,256); (2,2,128)). No option may shrink a partition/
    contraction tile below the hardware floor.
    """
    ir = build_canonical_ir()
    mm = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.isa(n).op_cls.__name__ == "NKIMatmul"
    )
    block = next(ir.tree.block(a) for a in reversed(ir.tree.ancestors(mm)) if isinstance(ir.tree.data(a), BlockNode))
    inverse = {concrete: abstract for abstract, concrete in block.axis_map.items()}
    min_tile = ir.tree.isa(mm).op_cls.MIN_TILE_SIZE
    tensorize = [o for o in Split().analyze(ir) if o.target_nid == mm and o.target_axis is not None]
    for opt in tensorize:
        target_axis = opt.target_axis
        assert target_axis is not None
        floor = min_tile[inverse[target_axis]]
        assert opt.factors[-1] >= floor, f"analyze offered {opt.factors} on {opt.target_axis} below floor {floor}"
    offered_axes = {o.target_axis for o in tensorize}
    assert "d0" not in offered_axes and "d1" not in offered_axes
    assert ("d2", (4, 128)) in {(o.target_axis, o.factors) for o in tensorize}


def test_split_rejects_over_cover():
    """Factors whose product EXCEEDS the extent are illegal (we are exact-division
    only — TVM would predicate the ragged tail; we reject). 4*5=20 > 16."""
    ir = build_canonical_ir()
    matmul_block_nid = block_for_op(ir, "NKIMatmul")
    target = first_for_in(ir, matmul_block_nid)
    assert ir.tree.loop(target).extent == 16
    with pytest.raises(TransformLegalityError):
        Split().apply(ir, SplitOption(target_nid=target, factors=(4, 5)))


def test_split_tensorize_rejects_over_cover():
    """The tensorize-flavour cover check also rejects over-cover via _covers_exactly.

    Matmul N (d2) tile is 512. factors=(4,256) over-covers (1024 > 512); its final
    tile 256 >= MIN_TILE_SIZE 128 so it clears the floor guard — the cover check
    is what must reject it. Confirms _covers_exactly is wired into BOTH legality
    branches (outer-trip and tensorize), not just the outer-trip path.
    """
    ir = build_canonical_ir()
    mm = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.isa(n).op_cls.__name__ == "NKIMatmul"
    )
    with pytest.raises(TransformLegalityError, match="exactly tile"):
        Split().apply(ir, SplitOption(target_nid=mm, factors=(4, 256), target_axis="d2"))
