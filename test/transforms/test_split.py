"""Tests for :class:`nkigym.transforms.Split`."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

import pytest

from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms import Split, SplitOption, TransformLegalityError


def _find_first_for_with_trip(ir, trip: int) -> int:
    """Return the nid of the first ``ForNode`` with the given trip count."""
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if isinstance(data, ForNode) and data.trip == trip:
            return nid
    raise AssertionError(f"no ForNode with trip={trip}")


def test_split_outer_trip_apply_changes_structure():
    """Splitting a ``ForNode`` with trip=16 by factors=(4, 4) replaces it with two nested ForNodes."""
    ir = build_canonical_ir()
    target = _find_first_for_with_trip(ir, 16)
    parent = ir.tree.parent(target)
    children_before = ir.tree.children(target)

    new_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(4, 4)))

    """The original IR is untouched (deep-copy)."""
    assert ir.tree.num_nodes < new_ir.tree.num_nodes or ir.tree.num_nodes == new_ir.tree.num_nodes
    assert isinstance(ir.tree.data(target), ForNode)
    assert ir.tree.data(target).trip == 16

    """In the new IR, the target was replaced with a (4 -> 4) chain."""
    new_for_nids = [
        nid
        for nid in new_ir.tree.preorder()
        if isinstance(new_ir.tree.data(nid), ForNode) and new_ir.tree.data(nid).trip == 4
    ]
    assert len(new_for_nids) >= 2

    """The deepest ForNode in the chain has the same number of children as the target had."""
    parent_in_new = parent
    deepest = new_for_nids[0]
    while True:
        kids = new_ir.tree.children(deepest)
        if not kids or not isinstance(new_ir.tree.data(kids[0]), ForNode) or new_ir.tree.data(kids[0]).trip != 4:
            break
        deepest = kids[0]
    assert len(new_ir.tree.children(deepest)) == len(children_before)


def test_split_apply_preserves_input_ir():
    """``apply`` must not mutate its input IR."""
    ir = build_canonical_ir()
    target = _find_first_for_with_trip(ir, 16)
    snapshot_num_nodes = ir.tree.num_nodes
    snapshot_data = {nid: ir.tree.data(nid) for nid in ir.tree.preorder()}

    Split().apply(ir, SplitOption(target_nid=target, factors=(4, 4)))

    assert ir.tree.num_nodes == snapshot_num_nodes
    for nid, data in snapshot_data.items():
        assert ir.tree.data(nid) == data


def test_split_outer_trip_factor_propagation():
    """Outer-trip factors flow into the new ForNodes outer->inner."""
    ir = build_canonical_ir()
    target = _find_first_for_with_trip(ir, 16)
    target_data = ir.tree.data(target)
    parent = ir.tree.parent(target)

    new_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(2, 8)))

    """Walk down from ``parent`` and find the first new ForNode of trip 2, then its child trip 8."""
    parent_kids = new_ir.tree.children(parent)
    outer_candidates = [
        nid
        for nid in parent_kids
        if isinstance(new_ir.tree.data(nid), ForNode)
        and new_ir.tree.data(nid).dim == target_data.dim
        and new_ir.tree.data(nid).trip == 2
    ]
    assert len(outer_candidates) == 1
    outer = outer_candidates[0]
    inner_kids = new_ir.tree.children(outer)
    assert len(inner_kids) == 1
    assert isinstance(new_ir.tree.data(inner_kids[0]), ForNode)
    assert new_ir.tree.data(inner_kids[0]).dim == target_data.dim
    assert new_ir.tree.data(inner_kids[0]).trip == 8


def test_split_rejects_non_for_target():
    """``apply`` raises ``TransformLegalityError`` if the target is not a ForNode."""
    ir = build_canonical_ir()
    isa_nids = [nid for nid in ir.tree.preorder() if isinstance(ir.tree.data(nid), ISANode)]
    assert isa_nids
    with pytest.raises(TransformLegalityError):
        Split().apply(ir, SplitOption(target_nid=isa_nids[0], factors=(2, 2)))


def test_split_rejects_factor_below_2():
    """Factors of 1 are rejected."""
    ir = build_canonical_ir()
    target = _find_first_for_with_trip(ir, 16)
    with pytest.raises(TransformLegalityError):
        Split().apply(ir, SplitOption(target_nid=target, factors=(1, 16)))


def test_split_rejects_non_divisor_factor_product():
    """``prod(factors)`` must equal ``target.trip``."""
    ir = build_canonical_ir()
    target = _find_first_for_with_trip(ir, 16)
    with pytest.raises(TransformLegalityError):
        Split().apply(ir, SplitOption(target_nid=target, factors=(3, 5)))


def test_split_rejects_single_factor():
    """``len(factors)`` must be ``>= 2``."""
    ir = build_canonical_ir()
    target = _find_first_for_with_trip(ir, 16)
    with pytest.raises(TransformLegalityError):
        Split().apply(ir, SplitOption(target_nid=target, factors=(16,)))


def test_split_analyze_returns_only_legal_outer_trip_options():
    """Every option from ``analyze`` (outer-trip flavor) must apply without raising."""
    ir = build_canonical_ir()
    options = Split().analyze(ir)
    outer_trip_opts = [opt for opt in options if opt.target_axis is None]
    assert outer_trip_opts, "expected at least one outer-trip Split option"
    for opt in outer_trip_opts:
        Split().apply(ir, opt)


def test_split_analyze_finds_outer_factorization_of_trip_16():
    """``analyze`` should surface ``(2, 8)``, ``(4, 4)``, and ``(8, 2)`` for a trip-16 ForNode."""
    ir = build_canonical_ir()
    target = _find_first_for_with_trip(ir, 16)
    opts = [opt for opt in Split().analyze(ir) if opt.target_axis is None and opt.target_nid == target]
    factor_sets = {opt.factors for opt in opts}
    assert (2, 8) in factor_sets
    assert (4, 4) in factor_sets
    assert (8, 2) in factor_sets


def _find_lhs_t_load(ir) -> int:
    """Return the nid of the first ISANode whose op_cls.NAME=='dma_copy' that reads ``lhs_T``."""
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if isinstance(data, ISANode) and data.op_cls.NAME == "dma_copy" and "lhs_T" in data.reads:
            return nid
    raise AssertionError("lhs_T load not found")


def _find_matmul(ir) -> int:
    """Return the nid of the unique NKIMatmul ISANode."""
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if isinstance(data, ISANode) and data.op_cls.NAME == "nc_matmul":
            return nid
    raise AssertionError("matmul not found")


def test_split_tensorize_apply_inserts_outer_for():
    """Splitting lhs_T load F tensorize=2048 by (16, 128) inserts a trip=16 ForNode and updates tensorize."""
    ir = build_canonical_ir()
    leaf = _find_lhs_t_load(ir)
    leaf_data = ir.tree.data(leaf)
    parent_before = ir.tree.parent(leaf)
    assert leaf_data.tensorize_sizes["F"] == 2048

    new_ir = Split().apply(ir, SplitOption(target_nid=leaf, factors=(16, 128), target_axis="F"))

    """Original untouched."""
    assert ir.tree.data(leaf).tensorize_sizes["F"] == 2048

    """In new_ir: the leaf has tensorize_sizes[F]=128 and a fresh ForNode parent of trip=16 on the F dim."""
    new_leaf_data = new_ir.tree.data(leaf)
    assert new_leaf_data.tensorize_sizes["F"] == 128

    new_parent = new_ir.tree.parent(leaf)
    assert new_parent != parent_before
    new_parent_data = new_ir.tree.data(new_parent)
    assert isinstance(new_parent_data, ForNode)
    assert new_parent_data.trip == 16
    assert new_parent_data.dim == leaf_data.axis_map["F"]

    """The new ForNode's parent equals the original leaf parent."""
    assert new_ir.tree.parent(new_parent) == parent_before


def test_split_tensorize_three_way_chain():
    """Splitting tensorize=2048 by (4, 4, 128) inserts two new ForNodes."""
    ir = build_canonical_ir()
    leaf = _find_lhs_t_load(ir)
    new_ir = Split().apply(ir, SplitOption(target_nid=leaf, factors=(4, 4, 128), target_axis="F"))
    assert new_ir.tree.data(leaf).tensorize_sizes["F"] == 128

    """Walk up the parent chain from the leaf; expect two consecutive ForNodes on the F dim with trips 4 and 4."""
    parent = new_ir.tree.parent(leaf)
    inner = new_ir.tree.data(parent)
    assert isinstance(inner, ForNode) and inner.trip == 4 and inner.dim == new_ir.tree.data(leaf).axis_map["F"]
    grandparent = new_ir.tree.parent(parent)
    outer = new_ir.tree.data(grandparent)
    assert isinstance(outer, ForNode) and outer.trip == 4 and outer.dim == new_ir.tree.data(leaf).axis_map["F"]


def test_split_tensorize_rejects_non_isa_target():
    """Tensorize flavor must reject ForNode targets."""
    ir = build_canonical_ir()
    target = _find_first_for_with_trip(ir, 16)
    with pytest.raises(TransformLegalityError):
        Split().apply(ir, SplitOption(target_nid=target, factors=(4, 4), target_axis="F"))


def test_split_tensorize_rejects_axis_not_in_axis_map():
    """Tensorize flavor must reject an axis name absent from the leaf's axis_map."""
    ir = build_canonical_ir()
    leaf = _find_lhs_t_load(ir)
    with pytest.raises(TransformLegalityError):
        Split().apply(ir, SplitOption(target_nid=leaf, factors=(2, 1024), target_axis="ZZZ"))


def test_split_tensorize_rejects_below_min_tile():
    """factors[-1] below MIN_TILE_SIZE for the op's axis is illegal."""
    ir = build_canonical_ir()
    leaf = _find_matmul(ir)
    """NKIMatmul N axis: MIN=128, MAX=512, current tensorize=512.
       (8, 64) would set tensorize to 64, below MIN."""
    with pytest.raises(TransformLegalityError):
        Split().apply(ir, SplitOption(target_nid=leaf, factors=(8, 64), target_axis="N"))


def test_split_tensorize_rejects_above_max_tile():
    """MAX upper bound is structurally unreachable when factors[-1] < tensorize_before; documented skip."""
    pytest.skip("MAX upper bound is structurally unreachable when factors[-1] < tensorize_before")


def test_split_analyze_includes_lhs_t_F_split():
    """``analyze`` should surface the (16, 128) tensorize Split for the lhs_T load F axis."""
    ir = build_canonical_ir()
    leaf = _find_lhs_t_load(ir)
    opts = [opt for opt in Split().analyze(ir) if opt.target_axis == "F" and opt.target_nid == leaf]
    factor_sets = {opt.factors for opt in opts}
    assert (16, 128) in factor_sets


def test_split_analyze_skips_fixed_axes():
    """For an op axis with MIN==MAX==current (matmul K, M), no tensorize Split is legal."""
    ir = build_canonical_ir()
    leaf = _find_matmul(ir)
    opts = [opt for opt in Split().analyze(ir) if opt.target_axis in {"K", "M"} and opt.target_nid == leaf]
    assert opts == []


def test_split_analyze_returns_only_legal_tensorize_options():
    """Every tensorize option from ``analyze`` must apply without raising."""
    ir = build_canonical_ir()
    options = [opt for opt in Split().analyze(ir) if opt.target_axis is not None]
    assert options, "expected at least one tensorize Split option"
    for opt in options:
        Split().apply(ir, opt)


def test_split_outer_trip_preserves_sibling_order():
    """Splitting a ForNode that has siblings under root must keep the new chain at the same position."""
    ir = build_canonical_ir()
    """Find the lhs_T-load's outer K ForNode (a child of root with later siblings)."""
    target = None
    for nid in ir.tree.preorder():
        d = ir.tree.data(nid)
        if isinstance(d, ForNode) and d.dim == "d0" and d.trip == 16:
            target = nid
            break
    assert target is not None
    parent = ir.tree.parent(target)
    siblings_before = ir.tree.children(parent)
    target_pos = siblings_before.index(target)

    new_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(2, 8)))

    """In new_ir, parent's children list at position target_pos must be a ForNode with the new
    outer trip (2) on the same dim. Position must equal target_pos (sibling order preserved)."""
    new_siblings = new_ir.tree.children(parent)
    new_node_at_pos = new_ir.tree.data(new_siblings[target_pos])
    assert isinstance(new_node_at_pos, ForNode)
    assert new_node_at_pos.dim == "d0"
    assert new_node_at_pos.trip == 2


def test_split_outer_trip_renders_dataflow_order():
    """Render after outer-trip Split. The lhs_T dma_copy must appear before nc_matmul in the
    rendered NKI source — otherwise matmul would read sbuf_lhs_T before it's written."""
    from nkigym.codegen import render

    ir = build_canonical_ir()
    target = None
    for nid in ir.tree.preorder():
        d = ir.tree.data(nid)
        if isinstance(d, ForNode) and d.dim == "d0" and d.trip == 16:
            target = nid
            break
    assert target is not None
    new_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(2, 8)))
    src = render(new_ir)
    """Find offset of lhs_T-load and nc_matmul in source."""
    lhs_t_pos = src.find("src=lhs_T[")
    matmul_pos = src.find("nisa.nc_matmul")
    assert lhs_t_pos != -1 and matmul_pos != -1
    assert lhs_t_pos < matmul_pos, f"lhs_T load at {lhs_t_pos} must precede matmul at {matmul_pos}"


def test_split_analyze_skips_nki_alloc():
    """``analyze`` must not emit tensorize Split options targeting NKIAlloc leaves —
    splitting an alloc moves the nl.ndarray() inside a for loop, changing buffer placement."""
    from nkigym.ops.alloc import NKIAlloc

    ir = build_canonical_ir()
    options = Split().analyze(ir)
    for opt in options:
        if opt.target_axis is None:
            continue
        leaf = ir.tree.data(opt.target_nid)
        assert leaf.op_cls is not NKIAlloc, (
            f"Split.analyze emitted a tensorize option on NKIAlloc nid={opt.target_nid} "
            f"axis={opt.target_axis} factors={opt.factors}"
        )


def test_split_apply_rejects_nki_alloc_tensorize():
    """Applying a hand-constructed tensorize Split on NKIAlloc must raise."""
    from nkigym.ops.alloc import NKIAlloc

    ir = build_canonical_ir()
    """Find an NKIAlloc leaf with a splittable F tensorize size."""
    alloc_nid = None
    for nid in ir.tree.preorder():
        d = ir.tree.data(nid)
        if isinstance(d, ISANode) and d.op_cls is NKIAlloc and "F" in d.axis_map:
            alloc_nid = nid
            break
    assert alloc_nid is not None
    leaf_data = ir.tree.data(alloc_nid)
    if leaf_data.tensorize_sizes.get("F", 0) >= 256:
        with pytest.raises(TransformLegalityError):
            Split().apply(
                ir, SplitOption(target_nid=alloc_nid, factors=(2, leaf_data.tensorize_sizes["F"] // 2), target_axis="F")
            )
    else:
        pytest.skip("no NKIAlloc with splittable F tensorize in fixture")
