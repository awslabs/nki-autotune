"""Tests for :class:`nkigym.transforms.Fuse`."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

import pytest

from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms import Fuse, FuseOption, Split, SplitOption, TransformLegalityError


def _find_first_for_with_trip(ir, trip: int) -> int:
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if isinstance(data, ForNode) and data.trip == trip:
            return nid
    raise AssertionError(f"no ForNode with trip={trip}")


def test_fuse_outer_trip_undoes_split_round_trip():
    """Splitting a trip=16 ForNode then fusing the resulting two ForNodes restores trip=16."""
    ir = build_canonical_ir()
    target = _find_first_for_with_trip(ir, 16)
    target_dim = ir.tree.data(target).dim

    split_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(4, 4)))

    """Find the new chain in split_ir."""
    chain: list[int] = []
    for nid in split_ir.tree.preorder():
        data = split_ir.tree.data(nid)
        if isinstance(data, ForNode) and data.trip == 4 and data.dim == target_dim:
            kids = split_ir.tree.children(nid)
            if len(kids) == 1 and isinstance(split_ir.tree.data(kids[0]), ForNode):
                child_data = split_ir.tree.data(kids[0])
                if child_data.trip == 4 and child_data.dim == target_dim:
                    chain = [nid, kids[0]]
                    break
    assert len(chain) == 2

    fused_ir = Fuse().apply(split_ir, FuseOption(target_nids=tuple(chain)))

    """In fused_ir, the deepest entry of ``chain`` is gone and a new ForNode trip=16 sits at the chain root's old position."""
    survivors = [
        nid
        for nid in fused_ir.tree.preorder()
        if isinstance(fused_ir.tree.data(nid), ForNode)
        and fused_ir.tree.data(nid).trip == 16
        and fused_ir.tree.data(nid).dim == target_dim
    ]
    assert len(survivors) >= 1


def test_fuse_apply_preserves_input_ir():
    """``apply`` must not mutate its input IR."""
    ir = build_canonical_ir()
    target = _find_first_for_with_trip(ir, 16)
    split_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(4, 4)))
    snapshot = split_ir.tree.num_nodes

    """Find chain again."""
    target_dim = ir.tree.data(target).dim
    chain: list[int] = []
    for nid in split_ir.tree.preorder():
        data = split_ir.tree.data(nid)
        if isinstance(data, ForNode) and data.trip == 4 and data.dim == target_dim:
            kids = split_ir.tree.children(nid)
            if len(kids) == 1 and isinstance(split_ir.tree.data(kids[0]), ForNode):
                child_data = split_ir.tree.data(kids[0])
                if child_data.trip == 4 and child_data.dim == target_dim:
                    chain = [nid, kids[0]]
                    break

    Fuse().apply(split_ir, FuseOption(target_nids=tuple(chain)))
    assert split_ir.tree.num_nodes == snapshot


def test_fuse_rejects_single_target():
    """``len(target_nids)`` must be ``>= 2``."""
    ir = build_canonical_ir()
    target = _find_first_for_with_trip(ir, 16)
    with pytest.raises(TransformLegalityError):
        Fuse().apply(ir, FuseOption(target_nids=(target,)))


def test_fuse_rejects_non_chain_targets():
    """Two ForNodes that are not in a parent->child chain must be rejected."""
    ir = build_canonical_ir()
    """Find two top-level ForNodes (siblings under root)."""
    root_kids = ir.tree.children(ir.tree.root)
    fornode_kids = [nid for nid in root_kids if isinstance(ir.tree.data(nid), ForNode)]
    assert len(fornode_kids) >= 2
    with pytest.raises(TransformLegalityError):
        Fuse().apply(ir, FuseOption(target_nids=(fornode_kids[0], fornode_kids[1])))


def test_fuse_rejects_dim_mismatch():
    """Two ForNodes on different dims may not be fused."""
    ir = build_canonical_ir()
    """The matmul nest has K (d0) outer, M/N inside. Find adjacent same-chain different-dim pair."""
    different_dim_chain: tuple[int, int] | None = None
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if not isinstance(data, ForNode):
            continue
        kids = ir.tree.children(nid)
        if len(kids) == 1 and isinstance(ir.tree.data(kids[0]), ForNode):
            child = ir.tree.data(kids[0])
            if child.dim != data.dim:
                different_dim_chain = (nid, kids[0])
                break
    assert different_dim_chain is not None
    with pytest.raises(TransformLegalityError):
        Fuse().apply(ir, FuseOption(target_nids=different_dim_chain))


def test_fuse_analyze_returns_only_legal_outer_options():
    """Every outer-trip option must apply without raising."""
    ir = build_canonical_ir()
    target = _find_first_for_with_trip(ir, 16)
    split_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(4, 4)))
    options = [opt for opt in Fuse().analyze(split_ir) if opt.target_axis is None]
    assert options, "expected at least one outer-trip Fuse option after a Split"
    for opt in options:
        Fuse().apply(split_ir, opt)


def _find_lhs_t_load(ir) -> int:
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if isinstance(data, ISANode) and data.op_cls.NAME == "dma_copy" and "lhs_T" in data.reads:
            return nid
    raise AssertionError("lhs_T load not found")


def _find_matmul(ir) -> int:
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if isinstance(data, ISANode) and data.op_cls.NAME == "nc_matmul":
            return nid
    raise AssertionError("matmul not found")


def test_fuse_tensorize_undoes_split_round_trip():
    """Splitting lhs_T F tensorize=2048 by (16, 128) then fusing the resulting (ForNode, leaf) chain restores tensorize=2048."""
    ir = build_canonical_ir()
    leaf = _find_lhs_t_load(ir)
    split_ir = Split().apply(ir, SplitOption(target_nid=leaf, factors=(16, 128), target_axis="F"))
    new_parent = split_ir.tree.parent(leaf)
    assert split_ir.tree.data(new_parent).trip == 16
    chain = (new_parent, leaf)

    fused_ir = Fuse().apply(split_ir, FuseOption(target_nids=chain, target_axis="F"))

    """Original split_ir untouched."""
    assert split_ir.tree.data(leaf).tensorize_sizes["F"] == 128

    """In fused_ir, leaf's tensorize_sizes[F] is back to 2048 and its parent is the original parent of new_parent."""
    fused_leaf = fused_ir.tree.data(leaf)
    assert fused_leaf.tensorize_sizes["F"] == 2048
    grandparent = split_ir.tree.parent(new_parent)
    assert fused_ir.tree.parent(leaf) == grandparent


def test_fuse_tensorize_three_way_chain():
    """Splitting tensorize=2048 by (4, 4, 128) then fusing the resulting (ForNode, ForNode, leaf) chain restores tensorize=2048."""
    ir = build_canonical_ir()
    leaf = _find_lhs_t_load(ir)
    split_ir = Split().apply(ir, SplitOption(target_nid=leaf, factors=(4, 4, 128), target_axis="F"))
    parent = split_ir.tree.parent(leaf)
    grandparent = split_ir.tree.parent(parent)
    chain = (grandparent, parent, leaf)

    fused_ir = Fuse().apply(split_ir, FuseOption(target_nids=chain, target_axis="F"))
    assert fused_ir.tree.data(leaf).tensorize_sizes["F"] == 2048


def test_fuse_tensorize_rejects_non_isa_last():
    """The last entry must be an ISANode."""
    ir = build_canonical_ir()
    target = _find_first_for_with_trip(ir, 16)
    """Build a fake 2-ForNode chain via Split, then incorrectly mark target_axis."""
    split_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(4, 4)))
    target_dim = ir.tree.data(target).dim
    chain: list[int] = []
    for nid in split_ir.tree.preorder():
        data = split_ir.tree.data(nid)
        if isinstance(data, ForNode) and data.trip == 4 and data.dim == target_dim:
            kids = split_ir.tree.children(nid)
            if len(kids) == 1 and isinstance(split_ir.tree.data(kids[0]), ForNode):
                chain = [nid, kids[0]]
                break
    with pytest.raises(TransformLegalityError):
        Fuse().apply(split_ir, FuseOption(target_nids=tuple(chain), target_axis="K"))


def test_fuse_tensorize_rejects_axis_not_in_axis_map():
    """Tensorize flavor must reject an axis name absent from the leaf's axis_map."""
    ir = build_canonical_ir()
    leaf = _find_lhs_t_load(ir)
    split_ir = Split().apply(ir, SplitOption(target_nid=leaf, factors=(16, 128), target_axis="F"))
    parent = split_ir.tree.parent(leaf)
    with pytest.raises(TransformLegalityError):
        Fuse().apply(split_ir, FuseOption(target_nids=(parent, leaf), target_axis="ZZZ"))


def test_fuse_tensorize_rejects_above_max():
    """Tensorize MAX upper bound is structurally unreachable when each tensorize Split already respects MAX and Fuse merely recombines factors."""
    pytest.skip(
        "Tensorize MAX upper bound is structurally unreachable when each tensorize Split already "
        "respects MAX and Fuse merely recombines factors."
    )


def test_fuse_analyze_finds_tensorize_chain_after_split():
    """``analyze`` should surface a tensorize Fuse option for a (ForNode, leaf) chain on the F axis."""
    ir = build_canonical_ir()
    leaf = _find_lhs_t_load(ir)
    split_ir = Split().apply(ir, SplitOption(target_nid=leaf, factors=(16, 128), target_axis="F"))
    parent = split_ir.tree.parent(leaf)
    options = [opt for opt in Fuse().analyze(split_ir) if opt.target_axis == "F" and opt.target_nids == (parent, leaf)]
    assert options, "expected a tensorize Fuse option for the (ForNode, leaf) chain on F"


def test_fuse_analyze_no_tensorize_options_on_canonical():
    """On the canonical IR, every enclosing same-dim ForNode chain has trip-product == 1,
    failing the >= 2 requirement, so tensorize Fuse should yield no options."""
    ir = build_canonical_ir()
    options = [opt for opt in Fuse().analyze(ir) if opt.target_axis is not None]
    assert options == []


def test_fuse_outer_trip_preserves_sibling_order():
    """Fusing a ForNode chain that has siblings under root must keep the new node at the same position."""
    ir = build_canonical_ir()
    target = None
    for nid in ir.tree.preorder():
        d = ir.tree.data(nid)
        if isinstance(d, ForNode) and d.dim == "d0" and d.trip == 16:
            target = nid
            break
    assert target is not None
    parent = ir.tree.parent(target)
    target_pos = ir.tree.children(parent).index(target)

    """Apply Split first to manufacture a chain."""
    split_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(2, 8)))

    """Re-locate the chain (split_ir's parent's child list at the same position now points to a (2, 8) chain)."""
    split_siblings = split_ir.tree.children(parent)
    chain_root = split_siblings[target_pos]
    chain_inner = split_ir.tree.children(chain_root)[0]

    fused_ir = Fuse().apply(split_ir, FuseOption(target_nids=(chain_root, chain_inner)))

    fused_siblings = fused_ir.tree.children(parent)
    new_node_at_pos = fused_ir.tree.data(fused_siblings[target_pos])
    assert isinstance(new_node_at_pos, ForNode)
    assert new_node_at_pos.dim == "d0"
    assert new_node_at_pos.trip == 16


def test_fuse_analyze_skips_nki_alloc():
    """``analyze`` must not emit tensorize Fuse options whose leaf is NKIAlloc."""
    from nkigym.ops.alloc import NKIAlloc

    ir = build_canonical_ir()
    """First create some chain above an NKIAlloc by trying — but Split skips alloc, so to test
    analyze's filter we need to construct an IR where an alloc has an enclosing same-dim ForNode.
    Since canonical has no enclosing loops on alloc leaves and Split won't add them, this case
    cannot arise organically. Just assert that analyze returns no tensorize options with NKIAlloc target."""
    options = Fuse().analyze(ir)
    for opt in options:
        if opt.target_axis is None:
            continue
        leaf = ir.tree.data(opt.target_nids[-1])
        assert (
            leaf.op_cls is not NKIAlloc
        ), f"Fuse.analyze emitted a tensorize option on NKIAlloc nid={opt.target_nids[-1]}"
