"""Tests for nkigym.transforms.RFactor."""

from __future__ import annotations

from dataclasses import replace
from test._simulation import assert_matmul_ir_simulates
from test.transforms._rfactor_fixtures import (
    k28_ir,
    k28_ko_loop_nid,
    ko_loop_nid,
    matmul_leaf_nid,
    mid_ladder_ir,
    split_k_ir,
)

import pytest

from nkigym.ir import KernelIR
from nkigym.ir.tree import BlockNode, ForNode, ISANode
from nkigym.ops.base import AxisRole
from nkigym.transforms import (
    BufferCompaction,
    BufferCompactionOption,
    BufferLayout,
    BufferLayoutOption,
    CodeMotion,
    CodeMotionOption,
    RFactor,
    RFactorOption,
    TransformLegalityError,
)


def _rfactored_ir():
    """Canonical matmul → Split(K) → RFactor(ko). The post-RFactor IR under test."""
    ir = split_k_ir()
    return RFactor().apply(ir, RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0))


def _compact_sbuf_prod_under_store(ir: KernelIR) -> KernelIR:
    """Move the drain under the store M loop, then compact their shared output."""
    tensor_copy = next(
        nid
        for nid in ir.tree.preorder()
        if isinstance((node := ir.tree.data(nid)), ISANode) and node.op_cls.__name__ == "NKITensorCopy"
    )
    store = next(
        nid
        for nid in ir.tree.preorder()
        if isinstance((node := ir.tree.data(nid)), ISANode) and node.op_cls.__name__ == "NKIStore"
    )
    tensor_copy_block = next(
        ancestor
        for ancestor in reversed(ir.tree.ancestors(tensor_copy))
        if isinstance(ir.tree.data(ancestor), BlockNode)
    )
    store_block = next(
        ancestor for ancestor in reversed(ir.tree.ancestors(store)) if isinstance(ir.tree.data(ancestor), BlockNode)
    )
    store_m_loop = next(
        descendant
        for descendant in ir.tree.preorder(store_block)
        if isinstance((node := ir.tree.data(descendant)), ForNode) and node.loop_var.startswith("i_d1_")
    )
    moved = CodeMotion().apply(ir, CodeMotionOption(block_nid=tensor_copy_block, target_loop_nid=store_m_loop, index=0))
    return BufferCompaction().apply(moved, BufferCompactionOption(tensor="sbuf_prod"))


def _replace_sbuf_prod(ir: KernelIR, shape: tuple[int, int], versions: int, list_len: int) -> None:
    """Replace the output buffer geometry in place for a legality fixture."""
    for block_nid in ir.tree.blocks():
        block = ir.tree.data(block_nid)
        if not isinstance(block, BlockNode) or not any(buf.name == "sbuf_prod" for buf in block.alloc_buffers):
            continue
        buffers = tuple(
            replace(buf, shape=shape, versions=versions, list_len=list_len) if buf.name == "sbuf_prod" else buf
            for buf in block.alloc_buffers
        )
        ir.tree.graph.nodes[block_nid]["data"] = replace(block, alloc_buffers=buffers)
        break


def test_analyze_finds_only_reduction_loops() -> None:
    """analyze offers only ForNodes binding the matmul's ACCUMULATION (K) axis."""
    ir = split_k_ir()
    opts = RFactor().analyze(ir)
    assert len(opts) >= 1
    for o in opts:
        node = ir.tree.data(o.target_loop_nid)
        assert isinstance(node, ForNode)
        assert node.loop_var.startswith("i_d0_")


def test_rfactor_rejects_inner_reduction_loop() -> None:
    """Only the outermost reduction loop can own the generated second-stage fold."""
    ir = split_k_ir()
    matmul = matmul_leaf_nid(ir)
    inner = next(
        nid
        for nid in ir.tree.ancestors(matmul)
        if isinstance((node := ir.tree.data(nid)), ForNode) and node.loop_var == "i_d0_1"
    )
    option = RFactorOption(target_loop_nid=inner, factor_axis=0)

    with pytest.raises(TransformLegalityError, match="outermost"):
        RFactor().apply(ir, option)
    assert option not in RFactor().analyze(ir)


def test_apply_rejects_non_forNode() -> None:
    """A target that is not a ForNode (the matmul leaf) is rejected loudly."""
    ir = split_k_ir()
    mm = matmul_leaf_nid(ir)
    with pytest.raises(TransformLegalityError):
        RFactor().apply(ir, RFactorOption(target_loop_nid=mm, factor_axis=0))


def test_apply_rejects_parallel_loop() -> None:
    """A PARALLEL loop (the M loop, i_d1_*) is not a reduction loop → rejected."""
    ir = split_k_ir()
    m_loop = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ForNode) and ir.tree.loop(n).loop_var.startswith("i_d1_")
    )
    with pytest.raises(TransformLegalityError):
        RFactor().apply(ir, RFactorOption(target_loop_nid=m_loop, factor_axis=0))


def test_apply_rejects_drain_block_that_contains_the_output_store() -> None:
    """RFactor cannot delete a drain block that owns another ISA operation."""
    ir = split_k_ir()
    tensor_copy = next(
        nid
        for nid in ir.tree.preorder()
        if isinstance(ir.tree.data(nid), ISANode) and ir.tree.isa(nid).op_cls.__name__ == "NKITensorCopy"
    )
    store = next(
        nid
        for nid in ir.tree.preorder()
        if isinstance(ir.tree.data(nid), ISANode) and ir.tree.isa(nid).op_cls.__name__ == "NKIStore"
    )
    drain_loop = next(
        ancestor for ancestor in reversed(ir.tree.ancestors(tensor_copy)) if isinstance(ir.tree.data(ancestor), ForNode)
    )
    store_block = next(
        ancestor for ancestor in reversed(ir.tree.ancestors(store)) if isinstance(ir.tree.data(ancestor), BlockNode)
    )
    ir = CodeMotion().apply(ir, CodeMotionOption(block_nid=store_block, target_loop_nid=drain_loop, index=1))
    option = RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0)
    with pytest.raises(TransformLegalityError):
        RFactor().apply(ir, option)
    assert option not in RFactor().analyze(ir)


def test_apply_rejects_partition_footprint_larger_than_compact_output() -> None:
    """RFactor cannot materialize 16 partition tiles in a one-tile output buffer."""
    ir = _compact_sbuf_prod_under_store(split_k_ir())
    rfactor = RFactor()
    option = RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0)
    matmul_leaf = matmul_leaf_nid(ir)
    ki_nid = rfactor._ki_loop_nid(ir, option.target_loop_nid)

    assert ir.buffer("sbuf_prod").shape == (128, 2048)
    assert ir.buffer("sbuf_prod").physical_shape() == (128, 1, 2048)
    assert rfactor._drain_block_is_removable(ir, matmul_leaf)
    assert rfactor._footprint(ir, ki_nid, matmul_leaf) == [("i_d1_0", 16)]
    assert not rfactor._gadget_region_fits_output(ir, option.target_loop_nid, matmul_leaf)
    with pytest.raises(TransformLegalityError, match="footprint|capacity"):
        rfactor.apply(ir, option)
    assert option not in rfactor.analyze(ir)


def test_apply_rejects_product_of_multiple_footprint_extents() -> None:
    """A 4 x 4 footprint exceeds eight logical tiles despite pipeline versions."""
    ir = _compact_sbuf_prod_under_store(mid_ladder_ir())
    _replace_sbuf_prod(ir, shape=(1024, 2048), versions=2, list_len=1)

    rfactor = RFactor()
    option = RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0)
    matmul_leaf = matmul_leaf_nid(ir)
    ki_nid = rfactor._ki_loop_nid(ir, option.target_loop_nid)

    assert ir.buffer("sbuf_prod").shape[0] // 128 == 8
    assert ir.buffer("sbuf_prod").physical_shape()[1] == 16
    assert rfactor._footprint(ir, ki_nid, matmul_leaf) == [("i_d1_0", 4), ("i_d1_1", 4)]
    with pytest.raises(TransformLegalityError, match="footprint|capacity"):
        rfactor.apply(ir, option)
    assert option not in rfactor.analyze(ir)


def test_apply_rejects_inherited_partition_offsets_larger_than_compact_output() -> None:
    """Outer M loops must fit even when the inner materialized footprint is empty."""
    ir = k28_ir()
    _replace_sbuf_prod(ir, shape=(128, 512), versions=2, list_len=1)
    rfactor = RFactor()
    option = RFactorOption(target_loop_nid=k28_ko_loop_nid(ir), factor_axis=0)
    matmul_leaf = matmul_leaf_nid(ir)
    ki_nid = rfactor._ki_loop_nid(ir, option.target_loop_nid)

    assert rfactor._footprint(ir, ki_nid, matmul_leaf) == []
    with pytest.raises(TransformLegalityError, match="region|capacity"):
        rfactor.apply(ir, option)
    assert option not in rfactor.analyze(ir)


def test_apply_rejects_absorbed_free_span_larger_than_compact_output() -> None:
    """The full absorbed free-axis gadget width must fit the output buffer."""
    ir = split_k_ir()
    _replace_sbuf_prod(ir, shape=(2048, 128), versions=1, list_len=1)
    rfactor = RFactor()
    option = RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0)

    with pytest.raises(TransformLegalityError, match="region|capacity"):
        rfactor.apply(ir, option)
    assert option not in rfactor.analyze(ir)


def test_apply_accepts_listed_output_with_sufficient_total_capacity() -> None:
    """A list-of-16 output retains all 16 logical partition tiles."""
    ir = split_k_ir()
    ir = BufferLayout().apply(ir, BufferLayoutOption(tensor="sbuf_prod", list_len=16))
    rfactor = RFactor()
    option = RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0)

    assert ir.buffer("sbuf_prod").per_tile_physical_shape() == (128, 1, 2048)
    assert option in rfactor.analyze(ir)
    rfactor.apply(ir, option)


def test_apply_sim_matches_matmul(tmp_path) -> None:
    """The rfactored kernel sims numerically equal to lhs_T.T @ rhs."""
    assert_matmul_ir_simulates(_rfactored_ir(), tmp_path, "rfactor_early_packed")


def test_apply_sim_matches_matmul_mid_tiled_m(tmp_path) -> None:
    """RFactor(ko) on the mid state (K split ko/ki AND M tiled i_d1_0 x i_d1_1,
    buffers still packed) sims equal to the golden — the tiled-M geometry, pinned
    as a regression test via the ``mid_ladder_ir`` fixture."""
    ir = mid_ladder_ir()
    rfactored = RFactor().apply(ir, RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0))
    assert_matmul_ir_simulates(rfactored, tmp_path, "rfactor_mid_tiled_m")


def test_ko_roles_split_across_blocks() -> None:
    """After RFactor, the K axis (d0) appears as PARALLEL in the matmul run-op block
    (each ko is an independent partial) and ACCUMULATION in the tensor_tensor fold block
    (carrying ko across the closing second-stage reduction) — one factored axis, two
    block roles."""
    ir = _rfactored_ir()
    roles = set()
    for nid in ir.tree.blocks():
        block = ir.tree.block(nid)
        for iv in block.iter_vars:
            if iv.axis == "d0":
                roles.add(iv.role)
    assert AxisRole.PARALLEL in roles
    assert AxisRole.ACCUMULATION in roles


def test_rf_memset_drain_nested_in_ko() -> None:
    """Spec §3.1: the rf-init memset and rf-drain tensor_copy are nested INSIDE the
    matmul's ko loop (per-slot), NOT flat sibling blocks outside it.

    Regression guard for the flat-vs-nested deviation: the rf-memset (writes psum,
    BEFORE the ki nest) and the rf-drain (psum -> psum_rf, AFTER the ki nest) must
    each have the matmul's ko ForNode among their loop ancestors.
    """
    ir = _rfactored_ir()
    matmul_leaf = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.isa(n).op_cls.__name__ == "NKIMatmul"
    )
    ko = next(
        a
        for a in ir.tree.ancestors(matmul_leaf)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.loop(a).loop_var.startswith("i_d0_")
    )
    psum_name = ir.tree.isa(matmul_leaf).operand_bindings["dst"].tensor
    rf_memset = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode)
        and ir.tree.isa(n).op_cls.NAME == "memset"
        and ir.tree.isa(n).operand_bindings["dst"].tensor == psum_name
    )
    rf_drain = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode)
        and ir.tree.isa(n).op_cls.NAME == "tensor_copy"
        and ir.tree.isa(n).operand_bindings["src"].tensor == psum_name
    )
    assert ko in ir.tree.ancestors(rf_memset), "rf-init memset must be nested inside the ko loop"
    assert ko in ir.tree.ancestors(rf_drain), "rf-drain tensor_copy must be nested inside the ko loop"


def test_apply_sim_matches_matmul_k28_to_k29(tmp_path) -> None:
    """The k28->k29 rfactored kernel sims numerically equal to lhs_T.T @ rhs."""
    ir = k28_ir()
    rfactored = RFactor().apply(ir, RFactorOption(target_loop_nid=k28_ko_loop_nid(ir), factor_axis=0))
    assert_matmul_ir_simulates(rfactored, tmp_path, "rfactor_k28_to_k29")
