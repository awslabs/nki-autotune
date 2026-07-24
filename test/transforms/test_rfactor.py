"""Tests for nkigym.transforms.RFactor."""

from __future__ import annotations

from dataclasses import replace
from test._simulation import assert_matmul_ir_simulates
from test.transforms import _matmul_lhsT_rhs_manual as manual_ladder
from test.transforms._fixtures import build_canonical_ir, f_matmul
from test.transforms._helpers import block_for_op
from test.transforms._ladder_compare import assert_matches_hand
from test.transforms._rfactor_fixtures import (
    k32_ir,
    k32_ko_loop_nid,
    ko_loop_nid,
    matmul_leaf_nid,
    mid_ladder_ir,
    split_k_ir,
)

import pytest

from nkigym.codegen import render
from nkigym.ir import KernelIR, build_initial_ir
from nkigym.ir.arith.expr import Const, FloorDiv, Mul, Sub, Var
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode
from nkigym.ops.base import AxisRole, ReduceCombinator
from nkigym.ops.matmul import NKIMatmul
from nkigym.transforms import (
    BufferCompaction,
    BufferCompactionOption,
    BufferLayout,
    BufferLayoutOption,
    CodeMotion,
    CodeMotionOption,
    RFactor,
    RFactorOption,
    Split,
    SplitOption,
    TransformLegalityError,
)
from nkigym.transforms.code_motion import _move


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


def test_rfactor_rejects_unsplit_reduction_loop() -> None:
    """RFactor requires a prior Split that produces distinct ko and ki loops."""
    ir = build_canonical_ir()
    matmul = matmul_leaf_nid(ir)
    unsplit_k = next(
        nid
        for nid in ir.tree.ancestors(matmul)
        if isinstance((node := ir.tree.data(nid)), ForNode) and node.loop_var == "i_d0_0"
    )
    option = RFactorOption(target_loop_nid=unsplit_k, factor_axis=0)

    assert option not in RFactor().analyze(ir)
    with pytest.raises(TransformLegalityError, match="exactly two loops"):
        RFactor().apply(ir, option)


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


def test_rfactor_rejects_more_than_two_reduction_loops() -> None:
    """The emitted fold metadata represents one ko/ki split, not a deeper K nest."""
    ir = split_k_ir()
    matmul = matmul_leaf_nid(ir)
    inner = next(
        nid
        for nid in ir.tree.ancestors(matmul)
        if isinstance((node := ir.tree.data(nid)), ForNode) and node.loop_var == "i_d0_1"
    )
    ir = Split().apply(ir, SplitOption(target_nid=inner, factors=(2, 4), target_axis=None))
    option = RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0)

    assert option not in RFactor().analyze(ir)
    with pytest.raises(TransformLegalityError, match="exactly two loops"):
        RFactor().apply(ir, option)


def test_apply_rejects_nonzero_factor_axis() -> None:
    """The fused recipe has no inserted factor axis, so only axis 0 is supported."""
    ir = split_k_ir()
    option = RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=1)

    with pytest.raises(TransformLegalityError, match="factor_axis must be 0"):
        RFactor().apply(ir, option)


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


def test_apply_rejects_drain_sunk_inside_factored_loop() -> None:
    """The one-stage drain must close after ko, not consume each partial inside it."""
    ir = split_k_ir()
    ko = ko_loop_nid(ir)
    _move(ir, block_nid=block_for_op(ir, "NKITensorCopy"), target_loop_nid=ko, index=1)
    option = RFactorOption(target_loop_nid=ko, factor_axis=0)

    assert option not in RFactor().analyze(ir)
    with pytest.raises(TransformLegalityError, match="outside-loop"):
        RFactor().apply(ir, option)


def test_apply_rejects_missing_one_stage_init() -> None:
    """Analyze must not offer an option that apply cannot retarget."""
    ir = split_k_ir()
    init = next(
        nid
        for nid in ir.tree.preorder()
        if isinstance((node := ir.tree.data(nid)), ISANode)
        and node.op_cls.NAME == "memset"
        and node.operand_bindings["dst"].tensor == "psum_prod"
    )
    rfactor = RFactor()
    rfactor._remove_flat_block(ir.tree, init)
    option = RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0)

    assert option not in rfactor.analyze(ir)
    with pytest.raises(TransformLegalityError, match="canonical outside-loop init"):
        rfactor.apply(ir, option)


def test_apply_rejects_unsupported_rmw_combiner(monkeypatch: pytest.MonkeyPatch) -> None:
    """The generated tensor_tensor fold must have a known associative operator."""
    ir = split_k_ir()
    monkeypatch.setattr(NKIMatmul, "REDUCE_COMBINATOR", ReduceCombinator(combiner="max", identity=0.0))
    option = RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0)

    assert option not in RFactor().analyze(ir)
    with pytest.raises(TransformLegalityError, match="supported combiner"):
        RFactor().apply(ir, option)


def test_apply_rejects_noncontiguous_free_footprint() -> None:
    """Absorbing a reversed free loop into one wide operation is unsupported."""
    ir = split_k_ir()
    matmul_nid = matmul_leaf_nid(ir)
    matmul = ir.tree.isa(matmul_nid)
    dst = matmul.operand_bindings["dst"]
    reversed_lo = Sub(left=Const(value=1536), right=Mul(left=Var(name="i_d2_0"), right=Const(value=512)))
    reversed_dst = BufferRegion(tensor=dst.tensor, ranges=(dst.ranges[0], (reversed_lo, dst.ranges[1][1])))
    ir.tree.graph.nodes[matmul_nid]["data"] = replace(
        matmul, operand_bindings={**matmul.operand_bindings, "dst": reversed_dst}
    )
    option = RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0)

    assert option not in RFactor().analyze(ir)
    with pytest.raises(TransformLegalityError, match="contiguous gadget footprint"):
        RFactor().apply(ir, option)


def test_analyze_rejects_non_affine_free_footprint() -> None:
    """Unsupported output indexing must withhold the option instead of raising."""
    ir = k32_ir()
    matmul_nid = matmul_leaf_nid(ir)
    matmul = ir.tree.isa(matmul_nid)
    dst = matmul.operand_bindings["dst"]
    non_affine_lo = FloorDiv(left=Var(name="i_d2_0"), right=Const(value=2))
    non_affine_dst = BufferRegion(tensor=dst.tensor, ranges=(dst.ranges[0], (non_affine_lo, dst.ranges[1][1])))
    ir.tree.graph.nodes[matmul_nid]["data"] = replace(
        matmul, operand_bindings={**matmul.operand_bindings, "dst": non_affine_dst}
    )
    option = RFactorOption(target_loop_nid=k32_ko_loop_nid(ir), factor_axis=0)

    assert option not in RFactor().analyze(ir)
    with pytest.raises(TransformLegalityError, match="contiguous gadget footprint"):
        RFactor().apply(ir, option)


def test_apply_rejects_non_identity_drain_mapping() -> None:
    """RFactor must not discard a drain's distinct source-to-output indexing."""
    ir = split_k_ir()
    drain_nid = next(
        nid
        for nid in ir.tree.preorder()
        if isinstance((node := ir.tree.data(nid)), ISANode) and node.op_cls.NAME == "tensor_copy"
    )
    drain = ir.tree.isa(drain_nid)
    dst = drain.operand_bindings["dst"]
    mismatched_dst = BufferRegion(tensor=dst.tensor, ranges=((Const(value=0), dst.ranges[0][1]), dst.ranges[1]))
    ir.tree.graph.nodes[drain_nid]["data"] = replace(
        drain, operand_bindings={**drain.operand_bindings, "dst": mismatched_dst}
    )
    option = RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0)

    assert option not in RFactor().analyze(ir)
    with pytest.raises(TransformLegalityError, match="identity-mapped"):
        RFactor().apply(ir, option)


def test_apply_avoids_staging_buffer_name_collision() -> None:
    """A pre-existing ``sbuf_rfactor`` allocation gets a deterministic suffix."""
    ir = split_k_ir()
    for block_nid in ir.tree.blocks():
        block = ir.tree.block(block_nid)
        if not any(buf.name == "psum_prod" for buf in block.alloc_buffers):
            continue
        existing = replace(ir.buffer("sbuf_prod"), name="sbuf_rfactor")
        ir.tree.graph.nodes[block_nid]["data"] = replace(block, alloc_buffers=(*block.alloc_buffers, existing))
        break

    rfactored = RFactor().apply(ir, RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0))
    assert rfactored.buffer("sbuf_rfactor").shape == ir.buffer("sbuf_prod").shape
    assert rfactored.buffer("sbuf_rfactor_1").shape == ir.buffer("sbuf_prod").shape
    generated_copy = next(
        rfactored.tree.isa(nid)
        for nid in rfactored.tree.preorder()
        if isinstance((node := rfactored.tree.data(nid)), ISANode)
        and node.op_cls.NAME == "tensor_copy"
        and node.operand_bindings["src"].tensor == "psum_prod"
    )
    assert generated_copy.operand_bindings["dst"].tensor == "sbuf_rfactor_1"


def test_apply_rejects_partition_footprint_larger_than_compact_output() -> None:
    """RFactor cannot materialize 16 partition tiles in a one-tile output buffer."""
    ir = _compact_sbuf_prod_under_store(split_k_ir())
    rfactor = RFactor()
    option = RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0)
    matmul_leaf = matmul_leaf_nid(ir)
    ki_nid = rfactor._ki_loop_nid(ir, option.target_loop_nid)

    assert ir.buffer("sbuf_prod").shape == (128, 2048)
    assert ir.buffer("sbuf_prod").physical_shape() == (128, 1, 2048)
    assert not rfactor._drain_block_is_removable(ir, option.target_loop_nid, matmul_leaf)
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
    ir = k32_ir()
    _replace_sbuf_prod(ir, shape=(128, 512), versions=2, list_len=1)
    rfactor = RFactor()
    option = RFactorOption(target_loop_nid=k32_ko_loop_nid(ir), factor_axis=0)
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


def test_generated_gadgets_inherit_non_square_matmul_domains() -> None:
    """Generated block metadata derives K and M independently."""
    input_specs = {"lhs_T": ((512, 256), "bfloat16"), "rhs": ((512, 512), "bfloat16")}
    ir = build_initial_ir(f_matmul, input_specs)
    matmul = matmul_leaf_nid(ir)
    k_loop = next(
        nid
        for nid in ir.tree.ancestors(matmul)
        if isinstance((node := ir.tree.data(nid)), ForNode) and node.loop_var == "i_d0_0"
    )
    ir = Split().apply(ir, SplitOption(target_nid=k_loop, factors=(2, 2), target_axis=None))
    ir = RFactor().apply(ir, RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0))

    generated = [ir.tree.block(nid) for nid in ir.tree.blocks() if set(ir.tree.block(nid).axis_map) == {"K", "P", "F"}]
    assert len(generated) == 3
    for block in generated:
        domains = {iter_var.axis: iter_var.dom for iter_var in block.iter_vars}
        assert domains == {"d0": (0, 512), "d1": (0, 256), "d2": (0, 512)}


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


def test_apply_byte_exact_k32_to_k33() -> None:
    """Structural RFactor alone reproduces the hand-written k33 rung."""
    ir = k32_ir()
    unchanged = ir.all_buffers()
    rfactored = RFactor().apply(ir, RFactorOption(target_loop_nid=k32_ko_loop_nid(ir), factor_axis=0))

    assert all(rfactored.buffer(name) == buf for name, buf in unchanged.items())
    assert (rfactored.buffer("psum_prod").shape, rfactored.buffer("psum_prod").list_len) == ((2048, 512), 16)
    assert (rfactored.buffer("sbuf_rfactor").shape, rfactored.buffer("sbuf_rfactor").list_len) == ((2048, 512), 1)
    compactable = {option.tensor for option in BufferCompaction().analyze(rfactored)}
    assert {"psum_prod", "sbuf_rfactor"} <= compactable
    assert_matches_hand(render(rfactored), manual_ladder.kernel_33)


def test_apply_byte_exact_k33_to_k34() -> None:
    """The first explicit compaction tightens only the PSUM partial."""
    ir = k32_ir()
    rfactored = RFactor().apply(ir, RFactorOption(target_loop_nid=k32_ko_loop_nid(ir), factor_axis=0))
    compacted = BufferCompaction().apply(rfactored, BufferCompactionOption(tensor="psum_prod"))

    assert (compacted.buffer("psum_prod").shape, compacted.buffer("psum_prod").list_len) == ((128, 512), 1)
    assert (compacted.buffer("sbuf_rfactor").shape, compacted.buffer("sbuf_rfactor").list_len) == ((2048, 512), 1)
    compactable = {option.tensor for option in BufferCompaction().analyze(compacted)}
    assert "psum_prod" not in compactable
    assert "sbuf_rfactor" in compactable
    assert_matches_hand(render(compacted), manual_ladder.kernel_34)


def test_apply_byte_exact_k34_to_k35() -> None:
    """The second explicit compaction tightens only the staging buffer."""
    ir = k32_ir()
    rfactored = RFactor().apply(ir, RFactorOption(target_loop_nid=k32_ko_loop_nid(ir), factor_axis=0))
    compacted = BufferCompaction().apply(rfactored, BufferCompactionOption(tensor="psum_prod"))
    final = BufferCompaction().apply(compacted, BufferCompactionOption(tensor="sbuf_rfactor"))

    assert (final.buffer("psum_prod").shape, final.buffer("psum_prod").list_len) == ((128, 512), 1)
    assert (final.buffer("sbuf_rfactor").shape, final.buffer("sbuf_rfactor").list_len) == ((128, 512), 1)
    compactable = {option.tensor for option in BufferCompaction().analyze(final)}
    assert "psum_prod" not in compactable
    assert "sbuf_rfactor" not in compactable
    assert_matches_hand(render(final), manual_ladder.kernel_35)


def test_apply_sim_matches_matmul_k32_to_k35(tmp_path) -> None:
    """Structural RFactor and both compactions preserve the matmul result."""
    ir = k32_ir()
    rfactored = RFactor().apply(ir, RFactorOption(target_loop_nid=k32_ko_loop_nid(ir), factor_axis=0))
    assert_matmul_ir_simulates(rfactored, tmp_path, "rfactor_k32_to_k33")
    compacted = BufferCompaction().apply(rfactored, BufferCompactionOption(tensor="psum_prod"))
    final = BufferCompaction().apply(compacted, BufferCompactionOption(tensor="sbuf_rfactor"))
    assert_matmul_ir_simulates(final, tmp_path, "rfactor_k35")
