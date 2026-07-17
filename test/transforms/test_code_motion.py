"""Unit and legality tests for CodeMotion."""

from __future__ import annotations

from test.transforms._fixtures import _ladder_helpers, build_canonical_ir, build_ladder_state
from test.transforms._helpers import block_for_op, first_for_in, load_block_reading, matmul_loop
from test.transforms._pipeline_fixtures import m_loop_and_children, tuned_ir

import pytest

from nkigym.ir.tree import BlockNode, ForNode, ISANode
from nkigym.ops.base import AxisRole
from nkigym.transforms import (
    CodeMotion,
    CodeMotionOption,
    Reorder,
    ReorderOption,
    RFactor,
    RFactorOption,
    SoftwarePipeline,
    SoftwarePipelineOption,
    Split,
    SplitOption,
    TransformLegalityError,
)
from nkigym.transforms.code_motion import _check_same_loop_prefix, _move


def _innermost_for(ir, block_nid: int) -> int:
    leaf = next(d for d in ir.tree.preorder(block_nid) if isinstance(ir.tree.data(d), ISANode))
    return ir.tree.ancestors(leaf)[-1]


def test_move_lifts_tensor_copy_under_matmul_inner_loop():
    """Lifting tensor_copy under the matmul's innermost loop nests it there."""
    ir = build_canonical_ir()
    tc = block_for_op(ir, "NKITensorCopy")
    mm = block_for_op(ir, "NKIMatmul")
    target = _innermost_for(ir, mm)
    _move(ir, block_nid=tc, target_loop_nid=target, index=-1)
    assert tc in ir.tree.descendants(target)


def test_reverse_compute_at_allows_fold_covering_its_own_ko():
    """The two-stage fold accumulates across its ENCLOSING ko (its sbuf_prod
    memset dominates ko via a CARRY edge), so covering ko by that loop is SAFE
    and must be allowed — the kernel_target fold-inlining precondition."""
    state = build_canonical_ir()
    state = Split().apply(state, SplitOption(target_nid=matmul_loop(state, "i_d0_0"), factors=(2, 8), target_axis=None))
    state = Split().apply(state, SplitOption(target_nid=matmul_loop(state, "i_d1_0"), factors=(4, 4), target_axis=None))
    for outer, inner in (
        ("i_d1_1", "i_d2_0"),
        ("i_d1_0", "i_d2_0"),
        ("i_d0_1", "i_d2_0"),
        ("i_d0_0", "i_d2_0"),
        ("i_d0_1", "i_d1_0"),
        ("i_d0_1", "i_d1_1"),
    ):
        state = Reorder().apply(
            state, ReorderOption(outer_nid=matmul_loop(state, outer), inner_nid=matmul_loop(state, inner))
        )
    state = RFactor().apply(state, RFactorOption(target_loop_nid=matmul_loop(state, "i_d0_0"), factor_axis=0))

    """After the ki-anchored RFactor the fold is ALREADY per-N-tile (d2 free extent
    512, region ``i_d2_0*512 : +512``) and per-Mi-tile, nested directly under
    ``i_d2_0 > i_d0_0(ko) > i_d1_0 > i_d1_1`` with no block-local loops — so the old
    ``Split(fold, d2, (4,512))`` + ``Split(fold_loop i_d1_0, (4,4))`` scaffolding
    (which shaped a 2048-wide ko-anchored fold) is now moot and is dropped.

    Barrier 1 is isolated here via _check_same_loop_prefix and the dependency check
    (span-promotion verifies init-domination). The fold's own enclosing i_d0_0 is
    allowed (init dominates that loop)."""
    fold = block_for_op(state, "NKITensorTensor")
    fold_block = state.tree.data(fold)
    assert any(iv.axis == "d0" and iv.role == AxisRole.ACCUMULATION for iv in fold_block.iter_vars)
    target_seq = _check_same_loop_prefix(state, fold, matmul_loop(state, "i_d1_1"))
    assert ("i_d0_0", 2) in target_seq, "ko (i_d0_0) must be in the matched prefix (allowed self-domination)"


def test_code_motion_allows_output_store_sink():
    """The output store (writes the return tensor) may sink under the drain's N
    loop — the dropped output-block guard would have rejected it; span-promotion
    permits it (drain writes the sbuf_prod slice the store reads, same N-iter).
    This is the _fixtures rung_13_14 move, done via CodeMotion."""
    state = build_ladder_state(13)
    blk, _leaf, _loop, _inner, _mm_loop, tc_loop = _ladder_helpers()
    store_blk = blk(state, "NKIStore")
    d2 = tc_loop(state, "i_d2_0")
    opt = CodeMotionOption(block_nid=store_blk, target_loop_nid=d2, index=-1)
    new_ir = CodeMotion().apply(state, opt)
    assert new_ir is not None
    assert any(o.block_nid == store_blk and o.target_loop_nid == d2 for o in CodeMotion().analyze(state))


def test_code_motion_rejects_non_fornode_target():
    ir = build_canonical_ir()
    load = block_for_op(ir, "NKILoad")
    mm = block_for_op(ir, "NKIMatmul")
    with pytest.raises(TransformLegalityError, match="ForNode"):
        CodeMotion().apply(ir, CodeMotionOption(block_nid=load, target_loop_nid=mm, index=-1))


def test_code_motion_rejects_target_inside_moved_block():
    ir = build_canonical_ir()
    tc = block_for_op(ir, "NKITensorCopy")
    own = first_for_in(ir, tc)
    with pytest.raises(TransformLegalityError, match="descendant|ancestor|own"):
        CodeMotion().apply(ir, CodeMotionOption(block_nid=tc, target_loop_nid=own, index=-1))


def test_code_motion_rejects_lift_that_drops_bound_enclosing_loop():
    """A move cannot detach a block from an enclosing loop that drives its regions."""
    ir = build_canonical_ir()
    memset = block_for_op(ir, "NKIMemset")
    lhs_load = load_block_reading(ir, "lhs_T")
    lhs_leaf = next(d for d in ir.tree.preorder(lhs_load) if isinstance(ir.tree.data(d), ISANode))
    ir = Split().apply(ir, SplitOption(target_nid=lhs_leaf, factors=(2, 1024), target_axis="d1"))
    memset_d1 = next(
        d
        for d in ir.tree.preorder(memset)
        if isinstance(ir.tree.data(d), ForNode) and ir.tree.data(d).loop_var == "i_d1_0"
    )
    ir = Split().apply(ir, SplitOption(target_nid=memset_d1, factors=(2, 8), target_axis=None))
    lhs_loops = [d for d in ir.tree.preorder(lhs_load) if isinstance(ir.tree.data(d), ForNode)]
    outer_d0, inner_d1 = lhs_loops
    nested = CodeMotion().apply(ir, CodeMotionOption(block_nid=memset, target_loop_nid=inner_d1, index=0))
    option = CodeMotionOption(block_nid=memset, target_loop_nid=outer_d0, index=0)
    with pytest.raises(TransformLegalityError, match="enclosing|bind"):
        CodeMotion().apply(nested, option)
    assert option not in CodeMotion().analyze(nested)


def test_code_motion_rejects_versioned_buffer_crossing_pipeline_boundary():
    """A block touching a multi-version buffer cannot leave its pipeline loop."""
    ir = tuned_ir()
    pipeline_loop, _children = m_loop_and_children(ir)
    ir = SoftwarePipeline().apply(ir, SoftwarePipelineOption(loop_nid=pipeline_loop, stages=(0, 0, 1), order=(0, 1, 2)))
    tensor_copy = block_for_op(ir, "NKITensorCopy")
    store = block_for_op(ir, "NKIStore")
    target = first_for_in(ir, store)
    option = CodeMotionOption(block_nid=tensor_copy, target_loop_nid=target, index=0)
    with pytest.raises(TransformLegalityError, match="pipeline|version"):
        CodeMotion().apply(ir, option)
    assert option not in CodeMotion().analyze(ir)


def test_code_motion_rejects_sinking_writer_under_accumulation_loop():
    """Sinking the memset (accumulator init) under the matmul K loop is rejected
    by the dependency model (memset->K-loop carry edge would point backward),
    not an ad-hoc role guard."""
    ir = build_canonical_ir()
    memset = block_for_op(ir, "NKIMemset")
    mm = block_for_op(ir, "NKIMatmul")
    kloop = next(
        d for d in ir.tree.preorder(mm) if isinstance(ir.tree.data(d), ForNode) and ir.tree.data(d).loop_var == "i_d0_0"
    )
    with pytest.raises(TransformLegalityError, match="reorder|dependency"):
        CodeMotion().apply(ir, CodeMotionOption(block_nid=memset, target_loop_nid=kloop, index=0))
    assert not any(o.block_nid == memset and o.target_loop_nid == kloop for o in CodeMotion().analyze(ir))


def test_code_motion_rejects_hoisting_rmw_reset_across_reduction_loop():
    """Hoisting an invariant reset out of an invariant RMW loop changes its frequency."""
    ir = build_canonical_ir()
    ir = Split().apply(ir, SplitOption(target_nid=matmul_loop(ir, "i_d0_0"), factors=(2, 4, 2), target_axis=None))
    ir = Split().apply(ir, SplitOption(target_nid=matmul_loop(ir, "i_d1_0"), factors=(4, 4), target_axis=None))
    ir = Reorder().apply(ir, ReorderOption(outer_nid=matmul_loop(ir, "i_d0_2"), inner_nid=matmul_loop(ir, "i_d1_0")))
    ir = Reorder().apply(ir, ReorderOption(outer_nid=matmul_loop(ir, "i_d0_1"), inner_nid=matmul_loop(ir, "i_d1_0")))
    ir = RFactor().apply(ir, RFactorOption(target_loop_nid=matmul_loop(ir, "i_d0_0"), factor_axis=0))
    psum_memset_leaf = next(
        nid
        for nid in ir.tree.preorder()
        if isinstance((node := ir.tree.data(nid)), ISANode)
        and node.op_cls.__name__ == "NKIMemset"
        and node.operand_bindings["dst"].tensor == "psum_prod"
    )
    psum_memset = next(
        ancestor
        for ancestor in reversed(ir.tree.ancestors(psum_memset_leaf))
        if isinstance(ir.tree.data(ancestor), BlockNode)
    )
    option = CodeMotionOption(block_nid=psum_memset, target_loop_nid=matmul_loop(ir, "i_d1_0"), index=0)

    with pytest.raises(TransformLegalityError, match="reset|read-modify-write"):
        CodeMotion().apply(ir, option)
    assert option not in CodeMotion().analyze(ir)


def test_code_motion_rejects_rmw_moved_into_invariant_reset_loop():
    """An accumulator cannot enter a loop that resets its region each iteration."""
    ir = build_canonical_ir()
    memset = block_for_op(ir, "NKIMemset")
    lhs_load = load_block_reading(ir, "lhs_T")
    rhs_load = load_block_reading(ir, "rhs")
    matmul = block_for_op(ir, "NKIMatmul")
    target = next(
        nid
        for nid in ir.tree.preorder(lhs_load)
        if isinstance((node := ir.tree.data(nid)), ForNode) and node.loop_var == "i_d0_0"
    )
    ir = CodeMotion().apply(ir, CodeMotionOption(block_nid=memset, target_loop_nid=target, index=0))
    ir = CodeMotion().apply(ir, CodeMotionOption(block_nid=rhs_load, target_loop_nid=target, index=-1))
    option = CodeMotionOption(block_nid=matmul, target_loop_nid=target, index=-1)

    with pytest.raises(TransformLegalityError, match="reset|read-modify-write"):
        CodeMotion().apply(ir, option)
    assert option not in CodeMotion().analyze(ir)


def test_code_motion_rejects_hoisting_consumer_out_of_producer_loop():
    """A consumer cannot leave a loop that produces its invariant input each iteration."""
    ir = build_canonical_ir()
    ir = Split().apply(ir, SplitOption(target_nid=matmul_loop(ir, "i_d0_0"), factors=(2, 2, 4), target_axis=None))
    ir = RFactor().apply(ir, RFactorOption(target_loop_nid=matmul_loop(ir, "i_d0_0"), factor_axis=0))
    fold = block_for_op(ir, "NKITensorTensor")
    option = CodeMotionOption(block_nid=fold, target_loop_nid=matmul_loop(ir, "i_d0_0"), index=1)

    with pytest.raises(TransformLegalityError, match="producer|consumer|scope"):
        CodeMotion().apply(ir, option)
    assert option not in CodeMotion().analyze(ir)


def test_code_motion_rejects_consumer_moved_to_equal_sibling_loop():
    """Equal loop payloads do not make distinct sibling execution scopes equivalent."""
    ir = build_canonical_ir()
    ir = Split().apply(ir, SplitOption(target_nid=matmul_loop(ir, "i_d0_0"), factors=(2, 2, 4), target_axis=None))
    ir = RFactor().apply(ir, RFactorOption(target_loop_nid=matmul_loop(ir, "i_d0_0"), factor_axis=0))
    fold = block_for_op(ir, "NKITensorTensor")
    outer = matmul_loop(ir, "i_d0_0")
    old_inner = matmul_loop(ir, "i_d0_1")
    target_sibling = ir.tree.add_node(ir.tree.data(old_inner), parent=outer)
    option = CodeMotionOption(block_nid=fold, target_loop_nid=target_sibling, index=0)

    with pytest.raises(TransformLegalityError, match="producer|consumer|scope"):
        CodeMotion().apply(ir, option)
    assert option not in CodeMotion().analyze(ir)


def test_code_motion_rejects_consumer_sunk_before_producer():
    """Hole #1: sinking the tensor_copy (consumer of psum_prod) under the memset's
    loop would place it before the matmul producer -> rejected by the same model."""
    ir = build_canonical_ir()
    tc = block_for_op(ir, "NKITensorCopy")
    memset = block_for_op(ir, "NKIMemset")
    memset_loop = next(d for d in ir.tree.preorder(memset) if isinstance(ir.tree.data(d), ForNode))
    with pytest.raises(TransformLegalityError, match="reorder|dependency"):
        CodeMotion().apply(ir, CodeMotionOption(block_nid=tc, target_loop_nid=memset_loop, index=0))


def test_code_motion_rejects_parallel_producer_sunk_past_consumer():
    """The direction bug: sinking the rhs load (PARALLEL producer of sbuf_rhs, no
    carry edge) under the tensor_copy loop places it AFTER the matmul that reads
    sbuf_rhs. The RAW load->matmul edge would point backward; reject it.

    This is the case ``examples/transform_debug.py`` exercises. The buggy check
    rebuilt the dependency graph on the moved tree, where the load-after-matmul
    order re-derives the hazard as a forward WAR matmul->load, hiding the
    violation. The fix freezes edge directions from the original program.
    """
    ir = build_canonical_ir()
    rhs_load = load_block_reading(ir, "rhs")
    tc = block_for_op(ir, "NKITensorCopy")
    tc_loop = first_for_in(ir, tc)
    with pytest.raises(TransformLegalityError, match="reorder|dependency"):
        CodeMotion().apply(ir, CodeMotionOption(block_nid=rhs_load, target_loop_nid=tc_loop, index=0))
    assert not any(o.block_nid == rhs_load and o.target_loop_nid == tc_loop for o in CodeMotion().analyze(ir))
