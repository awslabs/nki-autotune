"""Tests for nkigym.transforms.code_motion._move (structural move)."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

from nkigym.ir.tree import ISANode
from nkigym.transforms.code_motion import _move


def _block_for_op(ir, op_name: str) -> int:
    for nid in ir.tree.blocks():
        leaves = [d for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)]
        if len(leaves) == 1 and ir.tree.data(leaves[0]).op_cls.__name__ == op_name:
            return nid
    raise AssertionError(f"no leaf block for {op_name}")


def _innermost_for(ir, block_nid: int) -> int:
    leaf = next(d for d in ir.tree.preorder(block_nid) if isinstance(ir.tree.data(d), ISANode))
    return ir.tree.ancestors(leaf)[-1]


def test_move_lifts_tensor_copy_under_matmul_inner_loop():
    """Lifting tensor_copy under the matmul's innermost loop nests it there."""
    ir = build_canonical_ir()
    tc = _block_for_op(ir, "NKITensorCopy")
    mm = _block_for_op(ir, "NKIMatmul")
    target = _innermost_for(ir, mm)
    _move(ir, block_nid=tc, target_loop_nid=target, index=-1)
    assert tc in ir.tree.descendants(target)


def test_reverse_compute_at_allows_fold_covering_its_own_ko():
    """The two-stage fold accumulates across its ENCLOSING ko (its sbuf_prod
    memset dominates ko via a CARRY edge), so covering ko by that loop is SAFE
    and must be allowed — the kernel_target fold-inlining precondition."""
    import pytest

    from test.transforms._fixtures import INPUT_SPECS, f_matmul

    from nkigym.environment import KernelMDP
    from nkigym.transforms import Reorder, ReorderOption, RFactor, RFactorOption, Split, SplitOption

    def mm_loop(state, loop_var):
        from nkigym.ir.tree import ForNode, ISANode

        leaf = next(
            n
            for n in state.tree.preorder()
            if isinstance(state.tree.data(n), ISANode) and state.tree.data(n).op_cls.__name__ == "NKIMatmul"
        )
        return next(
            a
            for a in state.tree.ancestors(leaf)
            if isinstance(state.tree.data(a), ForNode) and state.tree.data(a).loop_var == loop_var
        )

    def fold_blk(state):
        from nkigym.ir.tree import ISANode

        for nid in state.tree.blocks():
            leaves = [d for d in state.tree.descendants(nid) if isinstance(state.tree.data(d), ISANode)]
            if len(leaves) == 1 and state.tree.data(leaves[0]).op_cls.__name__ == "NKITensorTensor":
                return nid
        raise AssertionError("no fold block")

    def fold_leaf(state):
        from nkigym.ir.tree import ISANode

        return next(d for d in state.tree.descendants(fold_blk(state)) if isinstance(state.tree.data(d), ISANode))

    def fold_loop(state, loop_var):
        from nkigym.ir.tree import ForNode

        return next(
            d
            for d in state.tree.descendants(fold_blk(state))
            if isinstance(state.tree.data(d), ForNode) and state.tree.data(d).loop_var == loop_var
        )

    from nkigym.ops.base import AxisRole
    from nkigym.transforms.code_motion import _check_same_loop_prefix

    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), Reorder(), RFactor()])
    s = env.reset()
    s = Split().apply(s, SplitOption(target_nid=mm_loop(s, "i_d0_0"), factors=(2, 8), target_axis=None))
    s = Split().apply(s, SplitOption(target_nid=mm_loop(s, "i_d1_0"), factors=(4, 4), target_axis=None))
    s = Reorder().apply(s, ReorderOption(outer_nid=mm_loop(s, "i_d1_1"), inner_nid=mm_loop(s, "i_d2_0")))
    s = Reorder().apply(s, ReorderOption(outer_nid=mm_loop(s, "i_d1_0"), inner_nid=mm_loop(s, "i_d2_0")))
    s = Reorder().apply(s, ReorderOption(outer_nid=mm_loop(s, "i_d0_1"), inner_nid=mm_loop(s, "i_d2_0")))
    s = Reorder().apply(s, ReorderOption(outer_nid=mm_loop(s, "i_d0_0"), inner_nid=mm_loop(s, "i_d2_0")))
    s = Reorder().apply(s, ReorderOption(outer_nid=mm_loop(s, "i_d0_1"), inner_nid=mm_loop(s, "i_d1_0")))
    s = Reorder().apply(s, ReorderOption(outer_nid=mm_loop(s, "i_d0_1"), inner_nid=mm_loop(s, "i_d1_1")))
    s = RFactor().apply(s, RFactorOption(target_loop_nid=mm_loop(s, "i_d0_0"), factor_axis=0))
    s = Split().apply(s, SplitOption(target_nid=fold_leaf(s), factors=(4, 512), target_axis="d2"))
    s = Split().apply(s, SplitOption(target_nid=fold_loop(s, "i_d1_0"), factors=(4, 4), target_axis=None))

    """Barrier 1 is isolated here via _check_same_loop_prefix and the dependency
    check (span-promotion verifies init-domination). The fold's own enclosing
    i_d0_0 is allowed (init dominates that loop). (The end-to-end
    CodeMotion (lift) of the fold is deferred to the Task 3 ladder, where the
    drain tensor_copy co-locates
    FIRST so the copy->fold RAW on sbuf_rfactor is satisfied; that ordering
    concern is the separate dependency check, not this reduction guard.)"""
    fold = fold_blk(s)
    fold_block = s.tree.data(fold)
    assert any(iv.axis == "d0" and iv.role == AxisRole.ACCUMULATION for iv in fold_block.iter_vars)
    target_seq = _check_same_loop_prefix(s, fold, mm_loop(s, "i_d1_1"))
    assert ("i_d0_0", 2) in target_seq, "ko (i_d0_0) must be in the matched prefix (allowed self-domination)"


def test_span_promotion_rejects_memset_sunk_into_matmul_kloop():
    """Init-domination: sinking the psum memset INTO the matmul's K loop is
    rejected — psum_prod is carried across K, span-promotion widens both to
    K-span so the init can no longer precede the accumulation. Verdict via the
    production insertion query on the pre-move tree (no hardcoded nids)."""
    from nkigym.ir.tree import ForNode, ISANode

    ir = build_canonical_ir()
    memset_blk = _block_for_op(ir, "NKIMemset")
    matmul_blk = _block_for_op(ir, "NKIMatmul")
    matmul_leaf = next(d for d in ir.tree.preorder(matmul_blk) if isinstance(ir.tree.data(d), ISANode))
    kloop = next(
        a for a in ir.tree.ancestors(matmul_leaf)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == "i_d0_0"
    )
    moved_leaf = ir.dependency._resolve(memset_blk)
    assert ir.dependency.first_backward_edge_for_insertion(moved_leaf, kloop, 0) is not None


def test_span_promotion_allows_pure_load_sunk_into_matmul_kloop():
    """A pure producer (the lhs_T load writes sbuf_lhs_T, never rmw'd -> NOT
    carried) may sink INTO the matmul's K loop: no span-promotion applies, and it
    still precedes the matmul that consumes it. Legal -> None. This is the benign
    reload sibling of the rejected accumulation-into-K case."""
    from nkigym.ir.tree import ForNode, ISANode

    ir = build_canonical_ir()
    load_blk = _block_for_op(ir, "NKILoad")
    matmul_blk = _block_for_op(ir, "NKIMatmul")
    matmul_leaf = next(d for d in ir.tree.preorder(matmul_blk) if isinstance(ir.tree.data(d), ISANode))
    kloop = next(
        a for a in ir.tree.ancestors(matmul_leaf)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == "i_d0_0"
    )
    moved_leaf = ir.dependency._resolve(load_blk)
    assert ir.dependency.first_backward_edge_for_insertion(moved_leaf, kloop, -2) is None


def test_code_motion_allows_output_store_sink():
    """The output store (writes the return tensor) may sink under the drain's N
    loop — the dropped output-block guard would have rejected it; span-promotion
    permits it (drain writes the sbuf_prod slice the store reads, same N-iter).
    This is the _fixtures rung_13_14 move, done via CodeMotion."""
    from test.transforms._fixtures import build_ladder_state, _ladder_helpers
    from nkigym.transforms.code_motion import CodeMotion, CodeMotionOption

    state = build_ladder_state(13)
    blk, _leaf, _loop, _inner, _mm_loop, tc_loop = _ladder_helpers()
    store_blk = blk(state, "NKIStore")
    d2 = tc_loop(state, "i_d2_0")
    opt = CodeMotionOption(block_nid=store_blk, target_loop_nid=d2, index=-1)
    new_ir = CodeMotion().apply(state, opt)
    assert new_ir is not None
    assert any(o.block_nid == store_blk and o.target_loop_nid == d2 for o in CodeMotion().analyze(state))


"""Migrated tests from test_compute_at.py and test_reverse_compute_at.py."""

import importlib.util
import pathlib
import tempfile
from test.transforms._fixtures import INPUT_SPECS, build_ladder_state

import numpy as np
import pytest

from nkigym.codegen import render
from nkigym.ir.tree import ForNode
from nkigym.synthesis.simulate_nki import simulate_fp32
from nkigym.transforms import CodeMotion, CodeMotionOption, Split, SplitOption, TransformLegalityError, Reorder, ReorderOption


def _block_for_op(ir, op_name: str) -> int:
    for nid in ir.tree.blocks():
        leaves = [d for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)]
        if len(leaves) == 1 and ir.tree.data(leaves[0]).op_cls.__name__ == op_name:
            return nid
    raise AssertionError(f"no leaf block for {op_name}")


def _first_for_in(ir, block_nid: int) -> int:
    for d in ir.tree.preorder(block_nid):
        if isinstance(ir.tree.data(d), ForNode):
            return d
    raise AssertionError("no ForNode")


def _load_block_reading(ir, tensor: str) -> int:
    """Return the single-leaf load block whose ISA ``src`` reads ``tensor``."""
    for nid in ir.tree.blocks():
        leaves = [d for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)]
        if len(leaves) == 1:
            leaf = ir.tree.data(leaves[0])
            if leaf.op_cls.__name__ == "NKILoad" and leaf.operand_bindings["src"].tensor == tensor:
                return nid
    raise AssertionError(f"no single-leaf load block reading {tensor}")


def test_code_motion_rejects_non_fornode_target():
    ir = build_canonical_ir()
    load = _block_for_op(ir, "NKILoad")
    mm = _block_for_op(ir, "NKIMatmul")
    with pytest.raises(TransformLegalityError, match="ForNode"):
        CodeMotion().apply(ir, CodeMotionOption(block_nid=load, target_loop_nid=mm, index=-1))


def test_code_motion_rejects_target_inside_moved_block():
    ir = build_canonical_ir()
    tc = _block_for_op(ir, "NKITensorCopy")
    own = _first_for_in(ir, tc)
    with pytest.raises(TransformLegalityError, match="descendant|ancestor|own"):
        CodeMotion().apply(ir, CodeMotionOption(block_nid=tc, target_loop_nid=own, index=-1))


def test_code_motion_rejects_sinking_writer_under_accumulation_loop():
    """Sinking the memset (accumulator init) under the matmul K loop is rejected
    by the dependency model (memset->K-loop carry edge would point backward),
    not an ad-hoc role guard."""
    ir = build_canonical_ir()
    memset = _block_for_op(ir, "NKIMemset")
    mm = _block_for_op(ir, "NKIMatmul")
    kloop = next(
        d for d in ir.tree.preorder(mm) if isinstance(ir.tree.data(d), ForNode) and ir.tree.data(d).loop_var == "i_d0_0"
    )
    with pytest.raises(TransformLegalityError, match="reorder|dependency"):
        CodeMotion().apply(ir, CodeMotionOption(block_nid=memset, target_loop_nid=kloop, index=0))
    assert not any(o.block_nid == memset and o.target_loop_nid == kloop for o in CodeMotion().analyze(ir))


def test_code_motion_rejects_consumer_sunk_before_producer():
    """Hole #1: sinking the tensor_copy (consumer of psum_prod) under the memset's
    loop would place it before the matmul producer -> rejected by the same model."""
    ir = build_canonical_ir()
    tc = _block_for_op(ir, "NKITensorCopy")
    memset = _block_for_op(ir, "NKIMemset")
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
    rhs_load = _load_block_reading(ir, "rhs")
    tc = _block_for_op(ir, "NKITensorCopy")
    tc_loop = _first_for_in(ir, tc)
    with pytest.raises(TransformLegalityError, match="reorder|dependency"):
        CodeMotion().apply(ir, CodeMotionOption(block_nid=rhs_load, target_loop_nid=tc_loop, index=0))
    assert not any(o.block_nid == rhs_load and o.target_loop_nid == tc_loop for o in CodeMotion().analyze(ir))


def test_analyze_does_not_crash_on_transformed_states():
    """analyze must filter (not crash on) candidates across ladder states 1..12.

    The move-sim legality runs ``_move`` on every candidate, including re-moving
    an already-nested block. A splice that left a node double-parented used to
    crash the downstream ``Dependency`` rebuild; ``analyze`` must filter such a
    candidate, never raise.
    """
    for n in range(1, 13):
        ir = build_ladder_state(n)
        CodeMotion().analyze(ir)


def test_code_motion_sink_load_under_matmul_renders_and_sims():
    """Sink lhs_T load under the matmul's inner loop; render + sim."""
    ir = build_canonical_ir()
    load = _block_for_op(ir, "NKILoad")
    mm = _block_for_op(ir, "NKIMatmul")
    leaf = next(d for d in ir.tree.preorder(mm) if isinstance(ir.tree.data(d), ISANode))
    inner = ir.tree.ancestors(leaf)[-1]
    new_ir = CodeMotion().apply(ir, CodeMotionOption(block_nid=load, target_loop_nid=inner, index=-2))
    assert load in new_ir.tree.descendants(inner)
    rng = np.random.default_rng(0)
    inputs = {n: rng.standard_normal(s).astype(np.float32) for n, (s, _d) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    path = pathlib.Path(tempfile.mkdtemp()) / "k.py"
    path.write_text(render(new_ir))
    spec = importlib.util.spec_from_file_location("k", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    actual = np.asarray(simulate_fp32(mod.nki_f_matmul)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_code_motion_lift_store_under_tensor_copy_renders_and_sims():
    """Full-extent lift of the store under the tensor_copy's PARALLEL M-loop renders + sims.

    The store consumes ``sbuf_prod``, which the tensor_copy's M-loop does not
    carry (PARALLEL role), so the lift respects carry-domination and is legal.
    Lifting the tensor_copy itself into the matmul's K (ACCUMULATION) loop is
    correctly rejected by the dependency model and is exercised by the
    rejection tests instead.
    """
    ir = build_canonical_ir()
    store = _block_for_op(ir, "NKIStore")
    tc = _block_for_op(ir, "NKITensorCopy")
    m_loop = _first_for_in(ir, tc)
    new_ir = CodeMotion().apply(ir, CodeMotionOption(block_nid=store, target_loop_nid=m_loop, index=-1))
    assert store in new_ir.tree.descendants(m_loop)
    rng = np.random.default_rng(0)
    inputs = {n: rng.standard_normal(s).astype(np.float32) for n, (s, _d) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    src = render(new_ir)
    path = pathlib.Path(tempfile.mkdtemp()) / "k.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("k", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    actual = np.asarray(simulate_fp32(mod.nki_f_matmul)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_code_motion_lift_preserves_covered_dim_across_block_wall():
    """Regression: lifting a block under a target nested inside ANOTHER block must
    preserve a covered dim driven by an enclosing loop above the intervening
    BlockNode wall — not collapse it to Const(0).

    Fixed deterministic trace: the rhs load (block 4, d0 driven by the matmul
    block's enclosing K-loop) is lifted under loop 22, which sits inside a
    different block. ``normalize_block``'s dim gather must see the K-loop above
    that wall (``_all_enclosing_loops``); otherwise the load's ``rhs`` source
    offset loses ``i_d0_0*128`` (reads tile 0 every K-step) -> matmul reads
    uninitialised sbuf_rhs tiles -> NaN.
    """
    from test.transforms._fixtures import f_matmul
    from nkigym.environment import KernelMDP

    trace = [
        (CodeMotion(), CodeMotionOption(block_nid=4, target_loop_nid=11, index=0)),
        (Split(), SplitOption(target_nid=17, factors=(8, 256), target_axis="d2")),
        (CodeMotion(), CodeMotionOption(block_nid=1, target_loop_nid=11, index=1)),
        (Split(), SplitOption(target_nid=3, factors=(2, 1024), target_axis="d1")),
        (CodeMotion(), CodeMotionOption(block_nid=4, target_loop_nid=22, index=0)),
    ]
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), CodeMotion()])
    state = env.reset()
    for action in trace:
        state = env.step(state, action)
    rng = np.random.default_rng(0)
    inputs = {n: rng.standard_normal(s).astype(np.float32) for n, (s, _d) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    path = pathlib.Path(tempfile.mkdtemp()) / "k.py"
    path.write_text(render(state))
    spec = importlib.util.spec_from_file_location("k", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    actual = np.asarray(simulate_fp32(mod.nki_f_matmul)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_code_motion_lift_deeply_nested_load_preserves_dim_driver():
    """Regression (mirror of the across-block-wall fix, in _domain_solve): a load
    nested SEVERAL blocks deep, whose covered d0 is driven by a K-loop above two
    intervening BlockNode walls, must keep that driver when lifted.

    ``dim_loops_of_block`` gathered enclosing loops with a block-local walk that
    reset at each BlockNode, so the deep lhs-load's d0 driver (the matmul block's
    i_d0_0, above the rhs-load and lhs-load block walls) was dropped -> empty
    dim_loops -> the load's lhs_T source offset lost i_d0_0*128 -> NaN. The gather
    now spans all ancestor ForNodes filtered by the block's bound loop vars.
    """
    from test.transforms._fixtures import f_matmul
    from nkigym.environment import KernelMDP

    trace = [
        (CodeMotion(), CodeMotionOption(block_nid=1, target_loop_nid=13, index=0)),
        (CodeMotion(), CodeMotionOption(block_nid=1, target_loop_nid=5, index=1)),
        (CodeMotion(), CodeMotionOption(block_nid=1, target_loop_nid=11, index=0)),
        (CodeMotion(), CodeMotionOption(block_nid=4, target_loop_nid=11, index=0)),
        (Split(), SplitOption(target_nid=17, factors=(2, 4, 256), target_axis="d2")),
        (Split(), SplitOption(target_nid=6, factors=(2, 4, 256), target_axis="d2")),
        (CodeMotion(), CodeMotionOption(block_nid=1, target_loop_nid=24, index=0)),
        (Split(), SplitOption(target_nid=9, factors=(4, 2, 256), target_axis="d2")),
        (Split(), SplitOption(target_nid=25, factors=(2, 2), target_axis=None)),
        (CodeMotion(), CodeMotionOption(block_nid=1, target_loop_nid=23, index=1)),
    ]
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), CodeMotion()])
    state = env.reset()
    for action in trace:
        state = env.step(state, action)
    rng = np.random.default_rng(0)
    inputs = {n: rng.standard_normal(s).astype(np.float32) for n, (s, _d) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    path = pathlib.Path(tempfile.mkdtemp()) / "k.py"
    path.write_text(render(state))
    spec = importlib.util.spec_from_file_location("k", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    actual = np.asarray(simulate_fp32(mod.nki_f_matmul)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_psum_hoist_descends_and_compacts():
    """After k11->k12, psum_prod is declared inside the matmul block and compacted to one tile."""
    ir = build_ladder_state(12)
    decls = {buf.name: (nid, buf) for nid in ir.tree.blocks() for buf in ir.tree.data(nid).alloc_buffers}
    nid, buf = decls["psum_prod"]
    assert nid != ir.tree.root, "psum_prod did not descend from root"
    assert buf.shape == (128, 512), f"psum_prod not compacted to one tile: {buf.shape}"


@pytest.mark.parametrize("n", list(range(1, 15)))
def test_ladder_state_sims(n):
    """Every ladder state 1..14 renders and CPU-sims to the matmul golden.

    Pairs with the byte-exact rung tests: byte-match alone can pass by luck on
    a structurally-wrong kernel and sim alone can pass on a kernel that differs
    cosmetically; requiring both per state pins each rung end-to-end.
    """
    ir = build_ladder_state(n)
    rng = np.random.default_rng(0)
    inputs = {name: rng.standard_normal(s).astype(np.float32) for name, (s, _d) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    path = pathlib.Path(tempfile.mkdtemp()) / "k.py"
    path.write_text(render(ir))
    spec = importlib.util.spec_from_file_location("k", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    actual = np.asarray(simulate_fp32(mod.nki_f_matmul)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


