"""Tests for nkigym.transforms.ReverseComputeAt."""

from __future__ import annotations

import importlib.util
import pathlib
import tempfile
from test.transforms._fixtures import INPUT_SPECS, build_canonical_ir, build_ladder_state
from test.transforms._ladder_compare import assert_matches_hand

import numpy as np
import pytest

import kernel_transforms as KT
from nkigym.codegen import render
from nkigym.ir.tree import ForNode, ISANode
from nkigym.synthesis.simulate_nki import simulate_fp32
from nkigym.transforms import ReverseComputeAt, ReverseComputeAtOption, TransformLegalityError


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


def test_reverse_rejects_non_fornode_target():
    ir = build_canonical_ir()
    tc = _block_for_op(ir, "NKITensorCopy")
    mm = _block_for_op(ir, "NKIMatmul")
    with pytest.raises(TransformLegalityError, match="ForNode"):
        ReverseComputeAt().apply(ir, ReverseComputeAtOption(block_nid=tc, target_loop_nid=mm, index=-1))


def test_reverse_rejects_target_inside_moved_block():
    ir = build_canonical_ir()
    tc = _block_for_op(ir, "NKITensorCopy")
    own = _first_for_in(ir, tc)
    with pytest.raises(TransformLegalityError, match="descendant|ancestor|own"):
        ReverseComputeAt().apply(ir, ReverseComputeAtOption(block_nid=tc, target_loop_nid=own, index=-1))


def test_reverse_lift_store_under_tensor_copy_renders_and_sims():
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
    new_ir = ReverseComputeAt().apply(ir, ReverseComputeAtOption(block_nid=store, target_loop_nid=m_loop, index=-1))
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


def test_reverse_lift_preserves_covered_dim_across_block_wall():
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
    from nkigym.transforms import ComputeAt, ComputeAtOption, Fuse, Reorder, Split, SplitOption

    trace = [
        (ComputeAt(), ComputeAtOption(block_nid=4, target_loop_nid=11, index=0)),
        (Split(), SplitOption(target_nid=17, factors=(8, 256), target_axis="d2")),
        (ReverseComputeAt(), ReverseComputeAtOption(block_nid=1, target_loop_nid=11, index=1)),
        (Split(), SplitOption(target_nid=3, factors=(2, 1024), target_axis="d1")),
        (ReverseComputeAt(), ReverseComputeAtOption(block_nid=4, target_loop_nid=22, index=0)),
    ]
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), Fuse(), Reorder(), ComputeAt(), ReverseComputeAt()])
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


def test_reverse_lift_deeply_nested_load_preserves_dim_driver():
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
    from nkigym.transforms import ComputeAt, ComputeAtOption, Fuse, Reorder, Split, SplitOption

    trace = [
        (ComputeAt(), ComputeAtOption(block_nid=1, target_loop_nid=13, index=0)),
        (ReverseComputeAt(), ReverseComputeAtOption(block_nid=1, target_loop_nid=5, index=1)),
        (ReverseComputeAt(), ReverseComputeAtOption(block_nid=1, target_loop_nid=11, index=0)),
        (ReverseComputeAt(), ReverseComputeAtOption(block_nid=4, target_loop_nid=11, index=0)),
        (Split(), SplitOption(target_nid=17, factors=(2, 4, 256), target_axis="d2")),
        (Split(), SplitOption(target_nid=6, factors=(2, 4, 256), target_axis="d2")),
        (ReverseComputeAt(), ReverseComputeAtOption(block_nid=1, target_loop_nid=24, index=0)),
        (Split(), SplitOption(target_nid=9, factors=(4, 2, 256), target_axis="d2")),
        (Split(), SplitOption(target_nid=25, factors=(2, 2), target_axis=None)),
        (ReverseComputeAt(), ReverseComputeAtOption(block_nid=1, target_loop_nid=23, index=1)),
    ]
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), Fuse(), Reorder(), ComputeAt(), ReverseComputeAt()])
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


@pytest.mark.parametrize("before_n, hand", [(11, KT.kernel_12), (13, KT.kernel_14)])
def test_reverse_rung_byte_exact(before_n, hand):
    """Each ReverseComputeAt rung reproduces its hand kernel byte-exact."""
    ir = build_ladder_state(before_n + 1)
    assert_matches_hand(render(ir), hand)


def test_psum_hoist_descends_and_compacts():
    """After k11->k12, psum_prod is declared inside the matmul block and compacted to one tile."""
    from test.transforms._fixtures import build_ladder_state

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


def test_reverse_rejects_lifting_full_read_consumer_into_tiled_producer_loop():
    """A consumer reading a buffer at FULL extent must not be lifted inside a loop
    that a co-located producer writes that buffer tiled by — it would read slices
    the producer has not yet written this iteration (region-coverage / COVER edge).

    Small (256x256x512) fixture, deterministic trace: the store (reads sbuf_prod
    full-N) is lifted under the tensor_copy's i_d1_0 nested in i_d2_0, where the
    tensor_copy writes sbuf_prod tiled by i_d2_0. The COVER edge
    ``i_d2_0-loop -> store`` would point backward -> rejected.
    """
    from test.transforms._fixtures import SMALL_INPUT_SPECS, f_matmul_small

    from nkigym.environment import KernelMDP
    from nkigym.transforms import (
        ComputeAt,
        ComputeAtOption,
        Fuse,
        Reorder,
        ReorderOption,
        Split,
        SplitOption,
        TransformLegalityError,
    )

    setup = [
        (ComputeAt(), ComputeAtOption(block_nid=1, target_loop_nid=5, index=0)),
        (Reorder(), ReorderOption(outer_nid=11, inner_nid=12)),
        (Split(), SplitOption(target_nid=16, factors=(4, 128), target_axis="d2")),
        (ReverseComputeAt(), ReverseComputeAtOption(block_nid=1, target_loop_nid=5, index=2)),
        (ReverseComputeAt(), ReverseComputeAtOption(block_nid=1, target_loop_nid=8, index=0)),
        (ReverseComputeAt(), ReverseComputeAtOption(block_nid=1, target_loop_nid=8, index=2)),
        (ReverseComputeAt(), ReverseComputeAtOption(block_nid=4, target_loop_nid=22, index=0)),
        (Reorder(), ReorderOption(outer_nid=15, inner_nid=20)),
    ]
    env = KernelMDP(
        f_matmul_small, SMALL_INPUT_SPECS, transforms=[Split(), Fuse(), Reorder(), ComputeAt(), ReverseComputeAt()]
    )
    state = env.reset()
    for action in setup:
        state = env.step(state, action)
    bad = ReverseComputeAtOption(block_nid=17, target_loop_nid=20, index=1)
    with pytest.raises(TransformLegalityError, match="reorders dependency edge"):
        ReverseComputeAt().apply(state, bad)
    assert not any(o.block_nid == 17 and o.target_loop_nid == 20 for o in ReverseComputeAt().analyze(state))
