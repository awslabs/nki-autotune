"""Tests for nkigym.transforms.ComputeAt."""

from __future__ import annotations

import inspect
from test.transforms._fixtures import INPUT_SPECS, build_canonical_ir, build_ladder_state
from test.transforms._ladder_compare import _normalize, assert_matches_hand

import numpy as np
import pytest

import kernel_transforms as KT
from nkigym.codegen import render
from nkigym.ir.tree import ForNode, ISANode
from nkigym.synthesis.simulate_nki import simulate_fp32
from nkigym.transforms import ComputeAt, ComputeAtOption, ReverseComputeAt, Split, SplitOption, TransformLegalityError


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


def test_compute_at_rejects_sinking_output_store():
    """Condition 4: sinking the store (writes hbm_out = return) is illegal."""
    ir = build_canonical_ir()
    store = _block_for_op(ir, "NKIStore")
    mm = _block_for_op(ir, "NKIMatmul")
    target = _first_for_in(ir, mm)
    with pytest.raises(TransformLegalityError, match="output|return"):
        ComputeAt().apply(ir, ComputeAtOption(block_nid=store, target_loop_nid=target, index=-1))


def test_compute_at_rejects_non_fornode_target():
    ir = build_canonical_ir()
    load = _block_for_op(ir, "NKILoad")
    mm = _block_for_op(ir, "NKIMatmul")
    with pytest.raises(TransformLegalityError, match="ForNode"):
        ComputeAt().apply(ir, ComputeAtOption(block_nid=load, target_loop_nid=mm, index=-1))


def test_compute_at_rejects_sinking_writer_under_accumulation_loop():
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
        ComputeAt().apply(ir, ComputeAtOption(block_nid=memset, target_loop_nid=kloop, index=0))
    assert not any(o.block_nid == memset and o.target_loop_nid == kloop for o in ComputeAt().analyze(ir))


def test_compute_at_rejects_consumer_sunk_before_producer():
    """Hole #1: sinking the tensor_copy (consumer of psum_prod) under the memset's
    loop would place it before the matmul producer -> rejected by the same model."""
    ir = build_canonical_ir()
    tc = _block_for_op(ir, "NKITensorCopy")
    memset = _block_for_op(ir, "NKIMemset")
    memset_loop = next(d for d in ir.tree.preorder(memset) if isinstance(ir.tree.data(d), ForNode))
    with pytest.raises(TransformLegalityError, match="reorder|dependency"):
        ComputeAt().apply(ir, ComputeAtOption(block_nid=tc, target_loop_nid=memset_loop, index=0))


def test_analyze_does_not_crash_on_transformed_states():
    """analyze must filter (not crash on) candidates across ladder states 1..12.

    The move-sim legality runs ``_move`` on every candidate, including re-moving
    an already-nested block. A splice that left a node double-parented used to
    crash the downstream ``Dependency`` rebuild; ``analyze`` must filter such a
    candidate, never raise.
    """
    for n in range(1, 13):
        ir = build_ladder_state(n)
        ComputeAt().analyze(ir)
        ReverseComputeAt().analyze(ir)


def test_compute_at_sink_load_under_matmul_renders_and_sims():
    """Sink lhs_T load under the matmul's inner loop; render + sim."""
    ir = build_canonical_ir()
    load = _block_for_op(ir, "NKILoad")
    mm = _block_for_op(ir, "NKIMatmul")
    leaf = next(d for d in ir.tree.preorder(mm) if isinstance(ir.tree.data(d), ISANode))
    inner = ir.tree.ancestors(leaf)[-1]
    new_ir = ComputeAt().apply(ir, ComputeAtOption(block_nid=load, target_loop_nid=inner, index=-2))
    assert load in new_ir.tree.descendants(inner)
    rng = np.random.default_rng(0)
    inputs = {n: rng.standard_normal(s).astype(np.float32) for n, (s, _d) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    import importlib.util
    import pathlib
    import tempfile

    path = pathlib.Path(tempfile.mkdtemp()) / "k.py"
    path.write_text(render(new_ir))
    spec = importlib.util.spec_from_file_location("k", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    actual = np.asarray(simulate_fp32(mod.nki_f_matmul)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_split_lhsT_load_matches_hand_k1():
    """Split lhs_T load d1 -> (16, 128) alone reproduces hand kernel_1 byte-exact.

    kernel_1 keeps sbuf_lhs_T full-size (128, 16, 2048): no buffer is moved,
    so no compaction applies. This isolates the _ladder_compare oracle's
    structural normalization (declaration placement, kwarg order, slice
    arithmetic) from the compaction fix exercised by kernel_2.
    """
    ir = build_canonical_ir()
    load = _block_for_op(ir, "NKILoad")
    load_leaf = next(d for d in ir.tree.preorder(load) if isinstance(ir.tree.data(d), ISANode))
    ir = Split().apply(ir, SplitOption(target_nid=load_leaf, factors=(16, 128), target_axis="d1"))
    assert_matches_hand(render(ir), KT.kernel_1)


def test_compute_at_sink_lhsT_load_matches_hand_k2():
    """Sink lhs_T load under matmul (d0,d1 cover) reproduces hand kernel_2 byte-exact."""
    ir = build_canonical_ir()
    load = _block_for_op(ir, "NKILoad")
    load_leaf = next(d for d in ir.tree.preorder(load) if isinstance(ir.tree.data(d), ISANode))
    ir = Split().apply(ir, SplitOption(target_nid=load_leaf, factors=(16, 128), target_axis="d1"))
    load2 = _block_for_op(ir, "NKILoad")
    mm = _block_for_op(ir, "NKIMatmul")
    d1_loop = next(
        d for d in ir.tree.preorder(mm) if isinstance(ir.tree.data(d), ForNode) and ir.tree.data(d).loop_var == "i_d1_0"
    )
    new_ir = ComputeAt().apply(ir, ComputeAtOption(block_nid=load2, target_loop_nid=d1_loop, index=-2))
    assert_matches_hand(render(new_ir), KT.kernel_2)


@pytest.mark.parametrize(
    "before_n, hand", [(1, KT.kernel_2), (3, KT.kernel_4), (6, KT.kernel_7), (7, KT.kernel_8), (9, KT.kernel_10)]
)
def test_compute_at_rung_byte_exact(before_n, hand):
    """Each forward ComputeAt rung reproduces its hand kernel byte-exact."""
    ir = build_ladder_state(before_n + 1)
    assert_matches_hand(render(ir), hand)


def test_compute_at_partial_coverage_byte_exact():
    """A range(16) load sunk under a range(4) target regenerates a range(4) residual.

    Byte oracle for the region-regen residual path the FULL-coverage ladder never
    exercises. ``kernel_partial`` is hand-written and sim-verified
    (``python kernel_transforms.py`` prints ``pass=True``). The current ComputeAt
    produces a numerically wrong residual nest (see the ``xfail`` reason), so this
    asserts the bug is present until the region-regen fix lands.
    """
    ir = build_canonical_ir()
    load0 = _block_for_op(ir, "NKILoad")
    leaf0 = next(d for d in ir.tree.preorder(load0) if isinstance(ir.tree.data(d), ISANode))
    ir = Split().apply(ir, SplitOption(target_nid=leaf0, factors=(16, 128), target_axis="d1"))
    mm = _block_for_op(ir, "NKIMatmul")
    m_loop = next(
        d for d in ir.tree.preorder(mm) if isinstance(ir.tree.data(d), ForNode) and ir.tree.data(d).loop_var == "i_d1_0"
    )
    ir = Split().apply(ir, SplitOption(target_nid=m_loop, factors=(4, 4)))
    mm = _block_for_op(ir, "NKIMatmul")
    m_outer = next(
        d for d in ir.tree.preorder(mm) if isinstance(ir.tree.data(d), ForNode) and ir.tree.data(d).loop_var == "i_d1_0"
    )
    load0 = _block_for_op(ir, "NKILoad")
    new_ir = ComputeAt().apply(ir, ComputeAtOption(block_nid=load0, target_loop_nid=m_outer, index=-2))
    assert_matches_hand(render(new_ir), KT.kernel_partial)


def test_oracle_rejects_genuinely_different_kernels():
    """The structural oracle must not equate kernels that differ semantically.

    Guards against over-normalization: an op-name change and a slice-width
    change each survive into the canonical AST, so the comparison fails.
    """
    base = inspect.getsource(KT.kernel_1)
    diff_op = base.replace("nc_matmul", "tensor_copy", 1)
    diff_width = base.replace("(i_d2_0) * 512 : (i_d2_0) * 512 + 512", "(i_d2_0) * 512 : (i_d2_0) * 512 + 256", 1)
    assert _normalize(base) != _normalize(diff_op)
    assert _normalize(base) != _normalize(diff_width)
