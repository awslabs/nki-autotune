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
