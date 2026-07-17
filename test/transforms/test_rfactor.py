"""Tests for nkigym.transforms.RFactor."""

from __future__ import annotations

import importlib.util
import os
import tempfile

import numpy as np
import pytest

from nkigym.codegen import render
from nkigym.ir.tree import ForNode, ISANode
from nkigym.ops.base import AxisRole
from nkigym.synthesis.simulate_nki import simulate_fp32
from nkigym.transforms import RFactor, RFactorOption, TransformLegalityError
from test.transforms._ladder_compare import assert_matches_hand
from test.transforms._rfactor_fixtures import ko_loop_nid, matmul_leaf_nid, mid_ladder_ir, split_k_ir

_HAND_KERNEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "kernel_library",
    "matmul",
    "lhsT_rhs",
    "kernel_rfactor_ko.py",
)


def _load_hand_kernel():
    """Path-load the expected post-RFactor hand kernel (kernel_library is not a package)."""
    spec = importlib.util.spec_from_file_location("kernel_rfactor_ko", _HAND_KERNEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.nki_f_matmul


def _rfactored_ir():
    """Canonical matmul → Split(K) → RFactor(ko). The post-RFactor IR under test."""
    ir = split_k_ir()
    return RFactor().apply(ir, RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0))


def test_fixture_splits_k() -> None:
    """split_k_ir yields a tree with an outer K loop (ko) of extent 2."""
    ir = split_k_ir()
    ko = ko_loop_nid(ir)
    node = ir.tree.data(ko)
    assert isinstance(node, ForNode)
    assert node.extent == 2


def test_analyze_finds_only_reduction_loops() -> None:
    """analyze offers only ForNodes binding the matmul's ACCUMULATION (K) axis."""
    ir = split_k_ir()
    opts = RFactor().analyze(ir)
    assert len(opts) >= 1
    for o in opts:
        node = ir.tree.data(o.target_loop_nid)
        assert isinstance(node, ForNode)
        assert node.loop_var.startswith("i_d0_")


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
        if isinstance(ir.tree.data(n), ForNode) and ir.tree.data(n).loop_var.startswith("i_d1_")
    )
    with pytest.raises(TransformLegalityError):
        RFactor().apply(ir, RFactorOption(target_loop_nid=m_loop, factor_axis=0))


@pytest.mark.skip(
    reason="RFactor byte-exact repro deferred to the RFactor-template redesign "
    "(2026-07-14 BufferCompaction spec, k33 out of scope). RFactor.apply is now "
    "structural-only, so its render carries the gadgets' pre-compaction offsets + "
    "wide psum shape; reproducing the compacted hand kernel needs the deferred "
    "template rewrite (+ the psum list_len 16->1 shrink). Structural assertions "
    "(role flip, gadget nesting, Dependency) stay green."
)
def test_apply_byte_exact() -> None:
    """render(Split→RFactor) is AST-identical to the hand kernel (fused two-stage:
    per-ko PSUM partial + tensor_tensor fold into sbuf_prod)."""
    assert_matches_hand(render(_rfactored_ir()), _load_hand_kernel())


def _sim_rendered_matmul(name: str, src: str) -> None:
    """Round-trip ``src`` through a temp module and assert its kernel sims equal to
    ``lhs_T.T @ rhs`` (fp32). ``simulate_fp32`` takes a NKI callable, not a KernelIR,
    so the rendered source is imported and its ``nki_f*`` kernel simmed. The rendered
    function name follows the source kernel (``nki_f_matmul`` for the matmul fixture,
    ``nki_f_nkigym`` for the k28 IR built from ``f_nkigym``), so it is discovered by
    prefix rather than hardcoded."""
    rng = np.random.default_rng(0)
    inputs = {
        "lhs_T": rng.standard_normal((2048, 2048)).astype(np.float32),
        "rhs": rng.standard_normal((2048, 2048)).astype(np.float32),
    }
    path = os.path.join(tempfile.gettempdir(), f"rfactor_sim_{name}.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(src)
    spec = importlib.util.spec_from_file_location(f"rfactor_sim_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = next(getattr(module, n) for n in dir(module) if n.startswith("nki_f"))
    actual = np.asarray(simulate_fp32(fn)(**inputs))
    np.testing.assert_allclose(actual, inputs["lhs_T"].T @ inputs["rhs"], atol=5e-3, rtol=5e-3)


def test_apply_sim_matches_matmul() -> None:
    """The rfactored kernel sims numerically equal to lhs_T.T @ rhs."""
    _sim_rendered_matmul("early_packed", render(_rfactored_ir()))


def test_apply_sim_matches_matmul_mid_tiled_m() -> None:
    """RFactor(ko) on the mid state (K split ko/ki AND M tiled i_d1_0 x i_d1_1,
    buffers still packed) sims equal to the golden — the tiled-M geometry, pinned
    as a regression test via the ``mid_ladder_ir`` fixture."""
    ir = mid_ladder_ir()
    rfactored = RFactor().apply(ir, RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0))
    _sim_rendered_matmul("mid_tiled_m", render(rfactored))


def test_ko_roles_split_across_blocks() -> None:
    """After RFactor, the K axis (d0) appears as PARALLEL in the matmul run-op block
    (each ko is an independent partial) and ACCUMULATION in the tensor_tensor fold block
    (carrying ko across the closing second-stage reduction) — one factored axis, two
    block roles."""
    ir = _rfactored_ir()
    roles = set()
    for nid in ir.tree.blocks():
        block = ir.tree.data(nid)
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
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKIMatmul"
    )
    ko = next(
        a
        for a in ir.tree.ancestors(matmul_leaf)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var.startswith("i_d0_")
    )
    psum_name = ir.tree.data(matmul_leaf).operand_bindings["dst"].tensor
    rf_memset = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode)
        and ir.tree.data(n).op_cls.NAME == "memset"
        and ir.tree.data(n).operand_bindings["dst"].tensor == psum_name
    )
    rf_drain = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode)
        and ir.tree.data(n).op_cls.NAME == "tensor_copy"
        and ir.tree.data(n).operand_bindings["src"].tensor == psum_name
    )
    assert ko in ir.tree.ancestors(rf_memset), "rf-init memset must be nested inside the ko loop"
    assert ko in ir.tree.ancestors(rf_drain), "rf-drain tensor_copy must be nested inside the ko loop"


@pytest.mark.skip(
    reason="RFactor byte-exact repro deferred to the RFactor-template redesign "
    "(2026-07-14 BufferCompaction spec, k33 out of scope). Pre-existing/stale at base "
    "(compares vs manual_transforms.kernel_29, now a Reorder rung after the manual "
    "renumber — RFactor is kernel_33); also RFactor.apply is now structural-only, so "
    "the compacted byte-exact form needs the deferred template rewrite + psum "
    "list_len 16->1 shrink. Structural assertions and the k28->k29 sim stay green."
)
def test_apply_byte_exact_k28_to_k29() -> None:
    """RFactor(ko) on the k28 IR (ki-innermost, all-list-buffer) renders byte-exact to
    the hand kernel_29 (fused two-stage, per-Mi-tile psum list-1 + sbuf_rfactor)."""
    from examples import manual_transforms
    from test.transforms._rfactor_fixtures import k28_ir, k28_ko_loop_nid

    ir = k28_ir()
    rfactored = RFactor().apply(ir, RFactorOption(target_loop_nid=k28_ko_loop_nid(ir), factor_axis=0))
    assert_matches_hand(render(rfactored), manual_transforms.kernel_29)


def test_apply_sim_matches_matmul_k28_to_k29() -> None:
    """The k28->k29 rfactored kernel sims numerically equal to lhs_T.T @ rhs."""
    from test.transforms._rfactor_fixtures import k28_ir, k28_ko_loop_nid

    ir = k28_ir()
    rfactored = RFactor().apply(ir, RFactorOption(target_loop_nid=k28_ko_loop_nid(ir), factor_axis=0))
    _sim_rendered_matmul("k28_to_k29", render(rfactored))
