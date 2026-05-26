"""Tests for :class:`nkigym.transforms.Reorder`."""

from __future__ import annotations

import copy
import importlib.util
import os
import shutil
import tempfile
from test.transforms._fixtures import build_canonical_ir
from test.transforms._seq_fixture import build_seq_ir

import numpy as np
import pytest

from nkigym.codegen import render
from nkigym.ir.tree import ForNode, ISANode
from nkigym.synthesis.simulate_nki import simulate_fp32
from nkigym.transforms import Reorder, ReorderOption, Split, SplitOption
from nkigym.transforms.base import TransformLegalityError


def test_reorder_imports_and_instantiates():
    """``Reorder`` and ``ReorderOption`` are public and constructable."""
    t = Reorder()
    opt = ReorderOption(outer_nid=0, inner_nid=1)
    assert opt.outer_nid == 0
    assert opt.inner_nid == 1
    assert isinstance(t.analyze(build_canonical_ir()), list)


def _find_first_for_with_trip(ir, trip: int) -> int:
    """Return the nid of the first :class:`ForNode` with the given trip count."""
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if isinstance(data, ForNode) and data.trip == trip:
            return nid
    raise AssertionError(f"no ForNode with trip={trip}")


def _find_matmul_outer_pair(ir) -> tuple[int, int]:
    """Return the (K-outer, M-middle) ForNode pair of the matmul nest.

    Canonical IR places the matmul at axis order K, M, N (outermost-first).
    The K-outer ForNode has dim 'd0', trip 16, and exactly one ForNode child
    (the M-middle, dim 'd1', trip 16). Walks pre-order looking for that pair.
    """
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if not (isinstance(data, ForNode) and data.dim == "d0" and data.trip == 16):
            continue
        kids = ir.tree.children(nid)
        if len(kids) != 1:
            continue
        kid_data = ir.tree.data(kids[0])
        if not (isinstance(kid_data, ForNode) and kid_data.dim == "d1" and kid_data.trip == 16):
            continue
        return nid, kids[0]
    raise AssertionError("matmul K/M outer ForNode pair not found")


def test_reorder_rejects_unknown_nid():
    """Both nids must exist in the tree."""
    ir = build_canonical_ir()
    outer, _ = _find_matmul_outer_pair(ir)
    with pytest.raises(TransformLegalityError):
        Reorder()._check_legality(ir, ReorderOption(outer_nid=outer, inner_nid=999_999))


def test_reorder_rejects_non_for_target():
    """``inner_nid`` must be a ``ForNode`` (passing an ``ISANode`` raises)."""
    ir = build_canonical_ir()
    outer, _ = _find_matmul_outer_pair(ir)
    isa_nid = next(nid for nid in ir.tree.preorder() if isinstance(ir.tree.data(nid), ISANode))
    with pytest.raises(TransformLegalityError):
        Reorder()._check_legality(ir, ReorderOption(outer_nid=outer, inner_nid=isa_nid))


def test_reorder_rejects_non_adjacent():
    """``inner_nid`` must be the sole child of ``outer_nid``."""
    ir = build_canonical_ir()
    outer, middle = _find_matmul_outer_pair(ir)
    """The N-innermost ForNode lives under ``middle``, not directly under ``outer``."""
    n_inner = ir.tree.children(middle)[0]
    with pytest.raises(TransformLegalityError):
        Reorder()._check_legality(ir, ReorderOption(outer_nid=outer, inner_nid=n_inner))


def test_reorder_rejects_outer_with_siblings():
    """``outer_nid`` must have exactly one child (single-element child list)."""
    ir = build_canonical_ir()
    """The matmul middle (M) ForNode has the matmul-N inner as its sole child,
    but the matmul-N inner has the ISA leaf as its sole child. We need a ForNode
    with multiple children to test rejection. Apply Split to introduce such a pair?
    Easier: synthesize the failure by constructing an option pointing at a ForNode
    that does have multiple children in the canonical IR.

    The canonical IR places several blocks (load, load, memset, matmul, copy, store)
    as children of the root, not of any ForNode. So no ForNode has siblings in
    canonical. We construct the test by first applying a Split to create the
    multi-child structure, then attempting Reorder on it."""
    target = _find_first_for_with_trip(ir, 16)
    split_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(4, 4)))
    """``split_ir``'s split chain still has single children. Manufacture
    a multi-child ForNode by direct graph mutation in a copy."""
    bad_ir = copy.deepcopy(split_ir)
    """Find any ForNode in bad_ir; add a synthetic ForNode child alongside its real child."""
    for nid in bad_ir.tree.preorder():
        data = bad_ir.tree.data(nid)
        if isinstance(data, ForNode):
            outer_n = nid
            existing_kids = bad_ir.tree.children(outer_n)
            if not existing_kids:
                continue
            inner_n = existing_kids[0]
            """Add a synthetic sibling under outer_n."""
            bad_ir.tree.add_node(ForNode(dim=data.dim, trip=2), parent=outer_n)
            with pytest.raises(TransformLegalityError):
                Reorder()._check_legality(bad_ir, ReorderOption(outer_nid=outer_n, inner_nid=inner_n))
            return
    raise AssertionError("no ForNode found for sibling-injection test")


def test_reorder_apply_swaps_payloads():
    """Applying ``Reorder`` swaps the two ForNode payloads; topology unchanged."""
    ir = build_canonical_ir()
    outer, inner = _find_matmul_outer_pair(ir)
    parent = ir.tree.parent(outer)
    inner_kids_before = ir.tree.children(inner)
    parent_kids_before = ir.tree.children(parent)

    new_ir = Reorder().apply(ir, ReorderOption(outer_nid=outer, inner_nid=inner))

    """outer_nid now carries inner's payload, inner_nid now carries outer's payload."""
    assert new_ir.tree.data(outer).dim == "d1"
    assert new_ir.tree.data(outer).trip == 16
    assert new_ir.tree.data(inner).dim == "d0"
    assert new_ir.tree.data(inner).trip == 16

    """Topology is bitwise unchanged."""
    assert new_ir.tree.children(parent) == parent_kids_before
    assert new_ir.tree.children(outer) == [inner]
    assert new_ir.tree.children(inner) == inner_kids_before


def test_reorder_apply_preserves_input_ir():
    """``apply`` must not mutate its input IR."""
    ir = build_canonical_ir()
    outer, inner = _find_matmul_outer_pair(ir)
    snapshot_payloads = {nid: ir.tree.data(nid) for nid in ir.tree.preorder()}

    Reorder().apply(ir, ReorderOption(outer_nid=outer, inner_nid=inner))

    for nid, data in snapshot_payloads.items():
        assert ir.tree.data(nid) == data, f"payload at nid={nid} mutated"


def test_reorder_self_inverse():
    """Applying the same ``Reorder`` twice restores the original payloads."""
    ir = build_canonical_ir()
    outer, inner = _find_matmul_outer_pair(ir)
    option = ReorderOption(outer_nid=outer, inner_nid=inner)

    once = Reorder().apply(ir, option)
    twice = Reorder().apply(once, option)

    """Twice-applied IR has the same payload at every nid as the original."""
    original_nids = list(ir.tree.preorder())
    twice_nids = list(twice.tree.preorder())
    assert original_nids == twice_nids
    for nid in original_nids:
        assert ir.tree.data(nid) == twice.tree.data(nid)


def test_reorder_rejects_sequential_role():
    """A leaf with SEQUENTIAL role on either swap dim must reject the reorder."""
    ir, outer, inner, _ = build_seq_ir()
    with pytest.raises(TransformLegalityError, match="SEQUENTIAL"):
        Reorder()._check_legality(ir, ReorderOption(outer_nid=outer, inner_nid=inner))


def test_reorder_allows_accumulation_parallel():
    """K/M (ACCUM/PARALLEL) and M/N (PARALLEL/PARALLEL) must pass legality."""
    ir = build_canonical_ir()
    """K-outer / M-middle is ACCUM (K) and PARALLEL (M)."""
    outer_km, inner_km = _find_matmul_outer_pair(ir)
    Reorder()._check_legality(ir, ReorderOption(outer_nid=outer_km, inner_nid=inner_km))

    """M-middle / N-inner is PARALLEL (M) and PARALLEL (N)."""
    n_inner = ir.tree.children(inner_km)[0]
    Reorder()._check_legality(ir, ReorderOption(outer_nid=inner_km, inner_nid=n_inner))


def test_reorder_analyze_canonical_matmul():
    """Canonical matmul nest yields at least the K/M and M/N adjacent ForNode pairs."""
    ir = build_canonical_ir()
    options = Reorder().analyze(ir)
    pairs = {(opt.outer_nid, opt.inner_nid) for opt in options}

    """K/M outer pair must be present."""
    km_outer, km_inner = _find_matmul_outer_pair(ir)
    assert (km_outer, km_inner) in pairs

    """M/N pair must also be present (M-middle has the matmul N-inner as sole child)."""
    n_inner = ir.tree.children(km_inner)[0]
    assert (km_inner, n_inner) in pairs


def test_reorder_analyze_skips_single_for_subtrees():
    """Subtrees whose only ForNode has no ForNode child (loads, stores, memset)
    must not surface options."""
    ir = build_canonical_ir()
    options = Reorder().analyze(ir)
    pairs = {(opt.outer_nid, opt.inner_nid) for opt in options}

    """The lhs_T load nest is a single ForNode → ISANode chain (after the
    canonical IR builds a per-axis chain, but with trip-1 ForNodes on
    axes whose MAX_TILE_SIZE is None, see `tree.py:_attach_op_subtree`).
    Walk every ForNode; ensure no option pairs a ForNode with an ISA child."""
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if not isinstance(data, ForNode):
            continue
        kids = ir.tree.children(nid)
        if len(kids) == 1 and isinstance(ir.tree.data(kids[0]), ISANode):
            assert (nid, kids[0]) not in pairs


def test_reorder_analyze_returns_only_legal_options():
    """Every option ``analyze`` returns must apply without raising."""
    ir = build_canonical_ir()
    options = Reorder().analyze(ir)
    assert options, "expected at least one Reorder option on the canonical IR"
    for opt in options:
        Reorder().apply(ir, opt)


def test_reorder_render_swap_visible():
    """Rendering the Reorder-applied IR shows ``i_d1_0`` outside ``i_d0_0`` in the matmul nest."""
    ir = build_canonical_ir()
    outer, inner = _find_matmul_outer_pair(ir)
    new_ir = Reorder().apply(ir, ReorderOption(outer_nid=outer, inner_nid=inner))
    src = render(new_ir)

    """The matmul nest in the rendered source must put i_d1_0 (M, was inner)
    above i_d0_0 (K, was outer)."""
    matmul_pos = src.find("nisa.nc_matmul")
    assert matmul_pos != -1
    """Walk backward from the matmul call; the immediate enclosing loop on the
    matmul nest is now i_d1_0 (M-axis), and the next enclosing one is i_d0_0 (K-axis).
    Find the K loop offset and the M loop offset; M must come BEFORE K (smaller offset)."""
    k_pos = src.rfind("for i_d0_0 in range(16):", 0, matmul_pos)
    m_pos = src.rfind("for i_d1_0 in range(16):", 0, matmul_pos)
    assert k_pos != -1 and m_pos != -1, f"K loop at {k_pos}, M loop at {m_pos}"
    assert m_pos < k_pos, f"after Reorder, M loop ({m_pos}) must precede K loop ({k_pos}) above matmul"


def test_reorder_round_trip_render_sim():
    """End-to-end: Reorder → render → fp32 sim → matches numpy golden."""
    ir = build_canonical_ir()
    outer, inner = _find_matmul_outer_pair(ir)
    new_ir = Reorder().apply(ir, ReorderOption(outer_nid=outer, inner_nid=inner))
    src = render(new_ir)

    """Write the rendered kernel to a temp path and import it as a module."""
    tmpdir = tempfile.mkdtemp()
    try:
        kernel_path = os.path.join(tmpdir, "kernel.py")
        with open(kernel_path, "w") as f:
            f.write(src)
        spec = importlib.util.spec_from_file_location("dumped_reorder_kernel", kernel_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        K, M, N = 2048, 2048, 2048
        rng = np.random.default_rng(0)
        lhs_T = rng.standard_normal((K, M)).astype(np.float32)
        rhs = rng.standard_normal((K, N)).astype(np.float32)
        expected = lhs_T.T @ rhs
        actual = np.asarray(simulate_fp32(module.nki_f_matmul)(lhs_T, rhs))
        np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
