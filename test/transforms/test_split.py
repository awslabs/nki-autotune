"""Tests for nkigym.transforms.Split under BlockNode IR."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

import pytest

from nkigym.ir.arith.expr import to_affine
from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms import Split, SplitOption, TransformLegalityError


def _matmul_block(ir):
    """Return (block_nid, block) for the matmul leaf-block."""
    from nkigym.ops.matmul import NKIMatmul

    for nid in ir.tree.blocks():
        block = ir.tree.data(nid)
        """Find blocks whose direct children (not deep descendants) include an ISANode with NKIMatmul."""
        for child_nid in ir.tree.children(nid):
            if isinstance(ir.tree.data(child_nid), ForNode):
                """Walk down from ForNode to find ISANode."""
                for desc in ir.tree.descendants(child_nid):
                    if isinstance(ir.tree.data(desc), ISANode) and ir.tree.data(desc).op_cls is NKIMatmul:
                        return nid, block
        """Direct ISANode child."""
        for child_nid in ir.tree.children(nid):
            if isinstance(ir.tree.data(child_nid), ISANode) and ir.tree.data(child_nid).op_cls is NKIMatmul:
                return nid, block
    raise AssertionError("matmul block not found")


def _first_for_under(ir, block_nid):
    """Return the first ForNode descended from block_nid."""
    for nid in ir.tree.preorder(block_nid):
        if isinstance(ir.tree.data(nid), ForNode):
            return nid
    raise AssertionError("no ForNode under block")


def test_split_outer_trip_replaces_for_with_chain():
    """Splitting a ForNode trip 16 by factors=(4, 4) gives a 4 -> 4 chain."""
    ir = build_canonical_ir()
    matmul_block_nid, _ = _matmul_block(ir)
    target = _first_for_under(ir, matmul_block_nid)
    target_extent = ir.tree.data(target).extent

    new_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(4, target_extent // 4)))

    """Old IR untouched."""
    assert ir.tree.data(target).extent == target_extent

    """New IR: parent's child slot now contains a fresh ForNode of extent 4 with one ForNode child."""
    parent = ir.tree.parent(target)
    new_kid = new_ir.tree.children(parent)[0]
    new_kid_data = new_ir.tree.data(new_kid)
    assert isinstance(new_kid_data, ForNode)
    assert new_kid_data.extent == 4
    inner = new_ir.tree.children(new_kid)[0]
    assert isinstance(new_ir.tree.data(inner), ForNode)
    assert new_ir.tree.data(inner).extent == target_extent // 4


def test_split_outer_trip_rewrites_iter_value_for_bound_axis():
    """The enclosing block's iter_value for the split iter_var becomes a sum of new loop_vars * strides."""
    ir = build_canonical_ir()
    matmul_block_nid, matmul_block = _matmul_block(ir)
    target = _first_for_under(ir, matmul_block_nid)
    target_for = ir.tree.data(target)
    target_loop_var = target_for.loop_var
    target_extent = target_for.extent

    """Identify which iter_var was bound by the original loop_var."""
    bound_axis_index = None
    for i, value in enumerate(matmul_block.iter_values):
        from nkigym.ir.arith.expr import Var

        if isinstance(value, Var) and value.name == target_loop_var:
            bound_axis_index = i
            break
    assert bound_axis_index is not None, "could not locate the iter_value bound by the target ForNode"

    new_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(4, target_extent // 4)))
    new_block = new_ir.tree.data(matmul_block_nid)
    new_value = new_block.iter_values[bound_axis_index]
    coeffs = to_affine(new_value)
    """The new value is a 2-term affine combination summing two loop_vars."""
    var_terms = {k: v for k, v in coeffs.items() if k is not None}
    assert len(var_terms) == 2
    """Coefficients match outer * inner_extent + inner."""
    assert sorted(var_terms.values()) == [1, target_extent // 4]


def test_split_apply_preserves_input_ir():
    """``apply`` must not mutate its input IR."""
    ir = build_canonical_ir()
    matmul_block_nid, _ = _matmul_block(ir)
    target = _first_for_under(ir, matmul_block_nid)
    target_extent = ir.tree.data(target).extent
    snapshot_num_nodes = ir.tree.num_nodes
    Split().apply(ir, SplitOption(target_nid=target, factors=(4, target_extent // 4)))
    assert ir.tree.num_nodes == snapshot_num_nodes


def test_split_rejects_factor_product_mismatch():
    ir = build_canonical_ir()
    matmul_block_nid, _ = _matmul_block(ir)
    target = _first_for_under(ir, matmul_block_nid)
    with pytest.raises(TransformLegalityError):
        Split().apply(ir, SplitOption(target_nid=target, factors=(3, 5)))


def test_split_analyze_offers_tensorize_on_load():
    """Split.analyze must offer a tensorize-flavor (target_axis set) option on the load leaf,
    whose d1 free-axis tile is width-2048 and factorizable to 16x128. Regression: the
    concrete(d1)-vs-abstract(F) axis-name mismatch made _current_tensorize_width return None."""
    from test.transforms._fixtures import build_canonical_ir

    from nkigym.ir.tree import ISANode
    from nkigym.transforms import Split

    ir = build_canonical_ir()
    load_leaf = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKILoad"
    )
    tensorize_opts = [o for o in Split().analyze(ir) if o.target_nid == load_leaf and o.target_axis is not None]
    assert tensorize_opts, "Split.analyze offered no tensorize option on the load leaf"
    """The d1 (concrete) free axis has width 2048 → factorizations include (16, 128)."""
    assert any(o.factors == (16, 128) for o in tensorize_opts), [o.factors for o in tensorize_opts]


def test_split_tensorize_load_d1_to_16x128(tmp_path):
    """Tensorize-Split the load's d1 free-axis tile 2048 -> (16, 128): the load's dst tile
    shrinks to 128, gains a 16-trip loop, and the kernel renders + sims correctly."""
    import importlib.util
    from test.transforms._fixtures import INPUT_SPECS, build_canonical_ir

    import numpy as np

    from nkigym.codegen import render
    from nkigym.ir.arith.expr import Const
    from nkigym.ir.tree import ISANode
    from nkigym.synthesis.simulate_nki import simulate_fp32
    from nkigym.transforms import Split, SplitOption

    ir = build_canonical_ir()
    load_leaf = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKILoad"
    )
    new_ir = Split().apply(ir, SplitOption(target_nid=load_leaf, factors=(16, 128), target_axis="d1"))
    new_leaf = next(
        n
        for n in new_ir.tree.preorder()
        if isinstance(new_ir.tree.data(n), ISANode) and new_ir.tree.data(n).op_cls.__name__ == "NKILoad"
    )
    dst = new_ir.tree.data(new_leaf).operand_bindings["dst"]
    assert any(isinstance(w, Const) and w.value == 128 for _lo, w in dst.ranges), dst.ranges
    src = render(new_ir)
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(src)
    rng = np.random.default_rng(0)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dt) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    spec = importlib.util.spec_from_file_location("k", kernel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    actual = np.asarray(simulate_fp32(mod.nki_f_matmul)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize(
    "op_name, which, axis, factors",
    [
        ("NKILoad", 1, "d2", (4, 512)),
        ("NKIMemset", 0, "d2", (4, 512)),
        ("NKITensorCopy", 0, "d2", (4, 512)),
        ("NKIStore", 0, "d2", (4, 512)),
    ],
)
def test_split_tensorize_ladder_ops_render_and_sim(tmp_path, op_name, which, axis, factors):
    """Each tensorize-Split in the kernel_transforms ladder renders + sims correctly."""
    import importlib.util
    from test.transforms._fixtures import INPUT_SPECS, build_canonical_ir

    import numpy as np

    from nkigym.codegen import render
    from nkigym.ir.tree import ISANode
    from nkigym.synthesis.simulate_nki import simulate_fp32
    from nkigym.transforms import Split, SplitOption

    ir = build_canonical_ir()
    leaves = [
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == op_name
    ]
    new_ir = Split().apply(ir, SplitOption(target_nid=leaves[which], factors=factors, target_axis=axis))
    src = render(new_ir)
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(src)
    rng = np.random.default_rng(0)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dt) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    spec = importlib.util.spec_from_file_location("k", kernel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    actual = np.asarray(simulate_fp32(mod.nki_f_matmul)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_split_tensorize_below_min_tile_rejected():
    """A tensorize-split whose innermost factor < the axis MIN_TILE_SIZE is illegal.

    The matmul M axis (d1) is the PSUM partition axis at tile 128 with
    MIN_TILE_SIZE 128, so factors=(16,8) (final tile 8) must be rejected by
    apply's legality check — it would otherwise render a sub-128 partition tile.
    """
    ir = build_canonical_ir()
    mm = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKIMatmul"
    )
    with pytest.raises(TransformLegalityError, match="MIN_TILE_SIZE"):
        Split().apply(ir, SplitOption(target_nid=mm, factors=(16, 8), target_axis="d1"))


def test_split_analyze_omits_below_min_tensorize_splits():
    """analyze never offers a tensorize-split whose final factor < the axis MIN_TILE_SIZE.

    The matmul K (d0) and M (d1) axes are at tile 128 = MIN, so they admit no
    tensorize-split at all; N (d2) at tile 512 admits only finals >= 128
    (i.e. (4,128); (2,256); (2,2,128)). No option may shrink a partition/
    contraction tile below the hardware floor.
    """
    ir = build_canonical_ir()
    mm = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKIMatmul"
    )
    block = next(
        ir.tree.data(a) for a in reversed(ir.tree.ancestors(mm)) if ir.tree.data(a).__class__.__name__ == "BlockNode"
    )
    inverse = {concrete: abstract for abstract, concrete in block.axis_map.items()}
    min_tile = ir.tree.data(mm).op_cls.MIN_TILE_SIZE
    tensorize = [o for o in Split().analyze(ir) if o.target_nid == mm and o.target_axis is not None]
    for opt in tensorize:
        floor = min_tile[inverse[opt.target_axis]]
        assert opt.factors[-1] >= floor, f"analyze offered {opt.factors} on {opt.target_axis} below floor {floor}"
    offered_axes = {o.target_axis for o in tensorize}
    assert "d0" not in offered_axes and "d1" not in offered_axes
    assert ("d2", (4, 128)) in {(o.target_axis, o.factors) for o in tensorize}


def test_split_tensorize_n_to_128_still_legal(tmp_path):
    """A tensorize-split whose final factor == MIN_TILE_SIZE stays legal and renders.

    Matmul N (d2) tile 512 -> (4,128): final 128 == MIN, so it is offered,
    applies, and renders + sims correctly. Guards against an off-by-one that
    would reject the boundary case.
    """
    import importlib.util
    from test.transforms._fixtures import INPUT_SPECS

    import numpy as np

    from nkigym.codegen import render
    from nkigym.synthesis.simulate_nki import simulate_fp32

    ir = build_canonical_ir()
    mm = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKIMatmul"
    )
    new_ir = Split().apply(ir, SplitOption(target_nid=mm, factors=(4, 128), target_axis="d2"))
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(render(new_ir))
    rng = np.random.default_rng(0)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dt) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    spec = importlib.util.spec_from_file_location("k", kernel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    actual = np.asarray(simulate_fp32(mod.nki_f_matmul)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_split_load_d1_matches_hand_k1_byteexact(tmp_path):
    """k0->k1: split the load's d1 tensorize 2048->(16,128). The rendered load loop must
    be exactly `for i_d1_0 in range(16): dma_copy(... i_d1_0*128 : +128)` — single dense
    loop, no i_d1_0_0, no trip-1 wrapper."""
    from test.transforms._fixtures import build_canonical_ir

    from nkigym.codegen import render
    from nkigym.ir.tree import ISANode
    from nkigym.transforms import Split, SplitOption

    ir = build_canonical_ir()
    load = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKILoad"
    )
    new_ir = Split().apply(ir, SplitOption(target_nid=load, factors=(16, 128), target_axis="d1"))
    lines = [l.strip() for l in render(new_ir).splitlines()]
    """The lhs_T load nest: a d0 loop, a d1 loop trip-16, then the dma_copy."""
    assert "for i_d1_0 in range(16):" in lines
    assert not any("i_d1_0_0" in l for l in lines), "double-suffix name leaked"
    assert not any("range(1)" in l for l in lines), "trip-1 wrapper leaked"
    load_line = next(l for l in lines if "dst=sbuf_lhs_T" in l)
    """Index must be i_d1_0*128 : +128 (no i_d1_0 trip-1 term, no i_d1_0_0)."""
    assert "i_d1_0 * 128" in load_line and "+ 128" in load_line, load_line
    assert "* 2048" not in load_line, load_line


def test_split_trip_dense_names():
    """Split a trip-16 matmul d1 loop -> (2,8): the two loops are i_d1_0, i_d1_1 (dense)."""
    from test.transforms._fixtures import build_canonical_ir

    from nkigym.ir.tree import ForNode
    from nkigym.transforms import Split, SplitOption

    ir = build_canonical_ir()
    d1 = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ForNode)
        and ir.tree.data(n).loop_var == "i_d1_0"
        and ir.tree.data(n).extent == 16
    )
    new_ir = Split().apply(ir, SplitOption(target_nid=d1, factors=(2, 8)))
    names = [
        new_ir.tree.data(n).loop_var
        for n in new_ir.tree.preorder()
        if isinstance(new_ir.tree.data(n), ForNode) and new_ir.tree.data(n).loop_var.startswith("i_d1_")
    ]
    assert "i_d1_0" in names and "i_d1_1" in names
    assert not any("_0_" in nm for nm in names), names


def test_split_matches_tvm_structure():
    """Layer-B guard: our outer-trip Split's loop nest matches TVM's own ``schedule.split``.

    Splits the canonical matmul's ``i_d1_0`` loop (extent 16) by ``(4, 4)`` and
    confronts the resulting d1-loop extents AND the recovered iter_var binding
    against TVM's own TensorIR ``schedule.split`` on an equivalent extent-16 loop
    (the Layer-B structural oracle). TVM's ``substitute_value`` is
    ``Σ var_i · Π(factor_j, j>i)`` = ``i0*4 + i1``; our :func:`normalize_block`
    must reproduce the same outer->inner extents ``[4, 4]`` and the same binding.
    This is a regression guard on structural fidelity, orthogonal to the
    byte-exact ladder gate.
    """
    pytest.importorskip("tvm")
    from test.transforms._oracle_helpers import enclosing_for_nids
    from test.transforms._tvm_struct_oracle import tvm_split_loopnest

    from nkigym.ir.arith.expr import Var, format_expr, substitute

    ir = build_canonical_ir()
    mm = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKIMatmul"
    )
    mloop = next(
        a
        for a in ir.tree.ancestors(mm)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == "i_d1_0"
    )
    extent = ir.tree.data(mloop).extent
    factors = (4, 4)
    assert extent == 16 and factors[0] * factors[1] == extent

    out = Split().apply(ir, SplitOption(target_nid=mloop, factors=factors, target_axis=None))
    nest = tvm_split_loopnest(extent=extent, factors=list(factors))

    """Restrict to the d1 loops ENCLOSING the matmul leaf — the block we split.
    Sibling blocks (loads, tensor_copy, store) each carry their own block-local
    ``i_d1_0`` loop, so a full-tree preorder would over-collect them."""
    out_mm = next(
        n
        for n in out.tree.preorder()
        if isinstance(out.tree.data(n), ISANode) and out.tree.data(n).op_cls.__name__ == "NKIMatmul"
    )
    d1 = enclosing_for_nids(out, out_mm, "i_d1")
    our_extents = [out.tree.data(n).extent for n in d1]
    assert our_extents == nest.extents == [4, 4]

    """The d1 iter_var binding our normalize recomputed must equal TVM's substitute_value
    (loop vars renamed positionally outer->inner): i_d1_0*4 + i_d1_1 == i0*4 + i1."""
    out_block = next(
        out.tree.data(a) for a in reversed(out.tree.ancestors(mm)) if out.tree.data(a).__class__.__name__ == "BlockNode"
    )
    d1_value = next(v for iv, v in zip(out_block.iter_vars, out_block.iter_values) if iv.axis == "d1")
    loop_vars = [out.tree.data(n).loop_var for n in d1]
    positional = {lv: f"i{idx}" for idx, lv in enumerate(loop_vars)}
    renamed = substitute(d1_value, {lv: Var(name=positional[lv]) for lv in loop_vars})
    assert format_expr(renamed).replace(" * ", "*") == nest.binding


def test_tensorize_split_matches_tvm_structure():
    """Layer-B guard: our tensorize Split's loop nest matches TVM's own ``schedule.split``.

    A tensorize-split of an op axis IS a loop-split structurally: it inserts
    ``factors[:-1]`` outer loops and SETS the access tile width to
    ``factors[-1]`` (the innermost factor becomes the tile width, not a loop).
    We split the canonical load's ``d1`` free axis (extent 2048) by
    ``(16, 128)`` and confront the result against TVM's own ``schedule.split``
    on an equivalent extent-2048 loop (the structural oracle).

    Correspondence: ``tvm_split_loopnest(extent=2048, factors=[16, 128])``
    returns ``extents == [16, 128]`` (outer -> inner). Our tensorize-split
    materializes the OUTER factors ``extents[:-1] == [16]`` as enclosing loops
    over the leaf and the INNERMOST factor ``extents[-1] == 128`` as the access
    tile width (no loop). So our single ``i_d1`` enclosing-loop extent ``[16]``
    must equal TVM's outer extents and our tile width ``128`` must equal TVM's
    innermost extent. Orthogonal to the byte-exact ladder gate.
    """
    pytest.importorskip("tvm")
    from test.transforms._oracle_helpers import enclosing_for_nids
    from test.transforms._tvm_struct_oracle import tvm_split_loopnest

    from nkigym.ir.arith.expr import Const

    ir = build_canonical_ir()
    load = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode)
        and ir.tree.data(n).op_cls.__name__ == "NKILoad"
        and ir.tree.data(n).operand_bindings["src"].tensor == "lhs_T"
    )
    extent = 2048
    factors = (16, 128)
    assert factors[0] * factors[1] == extent

    out = Split().apply(ir, SplitOption(target_nid=load, factors=factors, target_axis="d1"))
    nest = tvm_split_loopnest(extent=extent, factors=list(factors))
    assert nest.extents == [16, 128]

    """Restrict to the d1 loops ENCLOSING the split load leaf — sibling blocks each
    carry their own block-local d1 loop, so a full-tree preorder would over-collect."""
    out_load = next(
        n
        for n in out.tree.preorder()
        if isinstance(out.tree.data(n), ISANode)
        and out.tree.data(n).op_cls.__name__ == "NKILoad"
        and out.tree.data(n).operand_bindings["src"].tensor == "lhs_T"
    )
    d1 = enclosing_for_nids(out, out_load, "i_d1")
    our_loop_extents = [out.tree.data(n).extent for n in d1]

    """The OUTER factors (factors[:-1]) become loops; the INNERMOST factor is the tile width."""
    assert our_loop_extents == nest.extents[:-1] == [16]

    """The access tile width on the split d1 axis equals TVM's innermost extent (128).
    d1 maps (via axis_map) to the load's abstract free axis 'F'; the dst region's F-axis
    width must be the innermost factor, NOT a loop."""
    leaf = out.tree.data(out_load)
    block = next(
        out.tree.data(a)
        for a in reversed(out.tree.ancestors(out_load))
        if out.tree.data(a).__class__.__name__ == "BlockNode"
    )
    inverse_axis_map = {concrete: abstract for abstract, concrete in block.axis_map.items()}
    f_axis = inverse_axis_map["d1"]
    dst_axes = leaf.op_cls.OPERAND_AXES["dst"]
    f_index = dst_axes.index(f_axis)
    _lo, width = leaf.operand_bindings["dst"].ranges[f_index]
    assert isinstance(width, Const) and width.value == nest.extents[-1] == 128
