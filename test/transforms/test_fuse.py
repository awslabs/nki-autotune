"""Tests for nkigym.transforms.Fuse under BlockNode IR."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms import Fuse, FuseOption, Split, SplitOption


def _matmul_block_first_for(ir):
    from nkigym.ops.matmul import NKIMatmul

    for nid in ir.tree.blocks():
        for d in ir.tree.descendants(nid):
            if isinstance(ir.tree.data(d), ISANode) and ir.tree.data(d).op_cls is NKIMatmul:
                """First ForNode on the path from block to leaf."""
                for path_nid in ir.tree.preorder(nid):
                    if isinstance(ir.tree.data(path_nid), ForNode):
                        return path_nid
    raise AssertionError


def test_fuse_outer_trip_inverts_split():
    """Split then Fuse on the same axis returns the original ForNode extent."""
    ir = build_canonical_ir()
    target = _matmul_block_first_for(ir)
    original_extent = ir.tree.data(target).extent

    split_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(2, original_extent // 2)))
    """Locate the new outer ForNode."""
    parent = split_ir.tree.parent(target) if target in split_ir.tree.graph else None
    if parent is None:
        """Target was removed; pick the new top from same parent slot in original IR."""
        original_parent = ir.tree.parent(target)
        new_top = split_ir.tree.children(original_parent)[0]
    else:
        new_top = parent
    inner = split_ir.tree.children(new_top)[0]
    fuse_ir = Fuse().apply(split_ir, FuseOption(target_nids=(new_top, inner), target_axis=None))

    """The fused ForNode now has the original extent."""
    fused_parent = ir.tree.parent(target)
    fused_top = fuse_ir.tree.children(fused_parent)[0]
    fused_data = fuse_ir.tree.data(fused_top)
    assert isinstance(fused_data, ForNode)
    assert fused_data.extent == original_extent


def test_fuse_tensorize_matmul_n_renders_and_sims(tmp_path):
    """Tensorize-Fuse the matmul's innermost N loop (i_d2_0, 4 trips) back into the tile
    (512 -> 2048): renders + sims correctly. (Topology-only assertion was insufficient —
    it never caught that the tile width stayed 512.)"""
    import importlib.util
    from test.transforms._fixtures import INPUT_SPECS, build_canonical_ir

    import numpy as np

    from nkigym.codegen import render
    from nkigym.ir.expr import Var
    from nkigym.ir.tree import BlockNode, ISANode
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.synthesis.simulate_nki import simulate_fp32
    from nkigym.transforms import Fuse, FuseOption

    ir = build_canonical_ir()
    leaf_nid = next(
        n for n in ir.tree.preorder() if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls is NKIMatmul
    )
    parent_for = ir.tree.parent(leaf_nid)  # i_d2_0, the innermost matmul loop
    mb = next(
        ir.tree.data(a)
        for a in reversed(ir.tree.ancestors(leaf_nid))
        if isinstance(ir.tree.data(a), BlockNode) and ir.tree.data(a).iter_vars
    )
    target_axis = next(
        iv.axis
        for iv, v in zip(mb.iter_vars, mb.iter_values)
        if isinstance(v, Var) and v.name == ir.tree.data(parent_for).loop_var
    )
    fused = Fuse().apply(ir, FuseOption(target_nids=(parent_for, leaf_nid), target_axis=target_axis))
    src = render(fused)
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


def test_split_then_fuse_tensorize_round_trips(tmp_path):
    """Tensorize-Split the load d1 (2048->16x128) then tensorize-Fuse it back == original;
    both intermediate and final render + sim correctly."""
    import importlib.util
    from test.transforms._fixtures import INPUT_SPECS, build_canonical_ir

    import numpy as np

    from nkigym.codegen import render
    from nkigym.ir.tree import ForNode, ISANode
    from nkigym.synthesis.simulate_nki import simulate_fp32
    from nkigym.transforms import Fuse, FuseOption, Split, SplitOption

    ir = build_canonical_ir()
    load_leaf = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKILoad"
    )
    split_ir = Split().apply(ir, SplitOption(target_nid=load_leaf, factors=(16, 128), target_axis="d1"))
    """The load now has a new inner ForNode above the leaf; fuse it back."""
    new_leaf = next(
        n
        for n in split_ir.tree.preorder()
        if isinstance(split_ir.tree.data(n), ISANode) and split_ir.tree.data(n).op_cls.__name__ == "NKILoad"
    )
    inner_for = split_ir.tree.parent(new_leaf)
    assert isinstance(split_ir.tree.data(inner_for), ForNode)
    fused_ir = Fuse().apply(split_ir, FuseOption(target_nids=(inner_for, new_leaf), target_axis="d1"))
    src = render(fused_ir)
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


def test_fuse_merge_trips_dense_name():
    """Fuse two same-dim trip loops -> one loop named densely (i_d1_0), not i_d1_fused."""
    from test.transforms._fixtures import build_canonical_ir

    from nkigym.ir.tree import ForNode
    from nkigym.transforms import Fuse, FuseOption, Split, SplitOption

    ir = build_canonical_ir()
    d1 = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ForNode)
        and ir.tree.data(n).loop_var == "i_d1_0"
        and ir.tree.data(n).extent == 16
    )
    ir = Split().apply(ir, SplitOption(target_nid=d1, factors=(2, 8)))
    """Now d1 has i_d1_0(2), i_d1_1(8); fuse them back."""
    outer = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ForNode) and ir.tree.data(n).loop_var == "i_d1_0" and ir.tree.data(n).extent == 2
    )
    inner = next(c for c in ir.tree.children(outer) if isinstance(ir.tree.data(c), ForNode))
    fused = Fuse().apply(ir, FuseOption(target_nids=(outer, inner), target_axis=None))
    names = [
        fused.tree.data(n).loop_var
        for n in fused.tree.preorder()
        if isinstance(fused.tree.data(n), ForNode) and fused.tree.data(n).loop_var.startswith("i_d1")
    ]
    assert "i_d1_0" in names and not any("fused" in nm for nm in names), names


def test_split_then_fuse_round_trip_byteexact():
    """Split the load d1 2048->(16,128) then fuse back == the original trip-1-free k0 load
    (loopless d1, full 2048 width). Byte-exact."""
    from test.transforms._fixtures import build_canonical_ir

    from nkigym.codegen import render
    from nkigym.ir.tree import ForNode, ISANode
    from nkigym.transforms import Fuse, FuseOption, Split, SplitOption

    ir = build_canonical_ir()
    canonical_render = render(ir)
    load = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKILoad"
    )
    split_ir = Split().apply(ir, SplitOption(target_nid=load, factors=(16, 128), target_axis="d1"))
    new_load = next(
        n
        for n in split_ir.tree.preorder()
        if isinstance(split_ir.tree.data(n), ISANode) and split_ir.tree.data(n).op_cls.__name__ == "NKILoad"
    )
    d1_loop = split_ir.tree.parent(new_load)
    assert isinstance(split_ir.tree.data(d1_loop), ForNode) and split_ir.tree.data(d1_loop).extent == 16
    fused_ir = Fuse().apply(split_ir, FuseOption(target_nids=(d1_loop, new_load), target_axis="d1"))
    assert render(fused_ir) == canonical_render, "Split->Fuse did not round-trip to canonical"
