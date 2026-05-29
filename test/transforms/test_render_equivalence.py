"""Render-equivalence regression: applied transforms produce working NKI source."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

from nkigym.codegen import render
from nkigym.transforms import Fuse, FuseOption, Reorder, ReorderOption, Split, SplitOption


def test_split_outer_trip_renders_and_passes_numerics(tmp_path):
    """After one outer-trip Split, the rendered kernel still passes fp32 sim."""
    import importlib.util
    from test.transforms._fixtures import INPUT_SPECS

    import numpy as np

    from nkigym.ir.tree import ForNode
    from nkigym.synthesis.simulate_nki import simulate_fp32

    ir = build_canonical_ir()
    target = next(nid for nid in ir.tree.preorder() if isinstance(ir.tree.data(nid), ForNode))
    extent = ir.tree.data(target).extent
    new_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(2, extent // 2)))
    src = render(new_ir)
    cache = tmp_path / "split_kernel"
    cache.mkdir()
    kernel_path = cache / "kernel.py"
    kernel_path.write_text(src)
    rng = np.random.default_rng(0)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, shape in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    spec = importlib.util.spec_from_file_location("split_kernel", kernel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = np.asarray(simulate_fp32(module.nki_f_matmul)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_fuse_outer_trip_renders_and_passes_numerics(tmp_path):
    """Apply Split then Fuse on the same axis; rendered kernel still passes fp32 sim."""
    import importlib.util
    from test.transforms._fixtures import INPUT_SPECS

    import numpy as np

    from nkigym.ir.tree import ForNode
    from nkigym.synthesis.simulate_nki import simulate_fp32

    ir = build_canonical_ir()
    target = next(nid for nid in ir.tree.preorder() if isinstance(ir.tree.data(nid), ForNode))
    extent = ir.tree.data(target).extent
    split_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(2, extent // 2)))
    """Locate the new outer ForNode (target was removed by Split)."""
    original_parent = ir.tree.parent(target)
    new_top = split_ir.tree.children(original_parent)[0]
    inner = split_ir.tree.children(new_top)[0]
    fuse_ir = Fuse().apply(split_ir, FuseOption(target_nids=(new_top, inner), target_axis=None))
    src = render(fuse_ir)
    cache = tmp_path / "fuse_kernel"
    cache.mkdir()
    kernel_path = cache / "kernel.py"
    kernel_path.write_text(src)
    rng = np.random.default_rng(0)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, shape in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    spec = importlib.util.spec_from_file_location("fuse_kernel", kernel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = np.asarray(simulate_fp32(module.nki_f_matmul)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_reorder_renders_and_passes_numerics(tmp_path):
    """After Split + Reorder on adjacent same-axis ForNodes, the rendered kernel passes fp32 sim."""
    import importlib.util
    from test.transforms._fixtures import INPUT_SPECS

    import numpy as np

    from nkigym.ir.tree import ForNode
    from nkigym.synthesis.simulate_nki import simulate_fp32

    ir = build_canonical_ir()
    """Pick the matmul block's first ForNode and split it so a swap target exists."""
    target = next(nid for nid in ir.tree.preorder() if isinstance(ir.tree.data(nid), ForNode))
    extent = ir.tree.data(target).extent
    split_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(2, extent // 2)))
    """Find the adjacent ForNode pair created by Split."""
    original_parent = ir.tree.parent(target)
    outer = split_ir.tree.children(original_parent)[0]
    inner = split_ir.tree.children(outer)[0]
    new_ir = Reorder().apply(split_ir, ReorderOption(outer_nid=outer, inner_nid=inner))
    src = render(new_ir)
    cache = tmp_path / "reorder_kernel"
    cache.mkdir()
    kernel_path = cache / "kernel.py"
    kernel_path.write_text(src)
    rng = np.random.default_rng(0)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, shape in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    spec = importlib.util.spec_from_file_location("reorder_kernel", kernel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = np.asarray(simulate_fp32(module.nki_f_matmul)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)
