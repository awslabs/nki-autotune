"""Render, simulation, and structural-oracle tests for Split."""

from __future__ import annotations

from test._simulation import assert_matmul_ir_simulates
from test.transforms._fixtures import build_canonical_ir
from test.transforms._helpers import leaf_for_op

import pytest

from nkigym.codegen import render
from nkigym.ir.arith.expr import Const, Var, format_expr, substitute
from nkigym.ir.tree import BlockNode, ForNode, ISANode
from nkigym.transforms import Split, SplitOption


def test_split_outer_trip_renders_and_passes_numerics(tmp_path) -> None:
    """An outer-trip split preserves rendered-kernel behavior."""
    ir = build_canonical_ir()
    target = next(nid for nid in ir.tree.preorder() if isinstance(ir.tree.data(nid), ForNode))
    extent = ir.tree.data(target).extent
    split = Split().apply(ir, SplitOption(target_nid=target, factors=(2, extent // 2)))
    assert_matmul_ir_simulates(split, tmp_path, "split_outer_trip")


def test_split_tensorize_load_d1_to_16x128(tmp_path) -> None:
    """A tensorized load split changes its tile width and preserves behavior."""
    ir = build_canonical_ir()
    load = leaf_for_op(ir, "NKILoad")
    split = Split().apply(ir, SplitOption(target_nid=load, factors=(16, 128), target_axis="d1"))
    new_load = leaf_for_op(split, "NKILoad")
    destination = split.tree.data(new_load).operand_bindings["dst"]

    assert any(isinstance(width, Const) and width.value == 128 for _low, width in destination.ranges)
    assert_matmul_ir_simulates(split, tmp_path, "split_load_d1")


@pytest.mark.parametrize(
    "op_name, occurrence, axis, factors",
    [
        ("NKILoad", 1, "d2", (4, 512)),
        ("NKIMemset", 0, "d2", (4, 512)),
        ("NKITensorCopy", 0, "d2", (4, 512)),
        ("NKIStore", 0, "d2", (4, 512)),
    ],
)
def test_split_tensorize_ladder_ops_render_and_sim(
    tmp_path, op_name: str, occurrence: int, axis: str, factors: tuple[int, ...]
) -> None:
    """Each tensorized split used by the transform ladder preserves behavior."""
    ir = build_canonical_ir()
    leaf = leaf_for_op(ir, op_name, occurrence)
    split = Split().apply(ir, SplitOption(target_nid=leaf, factors=factors, target_axis=axis))
    module_name = f"split_{op_name.lower()}_{occurrence}"
    assert_matmul_ir_simulates(split, tmp_path, module_name)


def test_split_tensorize_n_to_min_tile_still_simulates(tmp_path) -> None:
    """A tensorized split ending at the minimum tile size remains valid."""
    ir = build_canonical_ir()
    matmul = leaf_for_op(ir, "NKIMatmul")
    split = Split().apply(ir, SplitOption(target_nid=matmul, factors=(4, 128), target_axis="d2"))
    assert_matmul_ir_simulates(split, tmp_path, "split_matmul_n_min_tile")


def test_split_load_d1_matches_hand_k1_render() -> None:
    """The first ladder split renders one dense loop without a trip-one wrapper."""
    ir = build_canonical_ir()
    load = leaf_for_op(ir, "NKILoad")
    split = Split().apply(ir, SplitOption(target_nid=load, factors=(16, 128), target_axis="d1"))
    lines = [line.strip() for line in render(split).splitlines()]
    load_line = next(line for line in lines if "dst=sbuf_lhs_T" in line)

    assert "for i_d1_0 in range(16):" in lines
    assert not any("i_d1_0_0" in line for line in lines)
    assert not any(line.startswith("for ") and "range(1)" in line for line in lines)
    assert "i_d1_0 * 128" in load_line and "+ 128" in load_line
    assert "* 2048" not in load_line


def test_split_matches_tvm_structure() -> None:
    """An outer-trip split matches TVM's loop extents and recovered binding."""
    pytest.importorskip("tvm")
    from test.transforms._oracle_helpers import enclosing_for_nids
    from test.transforms._tvm_struct_oracle import tvm_split_loopnest

    ir = build_canonical_ir()
    matmul = leaf_for_op(ir, "NKIMatmul")
    target = next(
        ancestor
        for ancestor in ir.tree.ancestors(matmul)
        if isinstance(ir.tree.data(ancestor), ForNode) and ir.tree.data(ancestor).loop_var == "i_d1_0"
    )
    split = Split().apply(ir, SplitOption(target_nid=target, factors=(4, 4), target_axis=None))
    oracle = tvm_split_loopnest(extent=16, factors=[4, 4])
    new_matmul = leaf_for_op(split, "NKIMatmul")
    loops = enclosing_for_nids(split, new_matmul, "i_d1")

    assert [split.tree.data(nid).extent for nid in loops] == oracle.extents == [4, 4]
    block = next(
        split.tree.data(ancestor)
        for ancestor in reversed(split.tree.ancestors(new_matmul))
        if isinstance(split.tree.data(ancestor), BlockNode)
    )
    value = next(value for var, value in zip(block.iter_vars, block.iter_values) if var.axis == "d1")
    loop_vars = [split.tree.data(nid).loop_var for nid in loops]
    renamed = substitute(value, {name: Var(name=f"i{index}") for index, name in enumerate(loop_vars)})
    assert format_expr(renamed).replace(" * ", "*") == oracle.binding


def test_tensorize_split_matches_tvm_structure() -> None:
    """A tensorized split matches TVM's outer loop and inner tile extent."""
    pytest.importorskip("tvm")
    from test.transforms._oracle_helpers import enclosing_for_nids
    from test.transforms._tvm_struct_oracle import tvm_split_loopnest

    ir = build_canonical_ir()
    load = next(
        nid
        for nid in ir.tree.preorder()
        if isinstance(ir.tree.data(nid), ISANode)
        and ir.tree.data(nid).op_cls.__name__ == "NKILoad"
        and ir.tree.data(nid).operand_bindings["src"].tensor == "lhs_T"
    )
    split = Split().apply(ir, SplitOption(target_nid=load, factors=(16, 128), target_axis="d1"))
    oracle = tvm_split_loopnest(extent=2048, factors=[16, 128])
    new_load = next(
        nid
        for nid in split.tree.preorder()
        if isinstance(split.tree.data(nid), ISANode)
        and split.tree.data(nid).op_cls.__name__ == "NKILoad"
        and split.tree.data(nid).operand_bindings["src"].tensor == "lhs_T"
    )
    loops = enclosing_for_nids(split, new_load, "i_d1")
    block = next(
        split.tree.data(ancestor)
        for ancestor in reversed(split.tree.ancestors(new_load))
        if isinstance(split.tree.data(ancestor), BlockNode)
    )
    inverse_axis_map = {concrete: abstract for abstract, concrete in block.axis_map.items()}
    destination = split.tree.data(new_load).operand_bindings["dst"]
    free_index = split.tree.data(new_load).op_cls.OPERAND_AXES["dst"].index(inverse_axis_map["d1"])
    _low, width = destination.ranges[free_index]

    assert [split.tree.data(nid).extent for nid in loops] == oracle.extents[:-1] == [16]
    assert isinstance(width, Const) and width.value == oracle.extents[-1] == 128
