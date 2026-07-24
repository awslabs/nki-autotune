"""Render and simulation tests for Split."""

from __future__ import annotations

from test._simulation import assert_matmul_ir_simulates
from test.transforms._fixtures import build_canonical_ir
from test.transforms._helpers import leaf_for_op

import pytest

from nkigym.codegen import render
from nkigym.ir.arith.expr import Const
from nkigym.ir.tree import ForNode
from nkigym.transforms import Split, SplitOption


def test_split_outer_trip_renders_and_passes_numerics(tmp_path) -> None:
    """An outer-trip split preserves rendered-kernel behavior."""
    ir = build_canonical_ir()
    target = next(nid for nid in ir.tree.preorder() if isinstance(ir.tree.data(nid), ForNode))
    extent = ir.tree.loop(target).extent
    split = Split().apply(ir, SplitOption(target_nid=target, factors=(2, extent // 2)))
    assert_matmul_ir_simulates(split, tmp_path, "split_outer_trip")


def test_split_tensorize_load_d1_to_16x128(tmp_path) -> None:
    """A tensorized load split changes its tile width and preserves behavior."""
    ir = build_canonical_ir()
    load = leaf_for_op(ir, "NKILoad")
    split = Split().apply(ir, SplitOption(target_nid=load, factors=(16, 128), target_axis="d1"))
    new_load = leaf_for_op(split, "NKILoad")
    destination = split.tree.isa(new_load).operand_bindings["dst"]

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
