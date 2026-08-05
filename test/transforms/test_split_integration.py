"""Render and simulation tests for Split."""

from __future__ import annotations

from test._simulation import assert_matmul_ir_simulates
from test.transforms._fixtures import build_canonical_ir
from test.transforms._helpers import leaf_for_op

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


def test_split_tensorize_n_to_min_tile_still_simulates(tmp_path) -> None:
    """A tensorized split ending at the minimum tile size remains valid."""
    ir = build_canonical_ir()
    matmul = leaf_for_op(ir, "NKIMatmul")
    split = Split().apply(ir, SplitOption(target_nid=matmul, factors=(4, 128), target_axis="d2"))
    assert_matmul_ir_simulates(split, tmp_path, "split_matmul_n_min_tile")
