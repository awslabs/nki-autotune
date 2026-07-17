"""Integration tests for CodeMotion and transform-ladder states."""

from __future__ import annotations

from test._simulation import assert_matmul_ir_simulates
from test.transforms._fixtures import INPUT_SPECS, build_canonical_ir, build_ladder_state, f_matmul
from test.transforms._helpers import block_for_op, first_for_in

import pytest

from nkigym.environment import KernelMDP
from nkigym.ir.tree import ISANode
from nkigym.transforms import CodeMotion, CodeMotionOption, Split, SplitOption


def test_analyze_does_not_crash_on_transformed_states() -> None:
    """Analyze filters invalid candidates across ladder states without raising."""
    for rung in range(1, 13):
        CodeMotion().analyze(build_ladder_state(rung))


def test_code_motion_sink_load_under_matmul_renders_and_sims(tmp_path) -> None:
    """Sink the lhs load under the matmul's inner loop and simulate the result."""
    ir = build_canonical_ir()
    load = block_for_op(ir, "NKILoad")
    matmul = block_for_op(ir, "NKIMatmul")
    leaf = next(desc for desc in ir.tree.preorder(matmul) if isinstance(ir.tree.data(desc), ISANode))
    inner = ir.tree.ancestors(leaf)[-1]
    moved = CodeMotion().apply(ir, CodeMotionOption(block_nid=load, target_loop_nid=inner, index=-2))

    assert load in moved.tree.descendants(inner)
    assert_matmul_ir_simulates(moved, tmp_path, "code_motion_sink_load")


def test_code_motion_lift_store_under_tensor_copy_renders_and_sims(tmp_path) -> None:
    """Lift the output store under the drain's parallel loop and simulate it."""
    ir = build_canonical_ir()
    store = block_for_op(ir, "NKIStore")
    tensor_copy = block_for_op(ir, "NKITensorCopy")
    target = first_for_in(ir, tensor_copy)
    moved = CodeMotion().apply(ir, CodeMotionOption(block_nid=store, target_loop_nid=target, index=-1))

    assert store in moved.tree.descendants(target)
    assert_matmul_ir_simulates(moved, tmp_path, "code_motion_lift_store")


def test_code_motion_lift_preserves_covered_dim_across_block_wall(tmp_path) -> None:
    """Keep an enclosing dimension driver when moving across a nested block."""
    trace = [
        (CodeMotion(), CodeMotionOption(block_nid=4, target_loop_nid=11, index=0)),
        (Split(), SplitOption(target_nid=17, factors=(8, 256), target_axis="d2")),
        (CodeMotion(), CodeMotionOption(block_nid=1, target_loop_nid=11, index=1)),
        (Split(), SplitOption(target_nid=3, factors=(2, 1024), target_axis="d1")),
        (CodeMotion(), CodeMotionOption(block_nid=4, target_loop_nid=22, index=0)),
    ]
    environment = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), CodeMotion()])
    state = environment.reset()
    for action in trace:
        state = environment.step(state, action)

    assert_matmul_ir_simulates(state, tmp_path, "code_motion_block_wall")


def test_code_motion_lift_deeply_nested_load_preserves_dim_driver(tmp_path) -> None:
    """Keep an enclosing dimension driver when moving across multiple blocks."""
    trace = [
        (CodeMotion(), CodeMotionOption(block_nid=1, target_loop_nid=13, index=0)),
        (CodeMotion(), CodeMotionOption(block_nid=1, target_loop_nid=5, index=1)),
        (CodeMotion(), CodeMotionOption(block_nid=1, target_loop_nid=11, index=0)),
        (CodeMotion(), CodeMotionOption(block_nid=4, target_loop_nid=11, index=0)),
        (Split(), SplitOption(target_nid=17, factors=(2, 4, 256), target_axis="d2")),
        (Split(), SplitOption(target_nid=6, factors=(2, 4, 256), target_axis="d2")),
        (CodeMotion(), CodeMotionOption(block_nid=1, target_loop_nid=24, index=0)),
        (Split(), SplitOption(target_nid=9, factors=(4, 2, 256), target_axis="d2")),
        (Split(), SplitOption(target_nid=25, factors=(2, 2), target_axis=None)),
        (CodeMotion(), CodeMotionOption(block_nid=1, target_loop_nid=23, index=1)),
    ]
    environment = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), CodeMotion()])
    state = environment.reset()
    for action in trace:
        state = environment.step(state, action)

    assert_matmul_ir_simulates(state, tmp_path, "code_motion_nested_load")


def test_psum_hoist_descends_and_compacts() -> None:
    """The psum allocation descends and compacts at ladder rung 12."""
    ir = build_ladder_state(12)
    declarations = {
        buffer.name: (nid, buffer) for nid in ir.tree.blocks() for buffer in ir.tree.block(nid).alloc_buffers
    }
    nid, buffer = declarations["psum_prod"]
    assert nid != ir.tree.root
    assert buffer.shape == (128, 512)


@pytest.mark.parametrize("rung", range(1, 15))
def test_ladder_state_sims(rung, tmp_path) -> None:
    """Every transform-ladder state simulates to the matmul result."""
    ir = build_ladder_state(rung)
    assert_matmul_ir_simulates(ir, tmp_path, f"ladder_state_{rung}")
