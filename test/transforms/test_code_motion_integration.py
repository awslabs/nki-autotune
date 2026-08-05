"""Integration tests for CodeMotion and transform-ladder states."""

from __future__ import annotations

from dataclasses import replace
from test._simulation import assert_matmul_ir_simulates
from test.transforms._fixtures import INPUT_SPECS, build_canonical_ir, f_matmul
from test.transforms._helpers import block_for_op, first_for_in, load_block_reading

from nkigym.environment import KernelMDP
from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Expr, Var, to_affine
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms import CodeMotion, CodeMotionOption, Split, SplitOption
from nkigym.transforms.code_motion import _substitute_block_loop_vars


def _rename_block_loops(ir: KernelIR, block_nid: int, names: dict[str, str]) -> None:
    """Alpha-rename one block's loops and every binding in its local scope."""
    substitutions: dict[str, Expr] = {old: Var(name=new) for old, new in names.items()}
    _substitute_block_loop_vars(ir.tree, block_nid, substitutions)
    for nid in ir.tree.preorder(block_nid):
        node = ir.tree.data(nid)
        if isinstance(node, ForNode) and node.loop_var in names:
            ir.tree.graph.nodes[nid]["data"] = replace(node, loop_var=names[node.loop_var])
    ir.dependency = Dependency(ir.tree)


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


def test_code_motion_matches_equivalent_prefix_with_distinct_loop_names(tmp_path) -> None:
    """Equivalent loop dimensions merge even when independently named."""
    ir = build_canonical_ir()
    memset = block_for_op(ir, "NKIMemset")
    lhs_load = load_block_reading(ir, "lhs_T")
    memset_m = next(
        nid
        for nid in ir.tree.preorder(memset)
        if isinstance((node := ir.tree.data(nid)), ForNode) and node.loop_var == "i_d1_0"
    )
    lhs_leaf = next(nid for nid in ir.tree.preorder(lhs_load) if isinstance(ir.tree.data(nid), ISANode))
    ir = Split().apply(ir, SplitOption(target_nid=memset_m, factors=(4, 2, 2), target_axis=None))
    ir = Split().apply(ir, SplitOption(target_nid=lhs_leaf, factors=(4, 2, 2, 128), target_axis="d1"))
    _rename_block_loops(ir, lhs_load, {f"i_d1_{index}": f"i_d1_{index + 3}" for index in range(3)})

    target = next(
        nid
        for nid in ir.tree.preorder(lhs_load)
        if isinstance((node := ir.tree.data(nid)), ForNode) and node.loop_var == "i_d1_4"
    )
    option = CodeMotionOption(block_nid=memset, target_loop_nid=target, index=0)
    moved = CodeMotion().apply(ir, option)

    bound_names = {
        name for value in moved.tree.block(memset).iter_values for name in to_affine(value) if name is not None
    }
    assert {"i_d1_3", "i_d1_4"} <= bound_names
    for leaf in moved.tree.preorder():
        if not isinstance(moved.tree.data(leaf), ISANode):
            continue
        loop_names = [
            moved.tree.loop(nid).loop_var
            for nid in moved.tree.ancestors(leaf)
            if isinstance(moved.tree.data(nid), ForNode)
        ]
        assert len(loop_names) == len(set(loop_names))
    assert_matmul_ir_simulates(moved, tmp_path, "code_motion_distinct_prefix_names")
