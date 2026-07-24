"""Tests for nkigym.transforms.Reorder under BlockNode IR."""

from __future__ import annotations

from test._simulation import assert_matmul_ir_simulates
from test.transforms._fixtures import build_canonical_ir
from test.transforms._helpers import load_block_reading, matmul_loop
from test.transforms._seq_fixture import build_seq_ir

import pytest

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Add, Const, Mul, Var
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode, IterVar, KernelTree
from nkigym.ops.base import AxisRole
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.transforms import (
    CodeMotion,
    CodeMotionOption,
    Reorder,
    ReorderOption,
    Split,
    SplitOption,
    TransformLegalityError,
)


def _first_two_adjacent_fors(ir):
    """Return (outer_nid, inner_nid) for the first parent-child ForNode pair."""
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if not isinstance(data, ForNode):
            continue
        kids = ir.tree.children(nid)
        if len(kids) != 1:
            continue
        kid_data = ir.tree.data(kids[0])
        if isinstance(kid_data, ForNode):
            return nid, kids[0]
    raise AssertionError("no adjacent ForNode pair")


def test_reorder_swaps_payloads():
    """Apply swaps the two ForNode payloads while keeping nids stable."""
    ir = build_canonical_ir()
    outer, inner = _first_two_adjacent_fors(ir)
    outer_data = ir.tree.data(outer)
    inner_data = ir.tree.data(inner)
    new_ir = Reorder().apply(ir, ReorderOption(outer_nid=outer, inner_nid=inner))
    assert new_ir.tree.data(outer) == inner_data
    assert new_ir.tree.data(inner) == outer_data


def test_reorder_self_inverse():
    """Apply twice returns the original payload."""
    ir = build_canonical_ir()
    outer, inner = _first_two_adjacent_fors(ir)
    opt = ReorderOption(outer_nid=outer, inner_nid=inner)
    new_ir = Reorder().apply(Reorder().apply(ir, opt), opt)
    assert new_ir.tree.data(outer) == ir.tree.data(outer)
    assert new_ir.tree.data(inner) == ir.tree.data(inner)


def test_reorder_renders_and_passes_numerics(tmp_path) -> None:
    """Reordering a split loop pair preserves rendered-kernel behavior."""
    ir = build_canonical_ir()
    target, _inner = _first_two_adjacent_fors(ir)
    extent = ir.tree.loop(target).extent
    split = Split().apply(ir, SplitOption(target_nid=target, factors=(2, extent // 2)))
    target_parent = ir.tree.parent(target)
    assert target_parent is not None
    outer = split.tree.children(target_parent)[0]
    inner = split.tree.children(outer)[0]
    reordered = Reorder().apply(split, ReorderOption(outer_nid=outer, inner_nid=inner))
    assert_matmul_ir_simulates(reordered, tmp_path, "reorder_split_loop")


def test_reorder_rejects_sequential_role():
    """Reorder rejects a swap on a dim whose enclosing block declares SEQUENTIAL role."""
    ir, outer, inner, _ = build_seq_ir()
    with pytest.raises(TransformLegalityError, match="SEQUENTIAL"):
        Reorder().apply(ir, ReorderOption(outer_nid=outer, inner_nid=inner))


def test_reorder_rejects_internal_loop_carried_flow():
    """Reorder rejects a scratch flow carried across the inner loop."""
    tree = KernelTree()
    outer_var = Var(name="i_d1_0")
    inner_var = Var(name="i_d2_0")
    suffix_var = Var(name="i_d2_1")
    scratch_read = BufferRegion(
        tensor="scratch",
        ranges=(
            (Const(value=0), Const(value=128)),
            (
                Add(
                    left=Mul(left=inner_var, right=Const(value=1024)),
                    right=Mul(left=suffix_var, right=Const(value=512)),
                ),
                Const(value=512),
            ),
        ),
    )
    output_write = BufferRegion(
        tensor="output",
        ranges=((Mul(left=outer_var, right=Const(value=128)), Const(value=128)), scratch_read.ranges[1]),
    )
    owner = BlockNode(
        iter_vars=(
            IterVar(axis="d1", dom=(0, 16), role=AxisRole.PARALLEL),
            IterVar(axis="d2", dom=(0, 4), role=AxisRole.PARALLEL),
        ),
        iter_values=(outer_var, Add(left=Mul(left=inner_var, right=Const(value=2)), right=suffix_var)),
        reads=(scratch_read,),
        writes=(output_write,),
        alloc_buffers=(Buffer(name="scratch", shape=(128, 2048), dtype="bfloat16", location="sbuf"),),
        axis_map={"P": "d1", "F": "d2"},
    )
    owner_nid = tree.add_node(owner, parent=tree.root)
    outer = tree.add_node(ForNode(loop_var=outer_var.name, extent=16), parent=owner_nid)
    inner = tree.add_node(ForNode(loop_var=inner_var.name, extent=2), parent=outer)
    suffix = tree.add_node(ForNode(loop_var=suffix_var.name, extent=2), parent=inner)
    scratch_write = BufferRegion(
        tensor="scratch",
        ranges=((Const(value=0), Const(value=128)), (Mul(left=suffix_var, right=Const(value=1024)), Const(value=1024))),
    )
    source_read = BufferRegion(tensor="source", ranges=((outer_var, Const(value=128)), scratch_write.ranges[1]))
    producer = BlockNode(
        iter_vars=(
            IterVar(axis="d1", dom=(0, 16), role=AxisRole.PARALLEL),
            IterVar(axis="d2", dom=(0, 2), role=AxisRole.PARALLEL),
        ),
        iter_values=(outer_var, suffix_var),
        reads=(source_read,),
        writes=(scratch_write,),
        axis_map={"P": "d1", "F": "d2"},
    )
    producer_nid = tree.add_node(producer, parent=suffix)
    tree.add_node(
        ISANode(op_cls=NKITensorCopy, operand_bindings={"src": source_read, "dst": scratch_write}), parent=producer_nid
    )
    tree.add_node(ISANode(op_cls=NKIStore, operand_bindings={"src": scratch_read, "dst": output_write}), parent=suffix)
    ir = KernelIR(
        func_name="loop_carried_flow",
        param_names=["source"],
        return_name="output",
        tree=tree,
        dependency=Dependency(tree),
    )
    option = ReorderOption(outer_nid=outer, inner_nid=inner)

    assert option not in Reorder().analyze(ir)
    with pytest.raises(TransformLegalityError, match="different dependence"):
        Reorder().apply(ir, option)


def test_reorder_same_dim_swap_then_compute_at_sims(tmp_path) -> None:
    """A same-dimension reorder stays consistent with later CodeMotion normalization."""
    ir = build_canonical_ir()
    ir = Split().apply(ir, SplitOption(target_nid=matmul_loop(ir, "i_d0_0"), factors=(4, 2, 2), target_axis=None))
    rhs_load = load_block_reading(ir, "rhs")
    rhs_k_loop = next(
        nid
        for nid in ir.tree.preorder(rhs_load)
        if isinstance((node := ir.tree.data(nid)), ForNode) and node.loop_var == "i_d0_0"
    )
    ir = Split().apply(ir, SplitOption(target_nid=rhs_k_loop, factors=(4, 2, 2), target_axis=None))

    ir = Reorder().apply(ir, ReorderOption(outer_nid=matmul_loop(ir, "i_d0_0"), inner_nid=matmul_loop(ir, "i_d0_1")))
    rhs_loops = [
        nid
        for nid in ir.tree.preorder(rhs_load)
        if isinstance((node := ir.tree.data(nid)), ForNode) and node.loop_var.startswith("i_d0_")
    ]
    ir = Reorder().apply(ir, ReorderOption(outer_nid=rhs_loops[0], inner_nid=rhs_loops[1]))
    target = matmul_loop(ir, "i_d0_0")
    moved = CodeMotion().apply(ir, CodeMotionOption(block_nid=rhs_load, target_loop_nid=target, index=0))

    assert_matmul_ir_simulates(moved, tmp_path, "reorder_same_dim_compute_at")
