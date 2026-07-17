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
    extent = ir.tree.data(target).extent
    split = Split().apply(ir, SplitOption(target_nid=target, factors=(2, extent // 2)))
    outer = split.tree.children(ir.tree.parent(target))[0]
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


def test_reorder_matches_tvm_structure():
    """Layer-B guard: our Reorder's resulting loop order matches TVM's own ``schedule.reorder``.

    Reorder is a PURE PAYLOAD-SWAP (no arith): ``apply`` swaps the two adjacent
    ForNode payloads, leaving every binding intact (loop vars travel with their
    payloads), so only the tree loop ORDER changes. This guard confronts that
    resulting outer->inner order against TVM's own TensorIR ``schedule.reorder``
    on the equivalent perfect nest (the Layer-B structural oracle).

    The canonical matmul block nests ``i_d0_0`` (K=16), ``i_d1_0`` (M=16),
    ``i_d2_0`` (N=4) outer->inner. We swap the adjacent ``d1``/``d2`` pair — both
    PARALLEL, so unambiguously legal — yielding ``[d0, d2, d1]`` with extents
    ``[16, 4, 16]``. TVM's ``reorder`` of the equivalent ``[16, 16, 4]`` nest with
    ``order=[0, 2, 1]`` must reproduce the same extents. Orthogonal to the
    byte-exact ladder gate.
    """
    pytest.importorskip("tvm")
    from test.transforms._oracle_helpers import enclosing_for_nids
    from test.transforms._tvm_struct_oracle import tvm_reorder_loopnest

    ir = build_canonical_ir()
    mm = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKIMatmul"
    )
    enclosing = enclosing_for_nids(ir, mm, "i_d")
    before_order = [(ir.tree.data(n).loop_var, ir.tree.data(n).extent) for n in enclosing]
    assert before_order == [("i_d0_0", 16), ("i_d1_0", 16), ("i_d2_0", 4)]

    """Swap the adjacent d1/d2 pair (both PARALLEL); d1 is the outer of the pair."""
    d1 = next(n for n in enclosing if ir.tree.data(n).loop_var == "i_d1_0")
    d2 = next(n for n in enclosing if ir.tree.data(n).loop_var == "i_d2_0")
    out = Reorder().apply(ir, ReorderOption(outer_nid=d1, inner_nid=d2))

    out_mm = next(
        n
        for n in out.tree.preorder()
        if isinstance(out.tree.data(n), ISANode) and out.tree.data(n).op_cls.__name__ == "NKIMatmul"
    )
    our_extents = [out.tree.data(n).extent for n in enclosing_for_nids(out, out_mm, "i_d")]

    """TVM reorders the equivalent [16, 16, 4] nest, placing original loop 2 (d2)
    before original loop 1 (d1): new outer->inner order = [d0, d2, d1]."""
    source_extents = [ext for _lv, ext in before_order]
    nest = tvm_reorder_loopnest(extents=source_extents, order=[0, 2, 1])
    assert our_extents == nest.extents == [16, 4, 16]


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
