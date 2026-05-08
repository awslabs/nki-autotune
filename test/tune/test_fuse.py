"""Unit tests for Fuse atom."""

from nkigym.codegen.dep_cache import DepCache
from nkigym.codegen.ir import BodyLeaf, KernelModule, LoopNode, resolve_node
from nkigym.ops.base import AxisRole
from nkigym.tune.fuse import Fuse


def _mod(body):
    return KernelModule(
        func_name="f", param_names=[], return_name="x", tensors={}, dims={}, body=body, dep=DepCache(scopes={})
    )


def test_fuse_par_par_collapses():
    leaf = BodyLeaf(op_cls=object, phase="main")
    inner = LoopNode("d1", 8, AxisRole.PARALLEL, children=[leaf])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, children=[inner])
    mod = _mod([outer])
    atom = Fuse(outer_path=(0,), inner_path=(0, 0))
    assert atom.is_legal(mod)
    new_mod = atom.apply(mod)
    fused = resolve_node(new_mod.body, (0,))
    assert isinstance(fused, LoopNode)
    assert fused.trip_count == 32


def test_fuse_rejects_non_perfect_outer():
    leaf_a = BodyLeaf(op_cls=object, phase="a")
    leaf_b = BodyLeaf(op_cls=object, phase="b")
    inner = LoopNode("d1", 8, AxisRole.PARALLEL, children=[leaf_a])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, children=[inner, leaf_b])
    mod = _mod([outer])
    atom = Fuse(outer_path=(0,), inner_path=(0, 0))
    assert not atom.is_legal(mod)


def test_fuse_rejects_sequential():
    leaf = BodyLeaf(op_cls=object, phase="main")
    inner = LoopNode("d1", 2, AxisRole.SEQUENTIAL, children=[leaf])
    outer = LoopNode("d0", 2, AxisRole.PARALLEL, children=[inner])
    mod = _mod([outer])
    atom = Fuse(outer_path=(0,), inner_path=(0, 0))
    assert not atom.is_legal(mod)


def test_fuse_acc_acc_different_reduce_op_illegal():
    leaf = BodyLeaf(op_cls=object, phase="main")
    inner = LoopNode("d1", 2, AxisRole.ACCUMULATION, reduce_op="max", children=[leaf])
    outer = LoopNode("d0", 2, AxisRole.ACCUMULATION, reduce_op="add", children=[inner])
    mod = _mod([outer])
    atom = Fuse(outer_path=(0,), inner_path=(0, 0))
    assert not atom.is_legal(mod)


def test_fuse_acc_acc_same_op_legal():
    leaf = BodyLeaf(op_cls=object, phase="main")
    inner = LoopNode("d1", 2, AxisRole.ACCUMULATION, reduce_op="add", children=[leaf])
    outer = LoopNode("d0", 2, AxisRole.ACCUMULATION, reduce_op="add", children=[inner])
    mod = _mod([outer])
    atom = Fuse(outer_path=(0,), inner_path=(0, 0))
    assert atom.is_legal(mod)
