"""Unit tests for the Reorder atom."""

from nkigym.codegen.dep_cache import DepCache
from nkigym.codegen.ir import BodyLeaf, KernelModule, LoopNode, resolve_node
from nkigym.ops.base import AxisRole
from nkigym.tune.reorder import Reorder


def _mod_with_body(body):
    return KernelModule(
        func_name="f", param_names=[], return_name="x", tensors={}, dims={}, body=body, dep=DepCache(scopes={})
    )


def test_reorder_par_par_legal():
    leaf = BodyLeaf(op_cls=object)
    inner = LoopNode("d1", 2, AxisRole.PARALLEL, children=[leaf])
    outer = LoopNode("d0", 2, AxisRole.PARALLEL, children=[inner])
    mod = _mod_with_body([outer])
    atom = Reorder(outer_path=(0,), inner_path=(0, 0))
    assert atom.is_legal(mod)


def test_reorder_par_par_swaps():
    leaf = BodyLeaf(op_cls=object)
    inner = LoopNode("d1", 2, AxisRole.PARALLEL, children=[leaf])
    outer = LoopNode("d0", 2, AxisRole.PARALLEL, children=[inner])
    mod = _mod_with_body([outer])
    atom = Reorder(outer_path=(0,), inner_path=(0, 0))
    new_mod = atom.apply(mod)
    new_outer = resolve_node(new_mod.body, (0,))
    assert isinstance(new_outer, LoopNode)
    assert new_outer.dim_id == "d1"


def test_reorder_acc_acc_same_op_legal():
    leaf = BodyLeaf(op_cls=object)
    inner = LoopNode("d1", 2, AxisRole.ACCUMULATION, reduce_op="add", children=[leaf])
    outer = LoopNode("d0", 2, AxisRole.ACCUMULATION, reduce_op="add", children=[inner])
    mod = _mod_with_body([outer])
    atom = Reorder(outer_path=(0,), inner_path=(0, 0))
    assert atom.is_legal(mod)


def test_reorder_acc_acc_different_op_illegal():
    leaf = BodyLeaf(op_cls=object)
    inner = LoopNode("d1", 2, AxisRole.ACCUMULATION, reduce_op="max", children=[leaf])
    outer = LoopNode("d0", 2, AxisRole.ACCUMULATION, reduce_op="add", children=[inner])
    mod = _mod_with_body([outer])
    atom = Reorder(outer_path=(0,), inner_path=(0, 0))
    assert not atom.is_legal(mod)


def test_reorder_rejects_sequential():
    leaf = BodyLeaf(op_cls=object)
    inner = LoopNode("d1", 2, AxisRole.SEQUENTIAL, children=[leaf])
    outer = LoopNode("d0", 2, AxisRole.PARALLEL, children=[inner])
    mod = _mod_with_body([outer])
    atom = Reorder(outer_path=(0,), inner_path=(0, 0))
    assert not atom.is_legal(mod)


def test_reorder_rejects_non_perfect_nest():
    leaf_a = BodyLeaf(op_cls=object)
    leaf_b = BodyLeaf(op_cls=object)
    inner = LoopNode("d1", 2, AxisRole.PARALLEL, children=[leaf_a])
    outer = LoopNode("d0", 2, AxisRole.PARALLEL, children=[inner, leaf_b])
    mod = _mod_with_body([outer])
    atom = Reorder(outer_path=(0,), inner_path=(0, 0))
    assert not atom.is_legal(mod)


def test_reorder_par_acc_legal_when_acc_subtree_pure():
    """ACC's subtree has no leaf writing a buffer indexed by PAR's dim."""
    leaf = BodyLeaf(
        op_cls=object, reads={}, writes=("out",), axis_map={"K": "d1"}, dim_role={"d1": AxisRole.ACCUMULATION}
    )
    inner = LoopNode("d1", 2, AxisRole.ACCUMULATION, reduce_op="add", children=[leaf])
    outer = LoopNode("d0", 2, AxisRole.PARALLEL, children=[inner])
    mod = _mod_with_body([outer])
    atom = Reorder(outer_path=(0,), inner_path=(0, 0))
    assert atom.is_legal(mod)


def test_reorder_par_acc_illegal_when_acc_subtree_writes_par_dim():
    """ACC's subtree has a leaf writing a buffer whose index depends on PAR's dim."""
    leaf = BodyLeaf(
        op_cls=object,
        reads={},
        writes=("psum",),
        axis_map={"M": "d0", "K": "d1"},
        dim_role={"d0": AxisRole.PARALLEL, "d1": AxisRole.ACCUMULATION},
    )
    inner = LoopNode("d1", 2, AxisRole.ACCUMULATION, reduce_op="add", children=[leaf])
    outer = LoopNode("d0", 2, AxisRole.PARALLEL, children=[inner])
    mod = _mod_with_body([outer])
    atom = Reorder(outer_path=(0,), inner_path=(0, 0))
    assert not atom.is_legal(mod)
