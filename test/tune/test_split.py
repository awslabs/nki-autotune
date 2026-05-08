"""Unit tests for Split atom."""

from nkigym.codegen.dep_cache import DepCache
from nkigym.codegen.ir import BodyLeaf, KernelModule, LoopNode, resolve_node
from nkigym.ops.base import AxisRole
from nkigym.tune.split import Split


def _mod(body):
    return KernelModule(
        func_name="f", param_names=[], return_name="x", tensors={}, dims={}, body=body, dep=DepCache(scopes={})
    )


def test_split_divisible_yields_single_pair():
    leaf = BodyLeaf(op_cls=object, phase="main")
    loop = LoopNode("d0", 16, AxisRole.PARALLEL, children=[leaf])
    mod = _mod([loop])
    atom = Split(loop_path=(0,), factor=4)
    assert atom.is_legal(mod)
    new_mod = atom.apply(mod)
    assert len(new_mod.body) == 1
    new_outer = resolve_node(new_mod.body, (0,))
    assert isinstance(new_outer, LoopNode)
    assert new_outer.trip_count == 4
    new_inner = resolve_node(new_mod.body, (0, 0))
    assert isinstance(new_inner, LoopNode)
    assert new_inner.trip_count == 4


def test_split_non_divisible_yields_two_siblings():
    leaf = BodyLeaf(op_cls=object, phase="main")
    loop = LoopNode("d0", 17, AxisRole.PARALLEL, children=[leaf])
    mod = _mod([loop])
    atom = Split(loop_path=(0,), factor=4)
    assert atom.is_legal(mod)
    new_mod = atom.apply(mod)
    assert len(new_mod.body) == 2
    full_outer = new_mod.body[0]
    tail_outer = new_mod.body[1]
    assert isinstance(full_outer, LoopNode) and full_outer.trip_count == 4
    assert isinstance(tail_outer, LoopNode) and tail_outer.trip_count == 1
    full_inner = full_outer.children[0]
    tail_inner = tail_outer.children[0]
    assert isinstance(full_inner, LoopNode) and full_inner.trip_count == 4
    assert isinstance(tail_inner, LoopNode) and tail_inner.trip_count == 1


def test_split_factor_greater_than_trip():
    leaf = BodyLeaf(op_cls=object, phase="main")
    loop = LoopNode("d0", 3, AxisRole.PARALLEL, children=[leaf])
    mod = _mod([loop])
    atom = Split(loop_path=(0,), factor=5)
    assert atom.is_legal(mod)
    new_mod = atom.apply(mod)
    """Only the tail pair should appear (full_iters == 0)."""
    assert len(new_mod.body) == 1
    tail_outer = new_mod.body[0]
    assert isinstance(tail_outer, LoopNode)
    assert tail_outer.trip_count == 1
    tail_inner = tail_outer.children[0]
    assert isinstance(tail_inner, LoopNode)
    assert tail_inner.trip_count == 3


def test_split_rejects_factor_zero():
    leaf = BodyLeaf(op_cls=object, phase="main")
    loop = LoopNode("d0", 16, AxisRole.PARALLEL, children=[leaf])
    mod = _mod([loop])
    atom = Split(loop_path=(0,), factor=0)
    assert not atom.is_legal(mod)


def test_split_rejects_leaf_target():
    leaf = BodyLeaf(op_cls=object, phase="main")
    mod = _mod([leaf])
    atom = Split(loop_path=(0,), factor=4)
    assert not atom.is_legal(mod)
