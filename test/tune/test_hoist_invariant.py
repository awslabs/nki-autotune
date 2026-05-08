"""Unit tests for HoistInvariant atom."""

from nkigym.codegen.dep_cache import DepCache
from nkigym.codegen.ir import BodyLeaf, KernelModule, LoopNode
from nkigym.ops.base import AxisRole
from nkigym.tune.hoist_invariant import HoistInvariant


def _mod(body, dims=None):
    return KernelModule(
        func_name="f", param_names=[], return_name="x", tensors={}, dims=dims or {}, body=body, dep=DepCache(scopes={})
    )


def test_hoist_invariant_moves_leaf_out_of_unrelated_loop():
    """Leaf references d0 only; hoisting from under d1 to above d1 is legal."""
    leaf = BodyLeaf(
        op_cls=object,
        phase="main",
        reads={"data": "rhs"},
        writes=("rhs_s",),
        axis_map={"P": "d0"},
        dim_role={"d0": AxisRole.PARALLEL},
    )
    inner = LoopNode("d1", 4, AxisRole.PARALLEL, children=[leaf])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, children=[inner])
    mod = _mod([outer])
    atom = HoistInvariant(leaf_path=(0, 0, 0), target_loop_path=(0,))
    assert atom.is_legal(mod)


def test_hoist_invariant_rejects_when_crossed_dim_in_leaf():
    """Leaf references d1; hoisting past d1 is not pure LICM."""
    leaf = BodyLeaf(
        op_cls=object,
        phase="main",
        reads={"data": "rhs"},
        writes=("rhs_s",),
        axis_map={"P": "d0", "F": "d1"},
        dim_role={"d0": AxisRole.PARALLEL, "d1": AxisRole.PARALLEL},
    )
    inner = LoopNode("d1", 4, AxisRole.PARALLEL, children=[leaf])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, children=[inner])
    mod = _mod([outer])
    atom = HoistInvariant(leaf_path=(0, 0, 0), target_loop_path=(0,))
    assert not atom.is_legal(mod)


def test_hoist_invariant_rejects_non_ancestor_target():
    leaf = BodyLeaf(op_cls=object, phase="main", axis_map={"P": "d0"}, dim_role={"d0": AxisRole.PARALLEL})
    loop_a = LoopNode("d0", 4, AxisRole.PARALLEL, children=[leaf])
    loop_b = LoopNode("d1", 4, AxisRole.PARALLEL)
    mod = _mod([loop_a, loop_b])
    atom = HoistInvariant(leaf_path=(0, 0), target_loop_path=(1,))
    assert not atom.is_legal(mod)
