"""Unit tests for ComputeAt atom."""

import numpy as np
import pytest

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.dep_cache import DepCache
from nkigym.codegen.ir import BodyLeaf, DimInfo, KernelModule, LoopNode, Tensor, leaves_under
from nkigym.ops import nkigym_kernel
from nkigym.ops.base import AxisRole
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.tune.compute_at import ComputeAt, enumerate_compute_at_atoms


def _mod_hand(body: list, dims: dict[str, DimInfo], tensors: dict[str, Tensor]) -> KernelModule:
    """Build a minimal KernelModule from hand-rolled body / dims / tensors."""
    return KernelModule(
        func_name="f",
        param_names=[],
        return_name=next(iter(tensors)) if tensors else "x",
        tensors=tensors,
        dims=dims,
        body=body,
        dep=DepCache(scopes={}),
    )


@nkigym_kernel
def _matmul_k(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Simple matmul kernel fixture for ComputeAt atom tests."""
    lhs_s = NKILoad()(data=lhs)
    rhs_s = NKILoad()(data=rhs)
    out_s = NKIMatmul()(stationary=lhs_s, moving=rhs_s)
    out = NKIStore()(data=out_s)
    return out


_INPUT_SPECS: dict[str, dict] = {
    "lhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
    "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
}


@pytest.fixture
def module():
    """Build a canonical KernelModule for the matmul fixture."""
    return build_canonical_module(_matmul_k, _INPUT_SPECS)


def test_enumerator_emits_atoms(module):
    """On canonical matmul, ComputeAt emits at least one legal atom.

    For example, moving a load leaf under the matmul's outer loop.
    """
    atoms = enumerate_compute_at_atoms(module)
    assert atoms, "expected at least one legal ComputeAt atom"
    for atom in atoms:
        assert atom.is_legal(module)


def test_apply_changes_tree(module):
    """Applying a legal atom changes tree structure but preserves leaf count."""
    atoms = enumerate_compute_at_atoms(module)
    assert atoms
    atom = atoms[0]
    new_mod = atom.apply(module)
    assert new_mod.body is not module.body
    assert len(list(_all_leaves(new_mod))) == len(list(_all_leaves(module)))


def test_rejects_ancestor_target(module):
    """Moving a leaf under one of its own ancestor loops is rejected."""
    leaf_path = None

    def find_leaf(node, path):
        if isinstance(node, BodyLeaf):
            return path
        for i, c in enumerate(node.children):
            r = find_leaf(c, path + (i,))
            if r is not None:
                return r
        return None

    for i, root in enumerate(module.body):
        leaf_path = find_leaf(root, (i,))
        if leaf_path is not None and len(leaf_path) >= 2:
            break
    assert leaf_path is not None
    ancestor_path = leaf_path[:-1]
    atom = ComputeAt(leaf_path=leaf_path, target_loop_path=ancestor_path)
    assert not atom.is_legal(module)


def test_canonical_names_after_apply(module):
    """Every LoopNode in the post-apply tree has canonical ``i_<dim>_<ordinal>`` name."""
    atoms = enumerate_compute_at_atoms(module)
    atom = atoms[0]
    new_mod = atom.apply(module)

    def walk(node):
        if isinstance(node, LoopNode):
            assert node.name is not None
            assert node.name.startswith("i_")
            for c in node.children:
                walk(c)

    for root in new_mod.body:
        walk(root)


def _all_leaves(module):
    """Yield every BodyLeaf across the whole module forest."""
    for root in module.body:
        yield from leaves_under(root)


def test_compute_at_preserves_target_after_pruning(module):
    """When the leaf being moved is the only child of an ancestor LoopNode,
    removal collapses that ancestor and shifts sibling indices. target_loop_path
    must be re-resolved against the new tree."""
    for atom in enumerate_compute_at_atoms(module):
        """Apply to the original module (not composed) to exercise every atom."""
        new_mod = atom.apply(module)
        assert new_mod is not None


def test_apply_regenerates_residual_trip_on_partial_ancestor() -> None:
    """Partial-coverage ancestor → residual inner loop regenerated.

    Setup: forest body has two root subtrees.
    * Subtree 0: producer ``write={"p"}`` with no ancestor — sibling at the root.
    * Subtree 1: ``L(d0, trip=2)`` wrapping consumer ``reads={"p"}``.
    ``dims["d0"].num_tiles = 16``. Move producer under subtree 1's
    root loop (target_path=(1,)); producer's dim_role has ``d0``.
    Expected: producer appended under ``L(d0, 2)`` with a regenerated
    ``L(d0, 8)`` between target and producer leaf, covering the residual
    ``16 / 2 = 8`` trips.
    """
    producer = BodyLeaf(
        op_cls=object,
        phase="main",
        reads={},
        writes=("p",),
        kwargs={},
        axis_map={},
        dim_role={"d0": AxisRole.PARALLEL},
        op_local_buffers={},
    )
    consumer = BodyLeaf(
        op_cls=object,
        phase="main",
        reads={"data": "p"},
        writes=(),
        kwargs={},
        axis_map={},
        dim_role={"d0": AxisRole.PARALLEL},
        op_local_buffers={},
    )
    consumer_outer = LoopNode("d0", 2, AxisRole.PARALLEL, children=[consumer])
    dims = {"d0": DimInfo(dim_id="d0", total_size=2048, tile_size=128, num_tiles=16)}
    tensors = {
        "p": Tensor(
            name="p", dim_ids=("d0",), shape=(2048,), dtype="float32", origin="intermediate", buffer_degree={"d0": 1}
        )
    }
    module = _mod_hand(body=[producer, consumer_outer], dims=dims, tensors=tensors)
    atom = ComputeAt(leaf_path=(0,), target_loop_path=(1,))
    assert atom.is_legal(module)
    new_module = atom.apply(module)
    """After apply: body collapses from length 2 to 1 — producer removed
    from body[0], consumer_outer shifts to body[0]. Under the new
    consumer_outer, expect the original consumer + a regenerated
    L(d0, 8) wrapping a clone of producer."""
    assert len(new_module.body) == 1
    new_target = new_module.body[0]
    assert isinstance(new_target, LoopNode) and new_target.trip_count == 2
    regen_wrappers = [c for c in new_target.children if isinstance(c, LoopNode) and c.dim_id == "d0"]
    assert len(regen_wrappers) == 1, f"expected one regenerated d0 loop, got {len(regen_wrappers)}"
    residual_loop = regen_wrappers[0]
    assert residual_loop.trip_count == 8
    residual_child = residual_loop.children[0]
    assert isinstance(residual_child, BodyLeaf)
    assert residual_child.writes == ("p",)


def test_apply_raises_on_indivisible_residual() -> None:
    """Ancestor trip does not divide num_tiles → AtomLegalityError."""
    from nkigym.tune import AtomLegalityError

    producer = BodyLeaf(op_cls=object, phase="main", writes=("p",), dim_role={"d0": AxisRole.PARALLEL})
    consumer = BodyLeaf(op_cls=object, phase="main", reads={"data": "p"}, dim_role={"d0": AxisRole.PARALLEL})
    """num_tiles=17 with ancestor trip=3 → residual 17/3 does not divide."""
    consumer_outer = LoopNode("d0", 3, AxisRole.PARALLEL, children=[consumer])
    dims = {"d0": DimInfo(dim_id="d0", total_size=17 * 128, tile_size=128, num_tiles=17)}
    tensors = {
        "p": Tensor(
            name="p",
            dim_ids=("d0",),
            shape=(17 * 128,),
            dtype="float32",
            origin="intermediate",
            buffer_degree={"d0": 1},
        )
    }
    module = _mod_hand(body=[producer, consumer_outer], dims=dims, tensors=tensors)
    atom = ComputeAt(leaf_path=(0,), target_loop_path=(1,))
    assert atom.is_legal(module)
    try:
        atom.apply(module)
    except AtomLegalityError:
        return
    raise AssertionError("expected AtomLegalityError for indivisible residual")
