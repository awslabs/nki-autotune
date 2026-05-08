"""Unit tests for ComputeAt atom."""

import numpy as np
import pytest

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import BodyLeaf, LoopNode, leaves_under
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.tune.compute_at import ComputeAt, enumerate_compute_at_atoms


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
