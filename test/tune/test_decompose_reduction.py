"""Unit tests for DecomposeReduction atom."""

import numpy as np
import pytest

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import BodyLeaf
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.tune.decompose_reduction import DecomposeReduction, enumerate_decompose_reduction_atoms


@nkigym_kernel
def _matmul_k(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Simple matmul kernel fixture for DecomposeReduction atom tests."""
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


def _path_to_leaf(node, phase, op_cls, path):
    """Return the path to the first matching BodyLeaf, or None."""
    if isinstance(node, BodyLeaf):
        if node.op_cls is op_cls and node.phase == phase:
            return path
        return None
    for i, c in enumerate(node.children):
        r = _path_to_leaf(c, phase, op_cls, path + (i,))
        if r is not None:
            return r
    return None


def test_decompose_reduction_produces_three_trees_from_matmul(module):
    """Applying DecomposeReduction to the matmul tree replaces the single
    subtree with three sibling trees (init + update + drain), growing body
    length."""
    compute_path = None
    for i, root in enumerate(module.body):
        compute_path = _path_to_leaf(root, "compute", NKIMatmul, (i,))
        if compute_path is not None:
            break
    assert compute_path is not None
    target_loop_path = (compute_path[0],)
    atom = DecomposeReduction(leaf_path=compute_path, target_loop_path=target_loop_path)
    assert atom.is_legal(module)
    new_module = atom.apply(module)
    assert len(new_module.body) > len(module.body)


def test_rejects_non_reducer_leaf(module):
    """A non-reducer leaf (e.g. NKILoad) should not be a legal target."""
    main_path = None
    for i, root in enumerate(module.body):
        main_path = _path_to_leaf(root, "main", NKILoad, (i,))
        if main_path is not None:
            break
    assert main_path is not None
    target_loop_path = (main_path[0],)
    atom = DecomposeReduction(leaf_path=main_path, target_loop_path=target_loop_path)
    assert not atom.is_legal(module)


def test_enumerator_emits_reducer_targets(module):
    """Enumerator emits legal atoms, all reducing-phase leaf targets."""
    atoms = enumerate_decompose_reduction_atoms(module)
    assert atoms
    for atom in atoms:
        assert atom.is_legal(module)
