"""Unit tests for ReverseComputeAt atom."""

import numpy as np
import pytest

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import leaves_under
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.tune.reverse_compute_at import enumerate_reverse_compute_at_atoms


@nkigym_kernel
def _matmul_k(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Simple matmul kernel fixture for ReverseComputeAt atom tests."""
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


def test_enumerator_emits_legal_atoms(module):
    """Enumerator returns only legal atoms for the matmul fixture."""
    atoms = enumerate_reverse_compute_at_atoms(module)
    assert atoms
    for atom in atoms:
        assert atom.is_legal(module)


def test_apply_changes_tree(module):
    """Applying a legal atom rewrites the tree and preserves the leaf count."""
    atoms = enumerate_reverse_compute_at_atoms(module)
    assert atoms
    atom = atoms[0]
    new_mod = atom.apply(module)
    assert new_mod.body is not module.body
    old_leaves = sum(1 for root in module.body for _ in leaves_under(root))
    new_leaves = sum(1 for root in new_mod.body for _ in leaves_under(root))
    assert old_leaves == new_leaves
