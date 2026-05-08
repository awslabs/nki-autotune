"""Unit tests for MultiBuffer atom."""

import numpy as np
import pytest

from nkigym.codegen.canonical import build_canonical_module
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.tune.compute_at import enumerate_compute_at_atoms
from nkigym.tune.multi_buffer import MultiBuffer, enumerate_multi_buffer_atoms


@nkigym_kernel
def _matmul_k(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Simple matmul kernel fixture for MultiBuffer atom tests."""
    lhs_s = NKILoad()(data=lhs)
    rhs_s = NKILoad()(data=rhs)
    out_s = NKIMatmul()(stationary=lhs_s, moving=rhs_s)
    out = NKIStore()(data=out_s)
    return out


_INPUT_SPECS: dict[str, dict] = {
    "lhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
    "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
}


def _fused_module():
    """Return a module with at least one intermediate fused under a consumer loop.

    Canonical build produces separate root trees per op, making
    producer/consumer LCA the forest root and required_tiles equal to
    num_tiles — no multi-buffering is then useful. Pick the first
    ComputeAt atom that produces a module where an intermediate's LCA
    sits below a tile-iterating ancestor, so MultiBuffer enumeration is
    non-empty.
    """
    mod = build_canonical_module(_matmul_k, _INPUT_SPECS)
    for atom in enumerate_compute_at_atoms(mod):
        candidate = atom.apply(mod)
        if enumerate_multi_buffer_atoms(candidate):
            return candidate
    raise RuntimeError("no ComputeAt atom yielded a multi-bufferable module")


@pytest.fixture
def module():
    return _fused_module()


def test_multi_buffer_mutates_tensor_degree(module):
    atoms = enumerate_multi_buffer_atoms(module)
    assert atoms, "fused fixture should yield at least one legal MultiBuffer atom"
    atom = atoms[0]
    new_mod = atom.apply(module)
    assert new_mod.tensors[atom.tensor_name].buffer_degree[atom.dim_id] == atom.degree
    assert module.tensors[atom.tensor_name].buffer_degree[atom.dim_id] == 1


def test_multi_buffer_rejects_unknown_tensor(module):
    atom = MultiBuffer(tensor_name="nonexistent", dim_id="d0", degree=2)
    assert not atom.is_legal(module)


def test_multi_buffer_rejects_out_of_range_degree(module):
    intermediate = next(n for n, t in module.tensors.items() if t.origin == "intermediate")
    d = module.tensors[intermediate].dim_ids[0]
    num_t = module.dims[d].num_tiles
    high = MultiBuffer(tensor_name=intermediate, dim_id=d, degree=num_t + 1)
    assert not high.is_legal(module)
    low = MultiBuffer(tensor_name=intermediate, dim_id=d, degree=0)
    assert not low.is_legal(module)


def test_enumerator_yields_only_legal_atoms(module):
    atoms = enumerate_multi_buffer_atoms(module)
    assert atoms
    for atom in atoms:
        assert atom.is_legal(module), f"illegal atom: {atom}"
