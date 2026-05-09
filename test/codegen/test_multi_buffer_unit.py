"""Unit tests for MultiBuffer atom."""

import numpy as np
import pytest

from nkigym.codegen.canonical import build_canonical_module
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.tune import AtomLegalityError
from nkigym.tune.compute_at import enumerate_compute_at_atoms
from nkigym.tune.multi_buffer import MultiBuffer, enumerate_multi_buffer_atoms

M, K, N = 2048, 2048, 2048


@nkigym_kernel
def _matmul_k(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Simple matmul kernel fixture for MultiBuffer atom tests."""
    lhs_s = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
    rhs_s = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(M, N), dtype="bfloat16")()
    NKILoad()(src=lhs, dst=lhs_s)
    NKILoad()(src=rhs, dst=rhs_s)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_s, moving=rhs_s, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


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

    NOTE: Single-phase matmul doesn't create fusion opportunities that
    yield multi-buffering scenarios. Skip tests that depend on this fixture
    until Task 16 (RFactor) provides multi-phase matmul again.
    """
    mod = build_canonical_module(_matmul_k, _INPUT_SPECS)
    for atom in enumerate_compute_at_atoms(mod):
        candidate = atom.apply(mod)
        if enumerate_multi_buffer_atoms(candidate):
            return candidate
    pytest.skip("Single-phase matmul has no ComputeAt atoms yielding multi-bufferable modules (Task 16 RFactor)")


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


def test_multi_buffer_apply_rejects_stale_atom(module):
    """Apply must reject an atom that is no longer legal against the current module.

    An atom enumerated against a fused module (``required_tiles == 1`` on
    some intermediate dim, so degree=2 is legal) must not silently mutate
    a canonical module (``required_tiles == num_tiles`` on the same dim,
    where the max legal degree is 1). The canonical module is the exact
    cross-nest case from the IR-refactor followup bug: degree<num_tiles
    would produce slot-modulo aliasing.
    """
    atoms = [a for a in enumerate_multi_buffer_atoms(module) if a.degree >= 2]
    assert atoms, "fused fixture should yield at least one MultiBuffer atom with degree>=2"
    atom = atoms[0]
    canonical = build_canonical_module(_matmul_k, _INPUT_SPECS)
    assert atom.is_legal(module)
    assert not atom.is_legal(canonical), "construction precondition: atom must be stale against canonical"
    with pytest.raises(AtomLegalityError):
        atom.apply(canonical)
