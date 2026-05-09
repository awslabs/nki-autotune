"""Unit tests for SoftwarePipeline atom."""

import numpy as np
import pytest

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import BodyLeaf, LoopNode, resolve_node
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.tune.software_pipeline import SoftwarePipeline, enumerate_software_pipeline_atoms

M, K, N = 2048, 2048, 2048


@nkigym_kernel
def _matmul_k(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Simple matmul kernel fixture for SoftwarePipeline atom tests."""
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


@pytest.fixture
def module():
    """Build a canonical KernelModule for the matmul fixture."""
    return build_canonical_module(_matmul_k, _INPUT_SPECS)


def test_apply_sets_pipeline_depth(module):
    """Pick an inner LoopNode with chain length > 1 and apply its atom.

    NOTE: Single-phase matmul doesn't have chain_length > 1 (no separate
    init/compute/drain), so no pipelining candidates exist. Skip until
    Task 16 (RFactor) provides multi-phase matmul again.
    """
    atoms = enumerate_software_pipeline_atoms(module)
    if not atoms:
        pytest.skip("Single-phase matmul has no pipelining candidates (Task 16 RFactor)")
    atom = atoms[0]
    new_mod = atom.apply(module)
    target = resolve_node(new_mod.body, atom.loop_path)
    assert isinstance(target, LoopNode)
    assert target.pipeline_depth == atom.depth


def test_rejects_leaf_target(module):
    """A path that resolves to a BodyLeaf fails legality."""

    def _find_leaf_path(node, path):
        if isinstance(node, BodyLeaf):
            return path
        for i, c in enumerate(node.children):
            r = _find_leaf_path(c, path + (i,))
            if r is not None:
                return r
        return None

    leaf_path = None
    for i, root in enumerate(module.body):
        leaf_path = _find_leaf_path(root, (i,))
        if leaf_path is not None:
            break
    assert leaf_path is not None
    atom = SoftwarePipeline(loop_path=leaf_path, depth=2)
    assert not atom.is_legal(module)


def test_rejects_depth_zero(module):
    """Depth < 1 is always illegal."""
    atom = SoftwarePipeline(loop_path=(0,), depth=0)
    assert not atom.is_legal(module)


def test_enumerator_only_legal_atoms(module):
    """Every atom emitted by the enumerator must pass ``is_legal``."""
    for atom in enumerate_software_pipeline_atoms(module):
        assert atom.is_legal(module), f"illegal atom: {atom}"
