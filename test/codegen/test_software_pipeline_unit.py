"""Unit tests for SoftwarePipeline atom."""

import numpy as np
import pytest

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import BodyLeaf, LoopNode, resolve_node
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.tune.software_pipeline import SoftwarePipeline, enumerate_software_pipeline_atoms


@nkigym_kernel
def _matmul_k(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Simple matmul kernel fixture for SoftwarePipeline atom tests."""
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


def test_apply_sets_pipeline_depth(module):
    """Pick an inner LoopNode with chain length > 1 and apply its atom."""
    atoms = enumerate_software_pipeline_atoms(module)
    assert atoms, "expected at least one pipelining candidate on matmul"
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
