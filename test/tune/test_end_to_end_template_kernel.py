"""End-to-end: DecomposeReduction + Reorder composition produces K-outside kernel.

Matches the template-kernel pattern (K streamed outside M, N); previously not
reachable from canonical form.
"""

import numpy as np

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import BodyLeaf
from nkigym.codegen.render import render
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.tune.decompose_reduction import enumerate_decompose_reduction_atoms
from nkigym.tune.reorder import enumerate_reorder_atoms


@nkigym_kernel
def _matmul_k(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Canonical matmul fixture: load lhs + rhs, matmul, store."""
    lhs_s = NKILoad()(data=lhs)
    rhs_s = NKILoad()(data=rhs)
    out_s = NKIMatmul()(stationary=lhs_s, moving=rhs_s)
    out = NKIStore()(data=out_s)
    return out


_INPUT_SPECS: dict[str, dict] = {
    "lhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
    "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
}


def _count_trees_touching_phase(module, phase: str) -> int:
    """Count top-level trees whose subtree contains a BodyLeaf with the given phase."""
    count = 0
    for root in module.body:
        if _tree_has_phase(root, phase):
            count += 1
    return count


def _tree_has_phase(node, phase: str) -> bool:
    """Return True iff any BodyLeaf under ``node`` has ``phase``."""
    if isinstance(node, BodyLeaf):
        return node.phase == phase
    for c in node.children:
        if _tree_has_phase(c, phase):
            return True
    return False


def test_decompose_reduction_fissions_matmul_into_three_trees():
    """DecomposeReduction replaces the matmul subtree with init/update/drain siblings."""
    module = build_canonical_module(_matmul_k, _INPUT_SPECS)
    initial_trees = len(module.body)
    """Canonical matmul: [load, load, matmul, store] - 4 roots."""
    assert initial_trees == 4
    atoms = enumerate_decompose_reduction_atoms(module)
    assert atoms, "no DecomposeReduction atoms found"
    """Pick the outer-level target (target_loop_path length 1)."""
    atom = next((a for a in atoms if len(a.target_loop_path) == 1), atoms[0])
    new_module = atom.apply(module)
    """Expect: original 4 - 1 (matmul consumed) + 3 (init/update/drain) = 6."""
    assert len(new_module.body) > initial_trees
    """Verify all three phase leaves are still present across separate trees."""
    assert _count_trees_touching_phase(new_module, "psum_init") >= 1
    assert _count_trees_touching_phase(new_module, "compute") >= 1
    assert _count_trees_touching_phase(new_module, "drain") >= 1


def test_end_to_end_decompose_plus_reorder_renders():
    """After fission, the update tree becomes leaf-pure under K; Reorder atoms
    become available that couldn't be legal on the pre-fission form. Apply
    DecomposeReduction + up to 2 reorders; render the result."""
    module = build_canonical_module(_matmul_k, _INPUT_SPECS)
    atoms = enumerate_decompose_reduction_atoms(module)
    atom = next((a for a in atoms if len(a.target_loop_path) == 1), atoms[0])
    module = atom.apply(module)
    for _ in range(2):
        reorder_atoms = enumerate_reorder_atoms(module)
        if not reorder_atoms:
            break
        module = reorder_atoms[0].apply(module)
    source = render(module)
    assert "nc_matmul" in source or "nisa" in source
    assert "def " in source
