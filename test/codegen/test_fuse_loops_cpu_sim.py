"""CPU-sim correctness gate for topologically-adjacent FuseLoops.

Renders each post-fuse kernel to source via the existing codegen
pipeline, runs it through ``nki.simulate``, and compares against the
numpy reference at fp32 tolerance. The standard nkigym validation
contract: ``atol=rtol=5e-3`` matching ``compile.py`` CPU-sim checks.
"""

import numpy as np
import pytest

from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.loop_forest import build_canonical_forest
from nkigym.codegen.render import render
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose
from nkigym.tune.fuse_loops import FuseLoops, enumerate_fusion_atoms

try:
    import nki
except ImportError:
    nki = None

_SEED = 0
_ATOL = 5e-3
_RTOL = 5e-3


@nkigym_kernel
def _rmsnorm_matmul(lhs, rhs):
    lhs_sbuf = NKILoad()(data=lhs)
    rhs_sbuf = NKILoad()(data=rhs)
    sum_sq = NKIActivationReduce(op="square", reduce_op="add")(data=lhs_sbuf)
    rms_inv = NKIActivation(op="rsqrt", scale=1 / 256, bias=1e-6)(data=sum_sq)
    lhs_rms = NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv)
    lhs_T = NKITranspose()(data=lhs_rms)
    prod = NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out


def _rmsnorm_matmul_numpy(lhs, rhs):
    m = np.mean(np.square(lhs.astype(np.float32)), axis=1, keepdims=True)
    rms_inv = 1.0 / np.sqrt(m + 1e-6)
    normed = lhs.astype(np.float32) * rms_inv
    return normed @ rhs.astype(np.float32)


_SPECS = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}


@nkigym_kernel
def _matmul_lhsT_rhs(lhs, rhs):
    lhs_sbuf = NKILoad()(data=lhs)
    rhs_sbuf = NKILoad()(data=rhs)
    lhs_T = NKITranspose()(data=lhs_sbuf)
    prod = NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out


def _matmul_lhsT_rhs_numpy(lhs, rhs):
    return (lhs.astype(np.float32)) @ (rhs.astype(np.float32))


def _cpu_sim(kernel_source: str, func_name: str, inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Execute ``kernel_source`` under ``nki.simulate`` and return its output array.

    Matches the fp32 contract from ``nkigym.compile._cpu_sim_check``:
    rewrite bf16/fp16 dtypes to fp32 throughout the rendered source.
    """
    sim_source = kernel_source.replace("nl.bfloat16", "nl.float32").replace("nl.float16", "nl.float32")
    ns: dict = {}
    exec(sim_source, ns)  # noqa: S102
    actual = nki.simulate(ns[func_name])(**inputs)
    if isinstance(actual, tuple):
        actual = actual[0]
    return actual


def _fp32_inputs(specs):
    rng = np.random.default_rng(_SEED)
    return {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _) in specs.items()}


@pytest.mark.skipif(nki is None, reason="nki runtime not available")
def test_rmsnorm_matmul_topological_fuse_cpu_sim_matches_numpy() -> None:
    """Apply FuseLoops(path=(), boundary=(0, 2), dim_id='d0') — Load(lhs) ↔ ActivationReduce —
    and confirm CPU-sim output matches the numpy reference within fp32 tolerance.
    """
    g = parse_and_resolve(_rmsnorm_matmul, _SPECS)
    forest = build_canonical_forest(g)
    atom = FuseLoops(path=(), boundary=(0, 2), dim_id="d0")
    assert atom.is_legal(g, forest) is True
    _, new_forest = atom.apply(g, forest)
    source = render(g, new_forest)
    inputs = _fp32_inputs(_SPECS)
    actual = _cpu_sim(source, g.func_name, inputs)
    expected = _rmsnorm_matmul_numpy(**inputs)
    assert np.allclose(
        actual, expected, atol=_ATOL, rtol=_RTOL
    ), f"CPU-sim mismatch: max_abs_diff={float(np.abs(actual - expected).max()):.3e}"


@pytest.mark.skipif(nki is None, reason="nki runtime not available")
def test_matmul_lhsT_rhs_topological_fuse_cpu_sim_matches_numpy() -> None:
    """Apply a topological fuse on pure matmul where independent Load(rhs) is between endpoints.

    The canonical forest for matmul_lhsT_rhs has a complex multi-level structure
    where root siblings 0 and 2 do NOT share the same d0 trip count (Transpose's
    d0 is swapped d1 from Load), so we need to find a legal pair. We enumerate
    atoms and take the first topological (non-adjacent) one if it exists, or skip.
    """
    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 128), "bfloat16")}
    g = parse_and_resolve(_matmul_lhsT_rhs, specs)
    forest = build_canonical_forest(g)
    atoms = [a for a in enumerate_fusion_atoms(g, forest) if a.path == () and a.boundary[1] > a.boundary[0] + 1]
    if not atoms:
        pytest.skip("No topological-adjacency atoms found in this forest structure")
    atom = atoms[0]
    _, new_forest = atom.apply(g, forest)
    source = render(g, new_forest)
    inputs = _fp32_inputs(specs)
    actual = _cpu_sim(source, g.func_name, inputs)
    expected = _matmul_lhsT_rhs_numpy(**inputs)
    assert np.allclose(actual, expected, atol=_ATOL, rtol=_RTOL)


@pytest.mark.skipif(nki is None, reason="nki runtime not available")
def test_chained_topological_then_literal_fuse_cpu_sim_matches_numpy() -> None:
    """Apply topological fuse (0, 2) then a second fuse on the post-fuse forest.

    After step 1 the forest has one fewer nest; re-enumerate to find a
    literal-adjacent d0 pair and apply it. The final kernel must
    still match the numpy reference.
    """
    g = parse_and_resolve(_rmsnorm_matmul, _SPECS)
    forest = build_canonical_forest(g)
    atom0 = FuseLoops(path=(), boundary=(0, 2), dim_id="d0")
    _, forest1 = atom0.apply(g, forest)
    next_atoms = [
        a
        for a in enumerate_fusion_atoms(g, forest1)
        if a.path == () and a.boundary[1] == a.boundary[0] + 1 and a.dim_id == "d0"
    ]
    assert next_atoms, "expected a literal-adjacent d0 atom after the topological fuse"
    _, forest2 = next_atoms[0].apply(g, forest1)
    source = render(g, forest2)
    inputs = _fp32_inputs(_SPECS)
    actual = _cpu_sim(source, g.func_name, inputs)
    expected = _rmsnorm_matmul_numpy(**inputs)
    assert np.allclose(actual, expected, atol=_ATOL, rtol=_RTOL)


def test_enumerator_refuses_pair_blocked_by_raw_intervening_dependency() -> None:
    """Negative control: on the canonical rmsnorm+matmul forest, the fuse (2, 4) is blocked.

    Sibling 3 (Activation) has a RAW edge with sibling 2 (ActivationReduce)
    via ``sum_sq`` — sibling 3 cannot pass the producer. The enumerator
    must not emit (2, 4) among root-level atoms.
    """
    g = parse_and_resolve(_rmsnorm_matmul, _SPECS)
    forest = build_canonical_forest(g)
    atoms = enumerate_fusion_atoms(g, forest)
    root_atoms = {a.boundary for a in atoms if a.path == ()}
    assert (2, 4) not in root_atoms
