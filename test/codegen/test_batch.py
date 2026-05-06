"""Unit tests for ``nkigym.tune.batch`` — frontier-expansion sampler."""

import random

import pytest

from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.loop_forest import build_canonical_forest, hash_forest
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose
from nkigym.tune.batch import enumerate_pool, sample_pool


@nkigym_kernel
def _rmsnorm_matmul_f_nkigym(lhs, rhs):
    """rmsnorm + matmul fixture reused across tests — same shape as test_compile."""
    lhs_sbuf = NKILoad()(data=lhs)
    rhs_sbuf = NKILoad()(data=rhs)
    sum_sq = NKIActivationReduce(op="square", reduce_op="add")(data=lhs_sbuf)
    rms_inv = NKIActivation(op="rsqrt", scale=1 / 256, bias=1e-6)(data=sum_sq)
    lhs_rms = NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv)
    lhs_T = NKITranspose()(data=lhs_rms)
    prod = NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out


_SPECS: dict[str, tuple[tuple[int, ...], str]] = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}


def _canonical_state() -> tuple:
    """Build the canonical (op_graph, forest) used as the starting state."""
    op_graph = parse_and_resolve(_rmsnorm_matmul_f_nkigym, _SPECS)
    forest = build_canonical_forest(op_graph)
    return op_graph, forest


def test_enumerate_pool_includes_initial():
    op_graph, forest = _canonical_state()
    pool = enumerate_pool(op_graph, forest, max_pool_size=100, rng=random.Random(0))
    assert hash_forest(forest) in pool


def test_enumerate_pool_no_legal_atoms(monkeypatch: pytest.MonkeyPatch):
    """Starting state with no legal atoms → pool of size 1, no error."""
    from nkigym.tune import batch as batch_mod

    monkeypatch.setattr(batch_mod, "enumerate_fusion_atoms", lambda og, f: [])
    monkeypatch.setattr(batch_mod, "enumerate_reorder_atoms", lambda f: [])

    op_graph, forest = _canonical_state()
    pool = enumerate_pool(op_graph, forest, max_pool_size=100, rng=random.Random(0))
    assert list(pool) == [hash_forest(forest)]


def test_enumerate_pool_cap_respected():
    """len(pool) == max_pool_size when reachable set exceeds the cap.

    rmsnorm+matmul has many legal atoms from the canonical state;
    capping at 3 must halt enumeration at exactly 3 pooled states.
    """
    op_graph, forest = _canonical_state()
    pool = enumerate_pool(op_graph, forest, max_pool_size=3, rng=random.Random(0))
    assert len(pool) == 3


def test_enumerate_pool_deterministic():
    """Two runs with the same seed on the same starting state produce identical pool keys."""
    op_graph, forest = _canonical_state()
    pool_a = enumerate_pool(op_graph, forest, max_pool_size=50, rng=random.Random(42))
    pool_b = enumerate_pool(op_graph, forest, max_pool_size=50, rng=random.Random(42))
    assert sorted(pool_a) == sorted(pool_b)


def test_enumerate_pool_exhausts_small_graph(monkeypatch: pytest.MonkeyPatch):
    """With reachable set |S|=3, cap >> |S|, two independent seeds → identical pool.

    Stubs atom enumerators to simulate a tiny rewrite graph:
      s0 → s1 (atom A)
      s0 → s2 (atom B)
      s1 → s2 (atom C — reaches s2 from a different parent)
      s1/s2: no outgoing atoms
    """
    from nkigym.codegen.loop_forest import BodyLeaf, LoopForest, LoopNode
    from nkigym.ops.base import AxisRole
    from nkigym.tune import batch as batch_mod

    def _forest(tag: int) -> LoopForest:
        return [LoopNode(dim_id=f"d{tag}", trip_count=1, role=AxisRole.PARALLEL, children=[BodyLeaf(op_idx=0)])]

    op_graph = object()
    forest_s0 = _forest(0)
    forest_s1 = _forest(1)
    forest_s2 = _forest(2)

    class _Atom:
        def __init__(self, dest: LoopForest) -> None:
            self.dest = dest

        def is_legal(self, og, f):
            return True

        def apply(self, og, f):
            return og, self.dest

    atom_a = _Atom(forest_s1)
    atom_b = _Atom(forest_s2)
    atom_c = _Atom(forest_s2)

    h0 = hash_forest(forest_s0)
    h1 = hash_forest(forest_s1)
    h2 = hash_forest(forest_s2)

    def _fusion(og, f):
        return {h0: [atom_a], h1: [atom_c], h2: []}[hash_forest(f)]

    def _reorder(f):
        return {h0: [atom_b], h1: [], h2: []}[hash_forest(f)]

    monkeypatch.setattr(batch_mod, "enumerate_fusion_atoms", _fusion)
    monkeypatch.setattr(batch_mod, "enumerate_reorder_atoms", _reorder)

    pool_a = enumerate_pool(op_graph, forest_s0, max_pool_size=100, rng=random.Random(0))
    pool_b = enumerate_pool(op_graph, forest_s0, max_pool_size=100, rng=random.Random(7))
    assert sorted(pool_a) == sorted([h0, h1, h2])
    assert sorted(pool_b) == sorted([h0, h1, h2])


def test_sample_pool_exact_fill():
    """Pool of 10, N=5 → 5 distinct states."""
    pool: dict[int, tuple] = {i: (f"og{i}", f"f{i}") for i in range(10)}
    out = sample_pool(pool, num_kernels=5, rng=random.Random(0))
    assert len(out) == 5
    assert len(set(id(x) for x in out)) == 5


def test_sample_pool_under_fill_warns():
    """Pool of 3, N=5 → emits UserWarning, returns all 3."""
    pool: dict[int, tuple] = {i: (f"og{i}", f"f{i}") for i in range(3)}
    with pytest.warns(UserWarning, match="pool size 3 < num_kernels 5"):
        out = sample_pool(pool, num_kernels=5, rng=random.Random(0))
    assert len(out) == 3


def test_sample_pool_deterministic():
    """Fixed pool + seed → same sample (by value equality)."""
    pool: dict[int, tuple] = {i: (f"og{i}", f"f{i}") for i in range(20)}
    out_a = sample_pool(pool, num_kernels=5, rng=random.Random(99))
    out_b = sample_pool(pool, num_kernels=5, rng=random.Random(99))
    assert out_a == out_b
