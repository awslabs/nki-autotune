"""Unit tests for ``nkigym.tune.batch`` — frontier-expansion sampler."""

import random

import pytest

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import BodyLeaf, KernelModule, LoopNode
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.base import AxisRole
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose
from nkigym.tune.batch import enumerate_pool, hash_state, sample_pool

_F = 256
_EPS = 1e-6


@nkigym_kernel
def _rmsnorm_matmul_f_nkigym(lhs, rhs):
    """rmsnorm + matmul fixture (first-class buffers form) — same shape as test_compile."""
    lhs_sbuf = NKIAlloc(location="sbuf", shape=(128, 256), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(256, 512), dtype="bfloat16")()
    ar_scratch = NKIAlloc(location="sbuf", shape=(128, 256), dtype="float32")()
    sum_sq = NKIAlloc(location="sbuf", shape=(128,), dtype="float32")()
    rms_inv = NKIAlloc(location="sbuf", shape=(128,), dtype="float32")()
    lhs_rms = NKIAlloc(location="sbuf", shape=(128, 256), dtype="bfloat16")()
    lhs_T_psum = NKIAlloc(location="psum", shape=(256, 128), dtype="float32")()
    lhs_T = NKIAlloc(location="sbuf", shape=(256, 128), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(128, 512), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(128, 512), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(128, 512), dtype="bfloat16")()
    NKILoad()(src=lhs, dst=lhs_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIActivationReduce(op="square", reduce_op="add")(data=lhs_sbuf, dst=ar_scratch, reduce_res=sum_sq)
    NKIActivation(op="rsqrt", scale=1.0 / _F, bias=_EPS)(data=sum_sq, dst=rms_inv)
    NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv, dst=lhs_rms)
    NKITranspose()(src=lhs_rms, dst=lhs_T_psum)
    NKITensorCopy()(src=lhs_T_psum, dst=lhs_T)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


_SPECS: dict[str, dict] = {
    "lhs": {"shape": (128, 256), "dtype": "bfloat16"},
    "rhs": {"shape": (256, 512), "dtype": "bfloat16"},
}


def _canonical_module() -> KernelModule:
    """Build the canonical :class:`KernelModule` used as the starting state."""
    return build_canonical_module(_rmsnorm_matmul_f_nkigym, _SPECS)


def test_enumerate_pool_includes_initial():
    module = _canonical_module()
    pool = enumerate_pool(module, max_pool_size=100, rng=random.Random(0))
    assert hash_state(module) in pool


def test_enumerate_pool_no_legal_atoms(monkeypatch: pytest.MonkeyPatch):
    """Starting state with no legal atoms → pool of size 1, no error."""
    from nkigym.tune import batch as batch_mod

    monkeypatch.setattr(batch_mod, "_enumerate_atoms", lambda module: [])

    module = _canonical_module()
    pool = enumerate_pool(module, max_pool_size=100, rng=random.Random(0))
    assert list(pool) == [hash_state(module)]


def test_enumerate_pool_cap_respected():
    """len(pool) == max_pool_size when reachable set exceeds the cap.

    rmsnorm+matmul has many legal atoms from the canonical state;
    capping at 3 must halt enumeration at exactly 3 pooled states.
    """
    module = _canonical_module()
    pool = enumerate_pool(module, max_pool_size=3, rng=random.Random(0))
    assert len(pool) == 3


def test_enumerate_pool_deterministic():
    """Two runs with the same seed on the same starting state produce identical pool keys."""
    module = _canonical_module()
    pool_a = enumerate_pool(module, max_pool_size=50, rng=random.Random(42))
    pool_b = enumerate_pool(module, max_pool_size=50, rng=random.Random(42))
    assert sorted(pool_a) == sorted(pool_b)


def test_enumerate_pool_exhausts_small_graph(monkeypatch: pytest.MonkeyPatch):
    """With reachable set |S|=3, cap >> |S|, two independent seeds → identical pool.

    Stubs ``_enumerate_atoms`` to simulate a tiny rewrite graph:
      s0 → s1 (atom A)
      s0 → s2 (atom B)
      s1 → s2 (atom C — reaches s2 from a different parent)
      s1/s2: no outgoing atoms
    """
    from nkigym.tune import batch as batch_mod

    def _module_with_body(tag: int) -> KernelModule:
        body = [LoopNode(dim_id=f"d{tag}", trip_count=1, role=AxisRole.PARALLEL, children=[BodyLeaf(op_cls=object)])]
        return KernelModule(func_name="t", param_names=[], return_name="", tensors={}, dims={}, body=body)

    module_s0 = _module_with_body(0)
    module_s1 = _module_with_body(1)
    module_s2 = _module_with_body(2)

    class _Atom:
        def __init__(self, dest: KernelModule) -> None:
            self.dest = dest

        def is_legal(self, module: KernelModule) -> bool:
            return True

        def apply(self, module: KernelModule) -> KernelModule:
            return self.dest

    atom_a = _Atom(module_s1)
    atom_b = _Atom(module_s2)
    atom_c = _Atom(module_s2)

    h0 = hash_state(module_s0)
    h1 = hash_state(module_s1)
    h2 = hash_state(module_s2)

    def _atoms(m: KernelModule) -> list:
        return {h0: [atom_a, atom_b], h1: [atom_c], h2: []}[hash_state(m)]

    monkeypatch.setattr(batch_mod, "_enumerate_atoms", _atoms)

    pool_a = enumerate_pool(module_s0, max_pool_size=100, rng=random.Random(0))
    pool_b = enumerate_pool(module_s0, max_pool_size=100, rng=random.Random(7))
    assert sorted(pool_a) == sorted([h0, h1, h2])
    assert sorted(pool_b) == sorted([h0, h1, h2])


def _fake_module(tag: int) -> KernelModule:
    """Build a minimal module stand-in for sample_pool tests."""
    return KernelModule(func_name=f"k{tag}", param_names=[], return_name="", tensors={}, dims={})


def test_sample_pool_exact_fill():
    """Pool of 10, N=5 → 5 distinct states."""
    pool: dict[int, KernelModule] = {i: _fake_module(i) for i in range(10)}
    out = sample_pool(pool, num_kernels=5, rng=random.Random(0))
    assert len(out) == 5
    assert len(set(id(x) for x in out)) == 5


def test_sample_pool_under_fill_warns():
    """Pool of 3, N=5 → emits UserWarning, returns all 3."""
    pool: dict[int, KernelModule] = {i: _fake_module(i) for i in range(3)}
    with pytest.warns(UserWarning, match="pool size 3 < num_kernels 5"):
        out = sample_pool(pool, num_kernels=5, rng=random.Random(0))
    assert len(out) == 3


def test_sample_pool_deterministic():
    """Fixed pool + seed → same sample (by identity)."""
    pool: dict[int, KernelModule] = {i: _fake_module(i) for i in range(20)}
    ids_a = [id(x) for x in sample_pool(pool, num_kernels=5, rng=random.Random(99))]
    ids_b = [id(x) for x in sample_pool(pool, num_kernels=5, rng=random.Random(99))]
    assert ids_a == ids_b


def test_batch_pool_contains_multi_buffer_variants() -> None:
    """Sampled pool includes states reachable via MultiBuffer.

    NOTE: Single-phase matmul doesn't create fusion opportunities that
    yield multi-buffering scenarios. Skip until Task 16 (RFactor) provides
    multi-phase matmul again.
    """
    pytest.skip("Single-phase matmul has no ComputeAt atoms yielding multi-bufferable modules (Task 16 RFactor)")
    from test.codegen._rmsnorm_matmul_fixture import INPUT_SPECS, f_nkigym

    specs_dict = {name: {"shape": shape, "dtype": dtype} for name, (shape, dtype) in INPUT_SPECS.items()}
    module = build_canonical_module(f_nkigym, specs_dict)
    rng = random.Random(0)
    pool = enumerate_pool(module, max_pool_size=200, rng=rng)

    any_mb = any(deg != 1 for m in pool.values() for t in m.tensors.values() for deg in t.buffer_degree.values())
    assert any_mb, "no MultiBuffer variants in pool"
