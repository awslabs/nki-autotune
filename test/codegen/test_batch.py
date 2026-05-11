"""Unit tests for the iter-var-IR batch sampler (7-atom enumeration).

Exercises :func:`enumerate_pool` / :func:`hash_state` / :func:`sample_pool`
end-to-end against the iter-var IR's 7 atoms (Split, Reorder, Fuse,
ComputeAt, ReverseComputeAt, RFactor, Annotate).
"""

import random

import pytest

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import DimInfo, ForNode, IterVar, KernelModule, NKIOpCall, SBlock
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.base import AxisRole
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.tune.batch import enumerate_pool, hash_state, sample_pool


@nkigym_kernel
def _matmul_large(lhs_T, rhs):
    """2048 matmul fixture — multi-tile on K/M (16) and N (4)."""
    lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(2048, 2048), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
    NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


_SPECS: dict[str, dict] = {
    "lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"},
    "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
}


def _canonical_module() -> KernelModule:
    """Build the canonical :class:`KernelModule` used as the starting state."""
    return build_canonical_module(_matmul_large, _SPECS)


def test_hash_state_canonical_is_deterministic() -> None:
    """Two identical canonical builds hash equal."""
    m1 = _canonical_module()
    m2 = _canonical_module()
    assert hash_state(m1) == hash_state(m2)


def test_enumerate_pool_includes_initial() -> None:
    """Pool always contains the starting state.

    ``hash_state`` is captured before :func:`enumerate_pool` runs because
    the :class:`Fuse` atom's ``apply`` mutates the input module's
    ``dims`` / ``fused_iter_var_map`` in place (tracked in the iter-var
    refactor follow-ups). The pool is keyed by the pre-mutation hash, so
    the starting state's key is still present even after enumeration.
    """
    module = _canonical_module()
    h0 = hash_state(module)
    pool = enumerate_pool(module, max_pool_size=50, rng=random.Random(0))
    assert h0 in pool


def test_enumerate_pool_grows_beyond_initial() -> None:
    """Pool contains >1 distinct modules after frontier expansion.

    The canonical matmul has Split / Reorder / Annotate atoms that yield
    new hashes, so the reachable set is strictly larger than the starting
    state.
    """
    module = _canonical_module()
    pool = enumerate_pool(module, max_pool_size=20, rng=random.Random(0))
    assert len(pool) > 1


def test_enumerate_pool_cap_respected() -> None:
    """``len(pool) == max_pool_size`` when reachable set exceeds the cap."""
    module = _canonical_module()
    pool = enumerate_pool(module, max_pool_size=3, rng=random.Random(0))
    assert len(pool) == 3


def test_enumerate_pool_no_legal_atoms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Starting state with no legal atoms → pool of size 1, no error."""
    from nkigym.tune import batch as batch_mod

    monkeypatch.setattr(batch_mod, "_enumerate_atoms", lambda module: [])

    module = _canonical_module()
    h0 = hash_state(module)
    pool = enumerate_pool(module, max_pool_size=100, rng=random.Random(0))
    assert list(pool) == [h0]


def test_enumerate_pool_deterministic() -> None:
    """Two runs with the same seed on equal starting states produce identical pool keys.

    Uses fresh :class:`KernelModule` instances per call: the :class:`Fuse`
    atom's ``apply`` mutates its input's ``dims`` / ``fused_iter_var_map``
    in place, so reusing a single module would feed the second call a
    different starting state.
    """
    pool_a = enumerate_pool(_canonical_module(), max_pool_size=30, rng=random.Random(42))
    pool_b = enumerate_pool(_canonical_module(), max_pool_size=30, rng=random.Random(42))
    assert sorted(pool_a) == sorted(pool_b)


def test_enumerate_pool_exhausts_small_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """With reachable set |S|=3, cap >> |S|, two independent seeds → identical pool.

    Stubs ``_enumerate_atoms`` to simulate a tiny rewrite graph:
      s0 → s1 (atom A)
      s0 → s2 (atom B)
      s1 → s2 (atom C — reaches s2 from a different parent)
      s1/s2: no outgoing atoms
    """
    from nkigym.tune import batch as batch_mod

    def _module_with_body(tag: int) -> KernelModule:
        """Build a minimal KernelModule whose subtree_signature varies by ``tag``."""
        iv = IterVar(var_id=tag, dim_id=f"d{tag}", extent=tag + 1, role=AxisRole.PARALLEL)
        stub_block = SBlock(
            iter_vars=[iv],
            reads={},
            writes={},
            reads_writes={},
            body=[NKIOpCall(op_cls=object, kwargs={}, axis_map={}, dim_role={})],
        )
        body = [ForNode(iter_var=iv, children=[stub_block])]
        return KernelModule(
            func_name=f"t_{tag}",
            param_names=[],
            return_name="",
            tensors={},
            dims={f"d{tag}": DimInfo(dim_id=f"d{tag}", total_size=tag + 1)},
            body=body,
        )

    module_s0 = _module_with_body(0)
    module_s1 = _module_with_body(1)
    module_s2 = _module_with_body(2)

    class _Atom:
        """Fake atom: legal everywhere, applies to a fixed destination."""

        def __init__(self, dest: KernelModule) -> None:
            self.dest = dest

        def is_legal(self, module: KernelModule) -> bool:
            _ = module
            return True

        def apply(self, module: KernelModule) -> KernelModule:
            _ = module
            return self.dest

    atom_a = _Atom(module_s1)
    atom_b = _Atom(module_s2)
    atom_c = _Atom(module_s2)

    h0 = hash_state(module_s0)
    h1 = hash_state(module_s1)
    h2 = hash_state(module_s2)

    def _atoms(m: KernelModule) -> list:
        """Dispatch fake enumeration by hash of ``m``."""
        return {h0: [atom_a, atom_b], h1: [atom_c], h2: []}[hash_state(m)]

    monkeypatch.setattr(batch_mod, "_enumerate_atoms", _atoms)

    pool_a = enumerate_pool(module_s0, max_pool_size=100, rng=random.Random(0))
    pool_b = enumerate_pool(module_s0, max_pool_size=100, rng=random.Random(7))
    assert sorted(pool_a) == sorted([h0, h1, h2])
    assert sorted(pool_b) == sorted([h0, h1, h2])


def _fake_module(tag: int) -> KernelModule:
    """Build a minimal module stand-in for sample_pool tests."""
    return KernelModule(func_name=f"k{tag}", param_names=[], return_name="", tensors={}, dims={})


def test_sample_pool_exact_fill() -> None:
    """Pool of 10, N=5 → 5 distinct states."""
    pool: dict[int, KernelModule] = {i: _fake_module(i) for i in range(10)}
    out = sample_pool(pool, num_kernels=5, rng=random.Random(0))
    assert len(out) == 5
    assert len({id(x) for x in out}) == 5


def test_sample_pool_under_fill_warns() -> None:
    """Pool of 3, N=5 → emits UserWarning, returns all 3."""
    pool: dict[int, KernelModule] = {i: _fake_module(i) for i in range(3)}
    with pytest.warns(UserWarning, match="pool size 3 < num_kernels 5"):
        out = sample_pool(pool, num_kernels=5, rng=random.Random(0))
    assert len(out) == 3


def test_sample_pool_deterministic() -> None:
    """Fixed pool + seed → same sample (by identity)."""
    pool: dict[int, KernelModule] = {i: _fake_module(i) for i in range(20)}
    ids_a = [id(x) for x in sample_pool(pool, num_kernels=5, rng=random.Random(99))]
    ids_b = [id(x) for x in sample_pool(pool, num_kernels=5, rng=random.Random(99))]
    assert ids_a == ids_b


def test_enumerate_pool_includes_annotate_buffer_degree_variants() -> None:
    """Pool contains at least one state with a non-default ``buffer_degree``.

    :class:`Annotate` on ``buffer_degree`` is an enumerated atom; the
    sampler must be able to reach states where some alloc SBlock carries
    a ``buffer_degree`` annotation in ``block.annotations``.
    """
    module = _canonical_module()
    pool = enumerate_pool(module, max_pool_size=60, rng=random.Random(0))

    def _has_buffer_degree_annotation(m: KernelModule) -> bool:
        for root in m.body:
            if isinstance(root, SBlock) and "buffer_degree" in root.annotations:
                return True
        return False

    assert any(
        _has_buffer_degree_annotation(m) for m in pool.values()
    ), "pool does not contain any Annotate(buffer_degree) state"
