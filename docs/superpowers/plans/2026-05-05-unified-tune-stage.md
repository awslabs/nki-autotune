# Unified Tune Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current single-kernel random-draw tune path with a frontier-expansion graph sampler that produces N distinct kernels per call, all CPU-sim-checked + HW-profiled via `autotune.remote_profile`, and tighten `nkigym_compile`'s public signature.

**Architecture:** Two pure functions in `nkigym/tune/batch.py` (`enumerate_pool`, `sample_pool`) power a new `_run_batch` branch inside `nkigym/tune/stage.py`. `nkigym_compile` loses its `stages` kwarg and always runs synthesis → initial_codegen → tune; the tune stage dispatches on `rewrites` (None = batch, list = explicit/back-compat). CPU-sim failures surface by reading the `cpu_sim_passed` field in `results.json` after `remote_profile` returns.

**Tech Stack:** Python 3.12, pytest, numpy, `nkigym.codegen.loop_forest`, `nkigym.tune.{fuse_loops, reorder_loops}`, `autotune.runner.api.remote_profile`, `autotune.runner.types.KernelJob`. Kernel venv at `~/venvs/kernel-env/bin/python`.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `nkigym/src/nkigym/tune/batch.py` | **create** | Pure `enumerate_pool` + `sample_pool` — no I/O, no rendering. |
| `nkigym/src/nkigym/tune/stage.py` | modify | Add `_run_batch` branch. Keep explicit-rewrites branch byte-equivalent. |
| `nkigym/src/nkigym/compile.py` | modify | Drop `stages` kwarg, make all three stages run unconditionally, forward new batch kwargs to `run_tune`, add `_assert_no_cpu_sim_failures` + `_trace_output_shape` helpers. |
| `test/codegen/test_batch.py` | **create** | Unit tests for both batch functions. |
| `test/codegen/test_compile.py` | modify | Drop tests that assert on `stages` kwarg; migrate `rewrites=[...]` tests to the new signature; add 2 batch-path integration tests with `remote_profile` mocked. |
| `examples/rmsnorm_matmul.py` | modify | Drop `stages=...`; add batch kwargs. |

**Out of scope** (spec §Out of scope): `UnfuseLoops`, new rewrite atoms, configurable `100×` multiplier, per-edge atom dedup, return-type change.

---

### Task 1: Scaffold `nkigym/tune/batch.py` with failing tests

**Files:**
- Create: `nkigym/src/nkigym/tune/batch.py`
- Create: `test/codegen/test_batch.py`

- [ ] **Step 1: Write the failing tests (module-level structure only)**

Create `test/codegen/test_batch.py`:

```python
"""Unit tests for ``nkigym.tune.batch`` — frontier-expansion sampler."""

import random
import warnings
from collections.abc import Callable

import pytest

from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.loop_forest import build_canonical_forest, hash_forest
from nkigym.ops import nkigym_kernel
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
    rms_inv = NKIActivationReduce(
        op="square", reduce_op="add", post_op="rsqrt", scale=1 / 256, bias=1e-6
    )(data=lhs_sbuf)
    lhs_rms = NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv)
    lhs_T = NKITranspose()(data=lhs_rms)
    prod = NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out


_SPECS: dict[str, tuple[tuple[int, ...], str]] = {
    "lhs": ((128, 256), "bfloat16"),
    "rhs": ((256, 512), "bfloat16"),
}


def _canonical_state() -> tuple:
    """Build the canonical (op_graph, forest) used as the starting state."""
    op_graph = parse_and_resolve(_rmsnorm_matmul_f_nkigym, _SPECS)
    forest = build_canonical_forest(op_graph)
    return op_graph, forest


def test_enumerate_pool_includes_initial():
    op_graph, forest = _canonical_state()
    pool = enumerate_pool(op_graph, forest, max_pool_size=100, rng=random.Random(0))
    assert hash_forest(forest) in pool
```

- [ ] **Step 2: Run test to verify it fails**

```
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/test_batch.py -x -q
```

Expected: `ModuleNotFoundError: No module named 'nkigym.tune.batch'`.

- [ ] **Step 3: Create `nkigym/tune/batch.py` with a function stub that still fails the test**

Create `nkigym/src/nkigym/tune/batch.py`:

```python
"""Frontier-expansion sampler for the tune stage batch path.

Two pure functions — no I/O, no rendering, no dependency on
``autotune`` — so the sampler can be unit-tested in isolation:

* :func:`enumerate_pool` explores the reachable rewrite graph via
  randomized frontier expansion, deduping states by
  :func:`hash_forest`.
* :func:`sample_pool` draws ``num_kernels`` states uniformly
  without replacement from the enumerated pool.

See ``docs/superpowers/specs/2026-05-05-unified-tune-stage-design.md``
for the algorithm rationale.
"""

import random
import warnings

from nkigym.codegen.graph import OpGraph
from nkigym.codegen.loop_forest import LoopForest, hash_forest
from nkigym.tune import KernelRewrite
from nkigym.tune.fuse_loops import enumerate_fusion_atoms
from nkigym.tune.reorder_loops import enumerate_reorder_atoms


def enumerate_pool(
    op_graph: OpGraph,
    forest: LoopForest,
    max_pool_size: int,
    rng: random.Random,
) -> dict[int, tuple[OpGraph, LoopForest]]:
    """Stub — fails intentionally to drive TDD."""
    raise NotImplementedError


def sample_pool(
    pool: dict[int, tuple[OpGraph, LoopForest]],
    num_kernels: int,
    rng: random.Random,
) -> list[tuple[OpGraph, LoopForest]]:
    """Stub — fails intentionally to drive TDD."""
    raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it fails on NotImplementedError**

```
pytest test/codegen/test_batch.py::test_enumerate_pool_includes_initial -x -q
```

Expected: `NotImplementedError` raised from `enumerate_pool`.

- [ ] **Step 5: Commit scaffolding**

```bash
git add nkigym/src/nkigym/tune/batch.py test/codegen/test_batch.py
git commit -m "Scaffold tune batch module with failing pool test"
```

---

### Task 2: Implement `enumerate_pool` minimal happy path

**Files:**
- Modify: `nkigym/src/nkigym/tune/batch.py`

- [ ] **Step 1: Replace the `enumerate_pool` stub with the real implementation**

Open `nkigym/src/nkigym/tune/batch.py`. Replace the `enumerate_pool` body:

```python
def enumerate_pool(
    op_graph: OpGraph,
    forest: LoopForest,
    max_pool_size: int,
    rng: random.Random,
) -> dict[int, tuple[OpGraph, LoopForest]]:
    """Enumerate the reachable rewrite graph via randomized frontier expansion.

    Maintains a frontier: pool nodes that still have un-tried outgoing
    atoms. Each iteration picks one frontier node uniformly, pops one
    of its unexplored atoms uniformly, applies it, and adds the
    destination to the pool when the hash is new. Terminates when
    ``len(pool) >= max_pool_size`` OR the frontier is empty.

    The per-node atom list is snapshot at pool-insertion time and
    never re-enumerated — safe because atom enumerators are pure
    functions of the forest and frontier nodes are immutable
    ``(op_graph, forest)`` tuples.

    Args:
        op_graph: Starting ``OpGraph``.
        forest: Starting ``LoopForest``.
        max_pool_size: Stop when the pool reaches this size.
        rng: Seeded ``random.Random`` — drives frontier-node pick and
            atom pick.

    Returns:
        Dict keyed by ``hash_forest``; values are ``(op_graph, forest)``
        tuples. Always contains the starting state.
    """
    h0 = hash_forest(forest)
    pool: dict[int, tuple[OpGraph, LoopForest]] = {h0: (op_graph, forest)}
    frontier: dict[int, list[KernelRewrite]] = {
        h0: enumerate_fusion_atoms(forest) + enumerate_reorder_atoms(forest)
    }
    if not frontier[h0]:
        del frontier[h0]

    while frontier and len(pool) < max_pool_size:
        frontier_keys = list(frontier)
        h = rng.choice(frontier_keys)
        atoms = frontier[h]
        j = rng.randrange(len(atoms))
        atoms[j], atoms[-1] = atoms[-1], atoms[j]
        atom = atoms.pop()
        if not atoms:
            del frontier[h]

        src_og, src_f = pool[h]
        new_og, new_f = atom.apply(src_og, src_f)
        h_new = hash_forest(new_f)
        if h_new in pool:
            continue
        pool[h_new] = (new_og, new_f)
        new_atoms = enumerate_fusion_atoms(new_f) + enumerate_reorder_atoms(new_f)
        if new_atoms:
            frontier[h_new] = new_atoms

    return pool
```

- [ ] **Step 2: Run the existing test**

```
pytest test/codegen/test_batch.py::test_enumerate_pool_includes_initial -x -q
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add nkigym/src/nkigym/tune/batch.py
git commit -m "Implement enumerate_pool with frontier expansion"
```

---

### Task 3: `enumerate_pool` — no-legal-atoms edge case

**Files:**
- Modify: `test/codegen/test_batch.py`

- [ ] **Step 1: Add the no-legal-atoms test**

Append to `test/codegen/test_batch.py`:

```python
def test_enumerate_pool_no_legal_atoms(monkeypatch: pytest.MonkeyPatch):
    """Starting state with no legal atoms → pool of size 1, no error."""
    from nkigym.tune import batch as batch_mod

    monkeypatch.setattr(batch_mod, "enumerate_fusion_atoms", lambda f: [])
    monkeypatch.setattr(batch_mod, "enumerate_reorder_atoms", lambda f: [])

    op_graph, forest = _canonical_state()
    pool = enumerate_pool(op_graph, forest, max_pool_size=100, rng=random.Random(0))
    assert list(pool) == [hash_forest(forest)]
```

- [ ] **Step 2: Run and verify pass**

```
pytest test/codegen/test_batch.py::test_enumerate_pool_no_legal_atoms -x -q
```

Expected: PASS (implementation already handles the empty-frontier-at-start case).

- [ ] **Step 3: Commit**

```bash
git add test/codegen/test_batch.py
git commit -m "Test enumerate_pool handles empty frontier at start"
```

---

### Task 4: `enumerate_pool` — cap respected

**Files:**
- Modify: `test/codegen/test_batch.py`

- [ ] **Step 1: Add the cap-respected test**

Append to `test/codegen/test_batch.py`:

```python
def test_enumerate_pool_cap_respected():
    """len(pool) == max_pool_size when reachable set exceeds the cap.

    rmsnorm+matmul has many legal atoms from the canonical state;
    capping at 3 must halt enumeration at exactly 3 pooled states.
    """
    op_graph, forest = _canonical_state()
    pool = enumerate_pool(op_graph, forest, max_pool_size=3, rng=random.Random(0))
    assert len(pool) == 3
```

- [ ] **Step 2: Run and verify pass**

```
pytest test/codegen/test_batch.py::test_enumerate_pool_cap_respected -x -q
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add test/codegen/test_batch.py
git commit -m "Test enumerate_pool honours max_pool_size cap"
```

---

### Task 5: `enumerate_pool` — deterministic given seed

**Files:**
- Modify: `test/codegen/test_batch.py`

- [ ] **Step 1: Add the determinism test**

Append to `test/codegen/test_batch.py`:

```python
def test_enumerate_pool_deterministic():
    """Two runs with the same seed on the same starting state produce identical pool keys."""
    op_graph, forest = _canonical_state()
    pool_a = enumerate_pool(op_graph, forest, max_pool_size=50, rng=random.Random(42))
    pool_b = enumerate_pool(op_graph, forest, max_pool_size=50, rng=random.Random(42))
    assert sorted(pool_a) == sorted(pool_b)
```

- [ ] **Step 2: Run and verify pass**

```
pytest test/codegen/test_batch.py::test_enumerate_pool_deterministic -x -q
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add test/codegen/test_batch.py
git commit -m "Test enumerate_pool determinism given seeded rng"
```

---

### Task 6: `enumerate_pool` — small-graph exhaustion

**Files:**
- Modify: `test/codegen/test_batch.py`

- [ ] **Step 1: Add the small-graph exhaustion test**

Append to `test/codegen/test_batch.py`:

```python
def test_enumerate_pool_exhausts_small_graph(monkeypatch: pytest.MonkeyPatch):
    """With reachable set |S|=3, cap >> |S|, two independent seeds → identical pool.

    Stubs atom enumerators to simulate a tiny rewrite graph:
      s0 → s1 (atom A)
      s0 → s2 (atom B)
      s1 → s2 (atom C — reaches s2 from a different parent)
      s1/s2: no outgoing atoms
    """
    from nkigym.codegen.graph import OpGraph
    from nkigym.codegen.loop_forest import BodyLeaf, LoopForest, LoopNode
    from nkigym.ops.base import AxisRole
    from nkigym.tune import batch as batch_mod
    from nkigym.tune import KernelRewrite

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

    def _fusion(f):
        return {h0: [atom_a], h1: [atom_c], h2: []}[hash_forest(f)]

    def _reorder(f):
        return {h0: [atom_b], h1: [], h2: []}[hash_forest(f)]

    monkeypatch.setattr(batch_mod, "enumerate_fusion_atoms", _fusion)
    monkeypatch.setattr(batch_mod, "enumerate_reorder_atoms", _reorder)

    pool_a = enumerate_pool(op_graph, forest_s0, max_pool_size=100, rng=random.Random(0))
    pool_b = enumerate_pool(op_graph, forest_s0, max_pool_size=100, rng=random.Random(7))
    """Use sorted-sorted set-equality: hash_forest values are not numerically ordered
    (Python hash() is per-process randomized), so comparing sorted(pool_X) against the
    literal [h0, h1, h2] is flaky."""
    assert sorted(pool_a) == sorted([h0, h1, h2])
    assert sorted(pool_b) == sorted([h0, h1, h2])
```

- [ ] **Step 2: Run and verify pass**

```
pytest test/codegen/test_batch.py::test_enumerate_pool_exhausts_small_graph -x -q
```

Expected: PASS. Invariant (3) in the spec: pool = full reachable set at exhaustion regardless of rng order.

- [ ] **Step 3: Commit**

```bash
git add test/codegen/test_batch.py
git commit -m "Test enumerate_pool exhausts small reachable graph"
```

---

### Task 7: Implement `sample_pool`

**Files:**
- Modify: `nkigym/src/nkigym/tune/batch.py`
- Modify: `test/codegen/test_batch.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/codegen/test_batch.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

```
pytest test/codegen/test_batch.py -x -q -k sample_pool
```

Expected: 3 failures on `NotImplementedError`.

- [ ] **Step 3: Implement `sample_pool`**

Replace the `sample_pool` stub body in `nkigym/src/nkigym/tune/batch.py`:

```python
def sample_pool(
    pool: dict[int, tuple[OpGraph, LoopForest]],
    num_kernels: int,
    rng: random.Random,
) -> list[tuple[OpGraph, LoopForest]]:
    """Sample ``num_kernels`` distinct states uniformly from ``pool``.

    When ``len(pool) < num_kernels``, emits a :class:`UserWarning`
    and returns every pool value. Pool values are returned in
    rng-draw order — callers that care about order should sort or
    otherwise post-process.

    Args:
        pool: Output of :func:`enumerate_pool`.
        num_kernels: Requested number of samples.
        rng: Seeded ``random.Random``.

    Returns:
        List of ``(op_graph, forest)`` pairs; length
        ``min(num_kernels, len(pool))``.
    """
    values = list(pool.values())
    if len(values) < num_kernels:
        warnings.warn(
            f"pool size {len(values)} < num_kernels {num_kernels}; returning all",
            UserWarning,
            stacklevel=2,
        )
        result = values
    else:
        result = rng.sample(values, num_kernels)
    return result
```

- [ ] **Step 4: Run to verify all three pass**

```
pytest test/codegen/test_batch.py -x -q
```

Expected: all tests pass (7 total at this point).

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/tune/batch.py test/codegen/test_batch.py
git commit -m "Implement sample_pool with under-fill warning"
```

---

### Task 8: Add `_assert_no_cpu_sim_failures` + `_trace_output_shape` helpers to `compile.py`

**Files:**
- Modify: `nkigym/src/nkigym/compile.py`

These helpers are *called* from the tune stage driver; they live in `compile.py` for two reasons: (1) they're private stage-driver glue, not reusable kernel-IR logic; (2) `_trace_output_shape` needs `_draw_fp32_inputs` which is already here; (3) `_assert_no_cpu_sim_failures` consumes the autotune-owned `results.json` schema, and `compile.py` already sits on the `nkigym`-side of the one-way boundary.

- [ ] **Step 1: Add helper stubs at the bottom of `nkigym/src/nkigym/compile.py`**

Append below `_draw_fp32_inputs`:

```python
def _trace_output_shape(
    f_numpy: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
) -> tuple[int, ...]:
    """Trace the numpy reference once to recover the output shape.

    ``KernelJob`` requires the output HBM shape on the coordinator so
    workers skip unreliable AST parsing. The same random fp32 inputs
    used by :func:`_cpu_sim_check` drive the trace — if the numpy
    reference is deterministic on shape this is stable regardless of
    seed.
    """
    inputs = _draw_fp32_inputs(input_specs)
    result = f_numpy(**inputs)
    if isinstance(result, tuple):
        result = result[0]
    return tuple(result.shape)


def _assert_no_cpu_sim_failures(results_json_path: Path) -> None:
    """Raise ``AssertionError`` when any kernel in ``results.json`` failed CPU sim.

    ``remote_profile`` writes a ``results.json`` whose ``"kernels"``
    array has one entry per kernel with a boolean ``"cpu_sim_passed"``.
    Per the design spec, CPU-sim divergence on any batched kernel is
    a bug; HW compile/runtime failures are tolerated (emitted into
    ``results.json`` but not raised here).
    """
    data = json.loads(results_json_path.read_text())
    failed = [k["kernel_name"] for k in data["kernels"] if not k["cpu_sim_passed"]]
    if failed:
        raise AssertionError(
            f"CPU-sim failures in batch tune stage: {failed}. "
            f"See {results_json_path} for details."
        )
```

Add `import json` alongside the other imports at the top of the file:

```python
import importlib.util
import json
import sys
```

- [ ] **Step 2: Syntax check only**

```
source ~/venvs/kernel-env/bin/activate
python -c "import nkigym.compile"
```

Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add nkigym/src/nkigym/compile.py
git commit -m "Add trace_output_shape and cpu-sim assertion helpers"
```

---

### Task 9: Wire the batch path into `run_tune` (dispatcher only, no test yet)

**Files:**
- Modify: `nkigym/src/nkigym/tune/stage.py`

- [ ] **Step 1: Extend `run_tune` signature with batch kwargs and dispatch on `rewrites`**

Replace the full contents of `nkigym/src/nkigym/tune/stage.py` with:

```python
"""Driver for the ``"tune"`` stage of ``nkigym_compile``.

Loads the synthesised ``f_nkigym``, builds the canonical
:class:`LoopForest`, then dispatches on ``rewrites``:

* ``rewrites`` is a list → **explicit path** — apply each rewrite in
  order, render to ``kernel_tuned.py``, CPU-sim-check inline. No HW
  profile. Used by tests and backward-compatible callers.

* ``rewrites`` is ``None`` → **batch path** — enumerate the rewrite
  graph via :func:`nkigym.tune.batch.enumerate_pool`, uniformly
  sample ``num_kernels`` states, render each into
  ``kernel_tuned_{idx:04d}.py``, hand the dict to
  ``autotune.remote_profile`` for remote CPU-sim + HW profile,
  raise on any CPU-sim failure.
"""

import random
from collections.abc import Callable
from pathlib import Path

import numpy as np

from autotune.runner.api import remote_profile
from autotune.runner.types import KernelJob
from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.loop_forest import build_canonical_forest, hash_forest
from nkigym.codegen.render import render
from nkigym.tune import KernelRewrite
from nkigym.tune.batch import enumerate_pool, sample_pool
from nkigym.tune.fuse_loops import enumerate_fusion_atoms
from nkigym.tune.reorder_loops import enumerate_reorder_atoms

_MAX_POOL_MULTIPLIER = 100


def run_tune(
    f_numpy: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    cache_path: Path,
    rewrites: list[KernelRewrite] | None,
    seed: int,
    load_f_nkigym: Callable[[Path], Callable[..., np.ndarray]],
    cpu_sim_check: Callable[[str, str, Callable[..., np.ndarray], dict[str, tuple[tuple[int, ...], str]]], None],
    num_kernels: int,
    hosts: list[str],
    venv_python: str,
    neuron_platform_target: str,
    collect_detailed_profile: bool,
    trace_output_shape: Callable[
        [Callable[..., np.ndarray], dict[str, tuple[tuple[int, ...], str]]], tuple[int, ...]
    ],
    assert_no_cpu_sim_failures: Callable[[Path], None],
    atol: float,
    rtol: float,
) -> None:
    """Apply structural rewrites or batch-sample and profile.

    See module docstring for the two dispatch paths.

    Args:
        f_numpy: Plain-numpy reference — CPU-sim golden (explicit
            path) and output-shape trace source (batch path).
        input_specs: Per-param ``(shape, dtype_str)``.
        cache_path: Directory holding ``f_nkigym.py``; receives
            ``kernel_tuned.py`` (explicit) or
            ``kernel_tuned_{idx:04d}.py`` + ``results.json`` (batch).
        rewrites: List → explicit path; ``None`` → batch path.
        seed: Seeds the random draw and the ``remote_profile`` input
            tensor generation.
        load_f_nkigym: Loader from ``compile.py`` — injected to keep
            the stage driver independent.
        cpu_sim_check: fp32-simulation correctness helper (explicit
            path only).
        num_kernels: Batch-path kernel count. Ignored on the explicit
            path.
        hosts: ``remote_profile`` SSH hostnames. Ignored on the
            explicit path.
        venv_python: Python executable path on remote hosts.
        neuron_platform_target: Neuron target, e.g. ``"trn2"``.
        collect_detailed_profile: Forwarded to ``remote_profile``.
        trace_output_shape: Helper that runs ``f_numpy`` once to
            recover the output shape for ``KernelJob``.
        assert_no_cpu_sim_failures: Helper that reads ``results.json``
            and raises ``AssertionError`` on any ``cpu_sim_passed ==
            False`` entry.
        atol: CPU-sim abs tolerance forwarded to ``KernelJob``.
        rtol: CPU-sim rel tolerance forwarded to ``KernelJob``.
    """
    f_nkigym_path = cache_path / "f_nkigym.py"
    if not f_nkigym_path.exists():
        raise ValueError(
            f"tune requires {f_nkigym_path!s} — run the 'synthesis' stage first "
            f"or place the file manually before invoking this stage."
        )
    f_nkigym = load_f_nkigym(f_nkigym_path)
    op_graph = parse_and_resolve(f_nkigym, input_specs)
    forest = build_canonical_forest(op_graph)

    if rewrites is None:
        _run_batch(
            f_numpy=f_numpy,
            f_nkigym=f_nkigym,
            input_specs=input_specs,
            cache_path=cache_path,
            op_graph=op_graph,
            forest=forest,
            seed=seed,
            num_kernels=num_kernels,
            hosts=hosts,
            venv_python=venv_python,
            neuron_platform_target=neuron_platform_target,
            collect_detailed_profile=collect_detailed_profile,
            trace_output_shape=trace_output_shape,
            assert_no_cpu_sim_failures=assert_no_cpu_sim_failures,
            atol=atol,
            rtol=rtol,
        )
    else:
        _run_explicit(
            f_numpy=f_numpy,
            f_nkigym=f_nkigym,
            input_specs=input_specs,
            cache_path=cache_path,
            op_graph=op_graph,
            forest=forest,
            rewrites=rewrites,
            cpu_sim_check=cpu_sim_check,
        )


def _run_explicit(
    *,
    f_numpy: Callable[..., np.ndarray],
    f_nkigym: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    cache_path: Path,
    op_graph,
    forest,
    rewrites: list[KernelRewrite],
    cpu_sim_check: Callable[[str, str, Callable[..., np.ndarray], dict[str, tuple[tuple[int, ...], str]]], None],
) -> None:
    """Apply ``rewrites`` deterministically and CPU-sim-check the result."""
    for r in rewrites:
        if not r.is_legal(op_graph, forest):
            raise ValueError(f"{r!r} illegal on current state")
        op_graph, forest = r.apply(op_graph, forest)
    kernel_source = render(op_graph, forest=forest)
    (cache_path / "kernel_tuned.py").write_text(kernel_source)
    cpu_sim_check(kernel_source, f_nkigym.__name__, f_numpy, input_specs)


def _run_batch(
    *,
    f_numpy: Callable[..., np.ndarray],
    f_nkigym: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    cache_path: Path,
    op_graph,
    forest,
    seed: int,
    num_kernels: int,
    hosts: list[str],
    venv_python: str,
    neuron_platform_target: str,
    collect_detailed_profile: bool,
    trace_output_shape: Callable[
        [Callable[..., np.ndarray], dict[str, tuple[tuple[int, ...], str]]], tuple[int, ...]
    ],
    assert_no_cpu_sim_failures: Callable[[Path], None],
    atol: float,
    rtol: float,
) -> None:
    """Enumerate the rewrite pool, render ``num_kernels`` samples, profile on HW."""
    rng = random.Random(seed)
    pool = enumerate_pool(
        op_graph=op_graph,
        forest=forest,
        max_pool_size=_MAX_POOL_MULTIPLIER * num_kernels,
        rng=rng,
    )
    sampled = sample_pool(pool, num_kernels=num_kernels, rng=rng)

    output_shape = trace_output_shape(f_numpy, input_specs)
    nkigym_source = (cache_path / "f_nkigym.py").read_text()

    kernels: dict[str, KernelJob] = {}
    for idx, (og, f) in enumerate(sampled):
        source = render(og, forest=f)
        name = f"kernel_tuned_{idx:04d}.py"
        (cache_path / name).write_text(source)
        kernels[name] = KernelJob(
            source=source,
            func_name=f_nkigym.__name__,
            output_shape=output_shape,
            input_specs=input_specs,
            nkigym_source=nkigym_source,
            nkigym_func_name=f_nkigym.__name__,
            atol=atol,
            rtol=rtol,
        )

    remote_profile(
        kernels=kernels,
        hosts=hosts,
        cache_dir=str(cache_path),
        seed=seed,
        neuron_platform_target=neuron_platform_target,
        venv_python=venv_python,
        collect_detailed_profile=collect_detailed_profile,
    )

    assert_no_cpu_sim_failures(cache_path / "results.json")
```

**Why the verbose injection?** `_run_batch` needs `_trace_output_shape` and `_assert_no_cpu_sim_failures`; both live in `compile.py`. Rather than circular-import them, they're passed in — same pattern as `load_f_nkigym` / `cpu_sim_check` already used by the explicit path. This keeps `stage.py` importable without the `compile.py` helpers and preserves testability via stub injection.

- [ ] **Step 2: Syntax check**

```
source ~/venvs/kernel-env/bin/activate
python -c "import nkigym.tune.stage"
```

Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add nkigym/src/nkigym/tune/stage.py
git commit -m "Add batch-path dispatch to run_tune"
```

---

### Task 10: Update `nkigym_compile` signature — drop `stages`, add batch kwargs

**Files:**
- Modify: `nkigym/src/nkigym/compile.py`

- [ ] **Step 1: Replace the `nkigym_compile` function definition**

Find the current `def nkigym_compile(...)` in `nkigym/src/nkigym/compile.py` and replace it — along with the `_KNOWN_STAGES` constant and the stage-dispatch loop — with:

```python
def nkigym_compile(
    f_numpy: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    cache_dir: str | Path,
    num_kernels: int,
    hosts: list[str],
    venv_python: str,
    neuron_platform_target: str,
    collect_detailed_profile: bool = False,
    rewrites: list[KernelRewrite] | None = None,
    seed: int = 0,
) -> None:
    """Run the full nkigym compile pipeline: synthesis → initial_codegen → tune.

    Args:
        f_numpy: Plain-numpy reference. Drives synthesis (input),
            ``initial_codegen``'s inline CPU-sim check (golden), and
            the batch path's output-shape trace + kernel-level
            correctness golden.
        input_specs: ``{param_name: (shape, dtype_str)}``; order
            matches ``f_numpy``'s positional parameters by name.
        cache_dir: Artifact directory. Created if missing;
            pre-existing files for each stage are overwritten.
        num_kernels: Batch path — number of kernels to sample + profile.
            Ignored when ``rewrites`` is not ``None``.
        hosts: SSH hosts for ``remote_profile``. Ignored when
            ``rewrites`` is not ``None``.
        venv_python: Python executable on remote hosts.
        neuron_platform_target: Neuron target (e.g. ``"trn2"``).
        collect_detailed_profile: Collect the full per-instruction
            profile + NEFF/NTFF per kernel. Off by default.
        rewrites: ``None`` → batch path (default). List →
            deterministic explicit path, writes
            ``kernel_tuned.py`` only, no HW profile.
        seed: Drives sampling (batch path) and ``remote_profile``
            input-tensor generation.

    Raises:
        AssertionError: Any sampled kernel fails CPU sim (batch) or
            the canonical ``initial_codegen`` CPU sim fails.
        ValueError: ``rewrites`` list has an illegal atom on the
            current state.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    _run_synthesis(f_numpy, input_specs, cache_path)
    _run_initial_codegen(f_numpy, input_specs, cache_path)
    run_tune(
        f_numpy=f_numpy,
        input_specs=input_specs,
        cache_path=cache_path,
        rewrites=rewrites,
        seed=seed,
        load_f_nkigym=_load_f_nkigym,
        cpu_sim_check=_cpu_sim_check,
        num_kernels=num_kernels,
        hosts=hosts,
        venv_python=venv_python,
        neuron_platform_target=neuron_platform_target,
        collect_detailed_profile=collect_detailed_profile,
        trace_output_shape=_trace_output_shape,
        assert_no_cpu_sim_failures=_assert_no_cpu_sim_failures,
        atol=_ATOL,
        rtol=_RTOL,
    )
```

Delete the line `_KNOWN_STAGES = ("synthesis", "initial_codegen", "tune")` (it's no longer used).

- [ ] **Step 2: Update the module docstring header (lines 1-19) to drop `stages` language**

Replace the module docstring with:

```python
"""``nkigym_compile``: synthesis → eager codegen → tune driver.

Single public entry point for the nkigym compilation pipeline. Runs
all three stages in order on every call and caches their artifacts
under ``cache_dir``:

``<cache_dir>/``

* ``f_nkigym.py``              — synthesised ``f_nkigym`` math function.
* ``kernel.py``                — rendered canonical eager NKI kernel.
* ``kernel_tuned_{idx:04d}.py``  — batch-path rendered samples
  (``rewrites=None``; default).
* ``kernel_tuned.py``          — explicit-path single rendered kernel
  (``rewrites=[...]``).
* ``results.json``             — per-kernel + aggregate profile data,
  written by ``autotune.remote_profile`` on the batch path.

``initial_codegen`` runs an automatic fp32 CPU-sim accuracy check.
The batch tune path runs the same check per kernel on remote workers
and raises if any kernel diverges.
"""
```

- [ ] **Step 3: Syntax + signature smoke test**

```
source ~/venvs/kernel-env/bin/activate
python -c "import inspect; import nkigym; sig=inspect.signature(nkigym.nkigym_compile); print(list(sig.parameters))"
```

Expected:
```
['f_numpy', 'input_specs', 'cache_dir', 'num_kernels', 'hosts', 'venv_python', 'neuron_platform_target', 'collect_detailed_profile', 'rewrites', 'seed']
```

- [ ] **Step 4: Commit**

```bash
git add nkigym/src/nkigym/compile.py
git commit -m "Drop stages kwarg from nkigym_compile and add batch kwargs"
```

---

### Task 11: Fix existing `test_compile.py` — delete stage-validation tests, migrate explicit-rewrites tests

**Files:**
- Modify: `test/codegen/test_compile.py`

All tests currently call `nkigym_compile(..., stages=[...], ...)`. The batch kwargs (`num_kernels`, `hosts`, `venv_python`, `neuron_platform_target`) are now required — explicit-rewrites callers must pass them even though they're unused on that path.

- [ ] **Step 1: Read current test contents**

```
cat test/codegen/test_compile.py
```

Expected: 195 lines (the tests listed earlier in this plan).

- [ ] **Step 2: Delete two obsolete tests**

Remove these two functions entirely:

1. `test_initial_codegen_without_synthesis_artifact_raises` — this asserted `stages=["initial_codegen"]` without `f_nkigym.py` raised; synthesis now runs unconditionally before initial_codegen, so the scenario no longer exists.
2. `test_unknown_stage_raises` — `stages` kwarg is gone, so no parameter to validate against.

- [ ] **Step 3: Migrate remaining tests — batch kwargs on every call, replace `stages=[...]` with direct `run_tune` calls where the test targets the tune stage only**

The remaining tests that run `stages=["initial_codegen"]` or `stages=["tune"]` need to change strategy:

- Tests that call `initial_codegen` standalone with a pre-seeded `f_nkigym.py` must be rewritten to skip synthesis. Easiest migration: import `_run_initial_codegen` directly and call it, bypassing `nkigym_compile`.
- Tests that call `tune` standalone with a pre-seeded `f_nkigym.py` must also skip synthesis. Import `run_tune` directly with stub helpers.

Replace the full contents of `test/codegen/test_compile.py` with:

```python
"""Tests for ``nkigym_compile`` and the tune-stage run_tune driver.

The synthesis stage is expensive (calls the Claude Agent SDK) and
non-deterministic, so these tests bypass it by seeding
``f_nkigym.py`` on disk and invoking the post-synthesis helpers
directly.
"""

from pathlib import Path

import numpy as np
import pytest

from nkigym.compile import _cpu_sim_check, _load_f_nkigym, _run_initial_codegen
from nkigym.tune.fuse_loops import FuseLoops
from nkigym.tune.stage import run_tune


def _rmsnorm_matmul_numpy(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Plain-numpy ``rmsnorm(lhs) @ rhs`` golden."""
    m = np.mean(np.square(lhs), axis=1, keepdims=True)
    rms_inv = 1.0 / np.sqrt(m + 1e-6)
    return (lhs * rms_inv) @ rhs


_F_NKIGYM_SOURCE = """\
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.store import NKIStore
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.transpose import NKITranspose
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.activation_reduce import NKIActivationReduce


@nkigym_kernel
def f_nkigym(lhs, rhs):
    lhs_sbuf = NKILoad()(data=lhs)
    rhs_sbuf = NKILoad()(data=rhs)
    rms_inv = NKIActivationReduce(op="square", reduce_op="add", post_op="rsqrt", scale=1 / 256, bias=1e-6)(
        data=lhs_sbuf
    )
    lhs_rms = NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv)
    lhs_T = NKITranspose()(data=lhs_rms)
    prod = NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out
"""


_SPECS: dict[str, tuple[tuple[int, ...], str]] = {
    "lhs": ((128, 256), "bfloat16"),
    "rhs": ((256, 512), "bfloat16"),
}


def _seed_f_nkigym(tmp_path: Path) -> None:
    (tmp_path / "f_nkigym.py").write_text(_F_NKIGYM_SOURCE)


def _explicit_run_tune(tmp_path: Path, rewrites: list) -> None:
    """Invoke run_tune on the explicit path with stub batch kwargs."""
    run_tune(
        f_numpy=_rmsnorm_matmul_numpy,
        input_specs=_SPECS,
        cache_path=tmp_path,
        rewrites=rewrites,
        seed=0,
        load_f_nkigym=_load_f_nkigym,
        cpu_sim_check=_cpu_sim_check,
        num_kernels=0,
        hosts=[],
        venv_python="",
        neuron_platform_target="",
        collect_detailed_profile=False,
        trace_output_shape=lambda f, s: (0,),
        assert_no_cpu_sim_failures=lambda p: None,
        atol=5e-3,
        rtol=5e-3,
    )


def test_initial_codegen_runs_cpu_sim_and_writes_kernel(tmp_path: Path) -> None:
    """initial_codegen produces kernel.py and auto-validates it against the numpy golden."""
    _seed_f_nkigym(tmp_path)
    _run_initial_codegen(_rmsnorm_matmul_numpy, _SPECS, tmp_path)
    kernel_path = tmp_path / "kernel.py"
    assert kernel_path.exists()
    kernel_source = kernel_path.read_text()
    assert "@nki.jit" in kernel_source
    assert "def f_nkigym(lhs, rhs):" in kernel_source


def test_cpu_sim_mismatch_raises(tmp_path: Path) -> None:
    """If the rendered kernel doesn't match the numpy golden, the check raises."""
    _seed_f_nkigym(tmp_path)

    def bogus_golden(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return lhs @ rhs[:, : rhs.shape[1]] + 1.0

    with pytest.raises(AssertionError, match="CPU-sim mismatch"):
        _run_initial_codegen(bogus_golden, _SPECS, tmp_path)


def test_run_tune_explicit_empty_rewrites_writes_kernel_tuned(tmp_path: Path) -> None:
    """Explicit path with rewrites=[] writes kernel_tuned.py and CPU-sim succeeds."""
    _seed_f_nkigym(tmp_path)
    _explicit_run_tune(tmp_path, rewrites=[])
    assert (tmp_path / "kernel_tuned.py").exists()


def test_run_tune_explicit_applies_fuse_loops_d0(tmp_path: Path) -> None:
    """Applying FuseLoops on activation_reduce↔tensor_scalar d0 still matches numpy."""
    _seed_f_nkigym(tmp_path)
    rewrites = [FuseLoops(path=(), boundary=(2, 3), dim_id="d0")]
    _explicit_run_tune(tmp_path, rewrites=rewrites)
    kernel_source = (tmp_path / "kernel_tuned.py").read_text()
    assert "nisa.activation_reduce(" in kernel_source
    assert "nisa.tensor_scalar(" in kernel_source


def test_run_tune_explicit_rejects_illegal_rewrite(tmp_path: Path) -> None:
    """A rewrite that fails is_legal raises ValueError before producing any artifact."""
    _seed_f_nkigym(tmp_path)
    bogus = FuseLoops(path=(), boundary=(99, 100), dim_id="d0")
    with pytest.raises(ValueError, match="illegal"):
        _explicit_run_tune(tmp_path, rewrites=[bogus])
    assert not (tmp_path / "kernel_tuned.py").exists()


def test_run_tune_explicit_applies_reorder_loops_inside_tensor_scalar(tmp_path: Path) -> None:
    """An explicit ReorderLoops on tensor_scalar's inner chain still matches numpy."""
    from nkigym.tune.reorder_loops import ReorderLoops

    _seed_f_nkigym(tmp_path)
    rewrites = [ReorderLoops(path=(3, 0), outer_dim="d0", inner_dim="d1")]
    _explicit_run_tune(tmp_path, rewrites=rewrites)
    assert (tmp_path / "kernel_tuned.py").exists()


def test_run_tune_explicit_compose_reorder_then_fuse(tmp_path: Path) -> None:
    """Reorder exposes a new fusion boundary; composed pipeline still CPU-sim-passes."""
    from nkigym.tune.reorder_loops import ReorderLoops

    _seed_f_nkigym(tmp_path)
    rewrites = [
        ReorderLoops(path=(3, 0), outer_dim="d0", inner_dim="d1"),
        FuseLoops(path=(), boundary=(2, 3), dim_id="d0"),
    ]
    _explicit_run_tune(tmp_path, rewrites=rewrites)
    assert (tmp_path / "kernel_tuned.py").exists()
```

**Dropped:** `test_tune_stage_random_draw_is_reproducible` and `test_tune_stage_random_draw_terminates_when_only_self_inverse_atom_exists` — both probe the old rewrites=None in-line random draw that no longer exists. Reproducibility of the batch path is covered by `test_enumerate_pool_deterministic` + upcoming `test_nkigym_compile_batch_path_mocks_remote_profile` (Task 12).

- [ ] **Step 4: Run the migrated tests**

```
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/test_compile.py -x -q
```

Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add test/codegen/test_compile.py
git commit -m "Migrate test_compile.py off stages kwarg and to direct run_tune calls"
```

---

### Task 12: Batch-path integration test — mock `remote_profile`

**Files:**
- Modify: `test/codegen/test_compile.py`

- [ ] **Step 1: Write the failing test**

Append to `test/codegen/test_compile.py`:

```python
def test_nkigym_compile_batch_path_mocks_remote_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Batch path calls remote_profile with one KernelJob per sampled kernel."""
    import json

    from nkigym import nkigym_compile
    from nkigym.tune import stage as stage_mod

    _seed_f_nkigym(tmp_path)

    captured: dict = {}

    def _fake_remote_profile(**kwargs):
        captured.update(kwargs)
        """Write a minimal results.json mimicking remote_profile's schema."""
        results = {
            "metadata": {"num_kernels": len(kwargs["kernels"]), "wallclock_s": 0.0, "hosts": []},
            "metrics": {},
            "kernels": [
                {"kernel_name": name, "cpu_sim_passed": True, "hardware_output": ""}
                for name in kwargs["kernels"]
            ],
        }
        (tmp_path / "results.json").write_text(json.dumps(results))

    """Also stub _run_synthesis so the seeded f_nkigym.py isn't overwritten."""

    def _noop_synthesis(f_numpy, input_specs, cache_path):
        pass

    monkeypatch.setattr(stage_mod, "remote_profile", _fake_remote_profile)
    monkeypatch.setattr("nkigym.compile._run_synthesis", _noop_synthesis)

    nkigym_compile(
        f_numpy=_rmsnorm_matmul_numpy,
        input_specs=_SPECS,
        cache_dir=tmp_path,
        num_kernels=3,
        hosts=["gym-1"],
        venv_python="/home/ubuntu/venvs/kernel-env/bin/python",
        neuron_platform_target="trn2",
        seed=0,
    )

    names = sorted(captured["kernels"].keys())
    assert names == ["kernel_tuned_0000.py", "kernel_tuned_0001.py", "kernel_tuned_0002.py"]
    for name, job in captured["kernels"].items():
        assert job.func_name == "f_nkigym"
        assert job.input_specs == _SPECS
        assert job.nkigym_func_name == "f_nkigym"
        assert (tmp_path / name).exists()
    assert captured["neuron_platform_target"] == "trn2"
    assert captured["venv_python"] == "/home/ubuntu/venvs/kernel-env/bin/python"
    assert captured["hosts"] == ["gym-1"]
```

- [ ] **Step 2: Run and verify PASS**

```
pytest test/codegen/test_compile.py::test_nkigym_compile_batch_path_mocks_remote_profile -x -q
```

Expected: PASS.

If the test fails because the rewrite graph has fewer than 3 reachable states, reduce `num_kernels=3` to `num_kernels=2` and adjust the assertion list.

- [ ] **Step 3: Commit**

```bash
git add test/codegen/test_compile.py
git commit -m "Integration test batch path invokes remote_profile correctly"
```

---

### Task 13: Batch-path CPU-sim failure raises

**Files:**
- Modify: `test/codegen/test_compile.py`

- [ ] **Step 1: Write the failing test**

Append to `test/codegen/test_compile.py`:

```python
def test_nkigym_compile_batch_raises_on_cpu_sim_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single failing kernel in results.json surfaces as AssertionError."""
    import json

    from nkigym import nkigym_compile
    from nkigym.tune import stage as stage_mod

    _seed_f_nkigym(tmp_path)

    def _fake_remote_profile(**kwargs):
        results = {
            "metadata": {"num_kernels": len(kwargs["kernels"]), "wallclock_s": 0.0, "hosts": []},
            "metrics": {},
            "kernels": [
                {
                    "kernel_name": name,
                    "cpu_sim_passed": (idx != 1),
                    "hardware_output": "",
                }
                for idx, name in enumerate(sorted(kwargs["kernels"]))
            ],
        }
        (tmp_path / "results.json").write_text(json.dumps(results))

    def _noop_synthesis(f_numpy, input_specs, cache_path):
        pass

    monkeypatch.setattr(stage_mod, "remote_profile", _fake_remote_profile)
    monkeypatch.setattr("nkigym.compile._run_synthesis", _noop_synthesis)

    with pytest.raises(AssertionError, match="kernel_tuned_0001.py"):
        nkigym_compile(
            f_numpy=_rmsnorm_matmul_numpy,
            input_specs=_SPECS,
            cache_dir=tmp_path,
            num_kernels=3,
            hosts=["gym-1"],
            venv_python="/home/ubuntu/venvs/kernel-env/bin/python",
            neuron_platform_target="trn2",
            seed=0,
        )
```

- [ ] **Step 2: Run and verify PASS**

```
pytest test/codegen/test_compile.py::test_nkigym_compile_batch_raises_on_cpu_sim_failure -x -q
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add test/codegen/test_compile.py
git commit -m "Test batch path raises AssertionError on CPU-sim failure"
```

---

### Task 14: Migrate `examples/rmsnorm_matmul.py`

**Files:**
- Modify: `examples/rmsnorm_matmul.py`

- [ ] **Step 1: Replace the `nkigym_compile` call**

Open `examples/rmsnorm_matmul.py`. Replace lines 46-53 (the block-comment + the `nkigym_compile(...)` call) with:

```python
    """Batch tune path: enumerate the rewrite graph from the canonical
    forest, sample 16 distinct kernels, render each, profile on the
    gym via autotune.remote_profile. Adjust ``num_kernels`` / ``hosts``
    as needed; the ``seed`` kwarg controls sampling reproducibility."""
    nkigym_compile(
        f_numpy=rmsnorm_matmul_numpy,
        input_specs=INPUT_SPECS,
        cache_dir=cache_dir,
        num_kernels=16,
        hosts=["gym-1"],
        venv_python="/home/ubuntu/venvs/kernel-env/bin/python",
        neuron_platform_target="trn2",
        seed=0,
    )
    print(f"[rmsnorm_matmul] canonical kernel: {cache_dir / 'kernel.py'}")
    print(f"[rmsnorm_matmul] results.json:     {cache_dir / 'results.json'}")
```

Also drop the "tune stage defaults to a seeded random draw" paragraph in the module docstring (lines 10-13) since the batch path replaces that language. Concretely, update the module docstring header to:

```python
"""Compile ``rmsnorm(lhs) @ rhs`` from numpy through the full nkigym pipeline.

Runs the three stages of ``nkigym_compile`` end-to-end:

    1. ``synthesis``        — synthesise ``f_nkigym`` via the Claude
                              Agent SDK; write ``<cache>/f_nkigym.py``.
    2. ``initial_codegen``  — render the canonical eager NKI kernel
                              into ``<cache>/kernel.py`` and validate
                              against the numpy reference through
                              ``nki.simulate``.
    3. ``tune`` (batch)     — enumerate the rewrite pool, sample
                              ``num_kernels`` kernels, render each
                              into ``<cache>/kernel_tuned_NNNN.py``,
                              CPU-sim + HW profile via
                              ``autotune.remote_profile``. Results
                              land in ``<cache>/results.json``.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/rmsnorm_matmul.py
"""
```

- [ ] **Step 2: Syntax check — import the module, don't execute**

```
source ~/venvs/kernel-env/bin/activate
python -c "import ast; ast.parse(open('examples/rmsnorm_matmul.py').read())"
```

Expected: no output, exit 0. Running the example end-to-end is out of scope — it requires gym hosts.

- [ ] **Step 3: Commit**

```bash
git add examples/rmsnorm_matmul.py
git commit -m "Migrate rmsnorm_matmul example to batch tune kwargs"
```

---

### Task 15: Full test suite sanity run

**Files:** none changed.

- [ ] **Step 1: Run the full test suite**

```
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/ -q
```

Expected: all tests pass. If unrelated tests fail, diagnose before proceeding — the migration should be additive.

- [ ] **Step 2: (If green) Push branch if upstream expects it**

```bash
git log --oneline origin/dev_1..HEAD
```

Expected: 13 new commits (Tasks 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 — Task 14 is one more → 14 total). Leave push to the user.

---

## Self-Review

**Spec coverage check.** Every section of the spec maps to a task:

| Spec section | Task(s) |
|---|---|
| Core decision / Invariants | Tasks 2-7 (implementation proves the invariants) |
| `nkigym_compile` signature | Task 10 |
| `run_tune` dispatcher | Task 9 |
| Batch sampler `enumerate_pool` | Tasks 2-6 |
| Batch sampler `sample_pool` | Task 7 |
| Driver `_run_batch` | Task 9 |
| Cache layout | Task 9 (produces files) + Task 12 (asserts naming) |
| Edge case: no legal atoms | Task 3 |
| Edge case: under-fill warn | Task 7 |
| Edge case: reachable > cap | Task 4 |
| Edge case: exhaustion | Task 6 |
| Edge case: CPU-sim tolerance | Tasks 9/10 wire `_ATOL`/`_RTOL` into `KernelJob` |
| Unit tests | Tasks 1-7 |
| Integration tests | Tasks 12-13 |
| Migration checklist entry 1 (rmsnorm example) | Task 14 |
| Migration checklist entry 2 (matmul examples — no-op) | N/A, verified empty |
| Migration checklist entry 3 (test_compile) | Task 11 |
| Migration checklist entry 4 (agentic re-grep) | Step below |

Agentic re-grep: `grep -rn "nkigym_compile\|stages=" agentic/ --include="*.py" | grep -v __pycache__` run at Task 14 end if anything surfaces, fold in as an extra migration task.

**Placeholder scan.** No "TBD", "TODO", "similar to", or undefined references. Every file path absolute. Every test has real assertions. Every `pytest -q` run has an explicit expected outcome.

**Type consistency check.**

- `enumerate_pool` returns `dict[int, tuple[OpGraph, LoopForest]]` in stub (Task 1), implementation (Task 2), and consumer (`_run_batch`, Task 9). ✓
- `sample_pool` returns `list[tuple[OpGraph, LoopForest]]` — consumers in Task 9 iterate `for idx, (og, f) in enumerate(sampled)`. ✓
- `_trace_output_shape` returns `tuple[int, ...]`; `KernelJob.output_shape` is `tuple[int, ...]`. ✓
- `_assert_no_cpu_sim_failures` takes `Path`; called with `cache_path / "results.json"` (Path). ✓
- `run_tune` signature in Task 9 defines `num_kernels`, `hosts`, etc. as keyword-only via `*,` (explicit in `_run_batch` / `_run_explicit`). Actually — review: `run_tune` itself does not use `*,` in my Task 9 code, so these are positional-or-keyword. The calling site in Task 10 uses all keyword arguments, which is safe. No consistency issue.
- `_fake_remote_profile(**kwargs)` in Tasks 12/13 matches the keyword-only call in `_run_batch`. ✓

**Scope check.** 14 code-touching tasks + 1 sanity task. Each task is narrowly scoped: a stub-and-test pair, a single function implementation, or a targeted migration. Commits are at each task boundary.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-05-unified-tune-stage.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
