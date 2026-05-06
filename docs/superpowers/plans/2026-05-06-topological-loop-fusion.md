# Topological Loop Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `FuseLoops` from literal-adjacent sibling fusion to topological-adjacent sibling fusion, driven by a persistent op-level dependency graph attached to `OpGraph`.

**Architecture:** Add `DepGraph` (flat 4-dict dataclass of producer/consumers/reads/writes over `ParsedOp.idx`) and subtree helpers in a new module `nkigym/src/nkigym/codegen/dep_graph.py`. Extend `parse_and_resolve` to build and attach the `DepGraph` once. Widen `FuseLoops.boundary` to accept any `(i, j)` with `j > i`, gated by a topological-adjacency legality rule. Strictly generalises the existing atom — literal-adjacent pairs produce byte-identical atoms.

**Tech Stack:** Python 3.12, pytest, `black`, `isort`, `pyright`. Kernel venv at `~/venvs/kernel-env/bin/activate`. Tests live in `/home/ubuntu/nki-autotune/test/codegen/`; source in `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/` and `nkigym/tune/`.

**Design doc:** `docs/superpowers/specs/2026-05-06-topological-loop-fusion-design.md`

---

## File Structure

New files:

- `nkigym/src/nkigym/codegen/dep_graph.py` — `DepGraph` dataclass, `build_dep_graph`, subtree helpers (`subtree_ops`, `subtree_reads`, `subtree_writes`, `commutes`).
- `test/codegen/test_dep_graph.py` — unit tests for the new module.
- `test/codegen/test_fuse_loops_cpu_sim.py` — CPU-sim correctness gate on the generalised fuse.

Modified files:

- `nkigym/src/nkigym/codegen/graph.py` — `OpGraph` gains `dep: DepGraph` field; `parse_and_resolve` builds it.
- `nkigym/src/nkigym/tune/fuse_loops.py` — widened `is_legal`, rewritten `apply`, `enumerate_fusion_atoms` signature change.
- `nkigym/src/nkigym/tune/batch.py` — two call-site updates to pass `op_graph` into `enumerate_fusion_atoms`.
- `test/codegen/test_tune.py` — add topological-adjacency cases.

Unchanged:

- `nkigym/src/nkigym/codegen/loop_forest.py`
- `nkigym/src/nkigym/codegen/render.py`
- `nkigym/src/nkigym/tune/reorder_loops.py`
- `nkigym/src/nkigym/tune/stage.py`
- `nkigym/src/nkigym/tune/__init__.py`

---

## Preconditions

All tasks assume:

- Working directory `/home/ubuntu/nki-autotune`
- Venv activated: `source ~/venvs/kernel-env/bin/activate`
- Existing test suite green on `dev_1`:
  ```bash
  pytest test/codegen/ -x
  ```

---

## Task 1: Scaffold `dep_graph.py` with dataclass and stub

**Files:**
- Create: `nkigym/src/nkigym/codegen/dep_graph.py`
- Create: `test/codegen/test_dep_graph.py`

- [ ] **Step 1: Write the failing test (imports + dataclass fields)**

Create `/home/ubuntu/nki-autotune/test/codegen/test_dep_graph.py`:

```python
"""Unit tests for the op-level dependency graph."""

from nkigym.codegen.dep_graph import DepGraph


def test_dep_graph_is_frozen_dataclass_with_expected_fields() -> None:
    """DepGraph exposes producer, consumers, reads, writes as the four persisted maps."""
    dep = DepGraph(producer={}, consumers={}, reads={}, writes={})
    assert dep.producer == {}
    assert dep.consumers == {}
    assert dep.reads == {}
    assert dep.writes == {}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ubuntu/nki-autotune
pytest test/codegen/test_dep_graph.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nkigym.codegen.dep_graph'`.

- [ ] **Step 3: Write minimal implementation**

Create `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/dep_graph.py`:

```python
"""Op-level dependency graph over an ``OpGraph``.

Nodes are ``ParsedOp.idx`` values. Edges are inferred from tensor
identities: reads = ``operand_names`` restricted to ``OPERAND_AXES``
slots; writes = ``output_names``. Built once by
:func:`nkigym.codegen.graph.parse_and_resolve` and attached to the
resulting :class:`~nkigym.codegen.graph.OpGraph`.

Consumers of the dep graph compose the persisted op-level maps with
subtree helpers (``subtree_reads`` / ``subtree_writes`` /
``commutes``) that walk a :class:`~nkigym.codegen.loop_forest.LoopNode`
and aggregate per-op edges.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DepGraph:
    """Op-level dependency DAG over an :class:`OpGraph`.

    Attributes:
        producer: Maps tensor name to the ``ParsedOp.idx`` that writes
            it. Parameter tensors (never written by any op) map to
            ``None``. Every tensor in ``op_graph.tensors`` appears as
            a key.
        consumers: Maps tensor name to a tuple of ``ParsedOp.idx``
            values that read the tensor, in source order.
        reads: Maps ``ParsedOp.idx`` to the frozenset of tensor names
            the op reads. Only operand slots whose key is in the op
            class's ``OPERAND_AXES`` are counted.
        writes: Maps ``ParsedOp.idx`` to the frozenset of tensor names
            the op writes (``ParsedOp.output_names``).
    """

    producer: dict[str, int | None]
    consumers: dict[str, tuple[int, ...]]
    reads: dict[int, frozenset[str]]
    writes: dict[int, frozenset[str]]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/codegen/test_dep_graph.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/dep_graph.py test/codegen/test_dep_graph.py
git commit -m "codegen: add DepGraph dataclass scaffolding"
```

---

## Task 2: Implement `build_dep_graph`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/dep_graph.py`
- Modify: `test/codegen/test_dep_graph.py`

- [ ] **Step 1: Write the failing tests**

Append to `/home/ubuntu/nki-autotune/test/codegen/test_dep_graph.py`:

```python
from nkigym.codegen.dep_graph import build_dep_graph
from nkigym.codegen.graph import parse_and_resolve
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose


@nkigym_kernel
def _two_op_chain(lhs):
    lhs_sbuf = NKILoad()(data=lhs)
    out = NKIStore()(data=lhs_sbuf)
    return out


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


_CHAIN_SPECS = {"lhs": ((128, 256), "bfloat16")}
_RMSNORM_SPECS = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}


def test_build_dep_graph_two_op_chain_producer_consumers() -> None:
    """load -> store: load writes lhs_sbuf, store reads it."""
    g = parse_and_resolve(_two_op_chain, _CHAIN_SPECS)
    dep = build_dep_graph(g.ops, g.tensors)
    assert dep.producer["lhs"] is None
    assert dep.producer["lhs_sbuf"] == 0
    assert dep.producer["out"] == 1
    assert dep.consumers["lhs"] == (0,)
    assert dep.consumers["lhs_sbuf"] == (1,)
    assert dep.consumers["out"] == ()


def test_build_dep_graph_two_op_chain_reads_writes() -> None:
    """Op-level reads/writes use OPERAND_AXES (reads) and output_names (writes)."""
    g = parse_and_resolve(_two_op_chain, _CHAIN_SPECS)
    dep = build_dep_graph(g.ops, g.tensors)
    assert dep.reads[0] == frozenset({"lhs"})
    assert dep.writes[0] == frozenset({"lhs_sbuf"})
    assert dep.reads[1] == frozenset({"lhs_sbuf"})
    assert dep.writes[1] == frozenset({"out"})


def test_build_dep_graph_rmsnorm_matmul_shared_producer_has_multiple_consumers() -> None:
    """sbuf_lhs (the Load's output) is consumed by both ActivationReduce and TensorScalar."""
    g = parse_and_resolve(_rmsnorm_matmul, _RMSNORM_SPECS)
    dep = build_dep_graph(g.ops, g.tensors)
    """Op indices: 0=Load(lhs), 1=Load(rhs), 2=ActivationReduce,
       3=Activation, 4=TensorScalar, 5=Transpose, 6=Matmul, 7=Store."""
    assert dep.producer["lhs_sbuf"] == 0
    assert dep.consumers["lhs_sbuf"] == (2, 4)


def test_build_dep_graph_param_tensors_have_none_producer() -> None:
    """Parameter tensors never have a producer; they appear with producer=None."""
    g = parse_and_resolve(_rmsnorm_matmul, _RMSNORM_SPECS)
    dep = build_dep_graph(g.ops, g.tensors)
    assert dep.producer["lhs"] is None
    assert dep.producer["rhs"] is None


def test_build_dep_graph_every_tensor_has_producer_key() -> None:
    """Every tensor in op_graph.tensors appears as a producer key."""
    g = parse_and_resolve(_rmsnorm_matmul, _RMSNORM_SPECS)
    dep = build_dep_graph(g.ops, g.tensors)
    for name in g.tensors:
        assert name in dep.producer


def test_build_dep_graph_rejects_duplicate_writes() -> None:
    """Two ops writing the same tensor name raise ValueError."""
    import pytest

    from nkigym.codegen.graph import ParsedOp

    class _FakeOpCls:
        OPERAND_AXES = {"data": ("P",)}
        OUTPUT_AXES = {"output": ("P",)}

    op_a = ParsedOp(
        idx=0,
        op_cls=_FakeOpCls,
        operand_names={"data": "x"},
        op_kwargs={},
        output_names=["y"],
        axis_map={"P": "d0"},
        touched_dims=("d0",),
        dim_role={},
    )
    op_b = ParsedOp(
        idx=1,
        op_cls=_FakeOpCls,
        operand_names={"data": "x"},
        op_kwargs={},
        output_names=["y"],
        axis_map={"P": "d0"},
        touched_dims=("d0",),
        dim_role={},
    )
    with pytest.raises(ValueError, match="duplicate write"):
        build_dep_graph([op_a, op_b], tensors={"x": None, "y": None})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/codegen/test_dep_graph.py -v
```

Expected: FAIL with `ImportError: cannot import name 'build_dep_graph'`.

- [ ] **Step 3: Implement `build_dep_graph`**

Append to `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/dep_graph.py`:

```python
from collections.abc import Mapping

from nkigym.codegen.graph import ParsedOp


def build_dep_graph(ops: list[ParsedOp], tensors: Mapping[str, object]) -> DepGraph:
    """Build the :class:`DepGraph` for a parsed op list.

    Walks ``ops`` once; for each op derives its read set from
    ``op_cls.OPERAND_AXES`` slots (restricted to slots actually bound
    in ``operand_names``) and its write set from ``output_names``.

    Args:
        ops: Parsed ops in source order.
        tensors: All tensor names present in the op graph. Used only
            to pre-populate ``producer`` with ``None`` for tensors
            never written (parameters).

    Returns:
        A fully resolved :class:`DepGraph`.

    Raises:
        ValueError: Two ops declare the same tensor name in their
            ``output_names`` — SSA violation.
    """
    reads: dict[int, frozenset[str]] = {}
    writes: dict[int, frozenset[str]] = {}
    producer: dict[str, int | None] = {name: None for name in tensors}
    consumers_mut: dict[str, list[int]] = {name: [] for name in tensors}
    for op in ops:
        op_reads: set[str] = set()
        for slot in op.op_cls.OPERAND_AXES:
            if slot in op.operand_names:
                op_reads.add(op.operand_names[slot])
        op_writes = set(op.output_names)
        reads[op.idx] = frozenset(op_reads)
        writes[op.idx] = frozenset(op_writes)
        for t in op_writes:
            if producer.get(t) is not None:
                raise ValueError(f"duplicate write on tensor {t!r}: ops {producer[t]} and {op.idx}")
            producer[t] = op.idx
        for t in op_reads:
            consumers_mut.setdefault(t, []).append(op.idx)
    consumers = {t: tuple(idxs) for t, idxs in consumers_mut.items()}
    return DepGraph(producer=producer, consumers=consumers, reads=reads, writes=writes)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test/codegen/test_dep_graph.py -v
```

Expected: PASS (all seven tests).

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/dep_graph.py test/codegen/test_dep_graph.py
git commit -m "codegen: implement build_dep_graph"
```

---

## Task 3: Implement subtree helpers

**Files:**
- Modify: `nkigym/src/nkigym/codegen/dep_graph.py`
- Modify: `test/codegen/test_dep_graph.py`

- [ ] **Step 1: Write the failing tests**

Append to `/home/ubuntu/nki-autotune/test/codegen/test_dep_graph.py`:

```python
from nkigym.codegen.dep_graph import commutes, subtree_ops, subtree_reads, subtree_writes
from nkigym.codegen.loop_forest import BodyLeaf, LoopNode
from nkigym.ops.base import AxisRole


def _leaf(op_idx: int) -> BodyLeaf:
    return BodyLeaf(op_idx=op_idx)


def _singleton_loop(op_idx: int) -> LoopNode:
    return LoopNode(dim_id="d0", trip_count=1, role=AxisRole.PARALLEL, children=[_leaf(op_idx)])


def _dep_two_ops(r0: set[str], w0: set[str], r1: set[str], w1: set[str]) -> DepGraph:
    """Build a DepGraph for two ops with hand-picked read/write sets."""
    producer: dict[str, int | None] = {}
    for t in w0:
        producer[t] = 0
    for t in w1:
        producer[t] = 1
    return DepGraph(
        producer=producer,
        consumers={},
        reads={0: frozenset(r0), 1: frozenset(r1)},
        writes={0: frozenset(w0), 1: frozenset(w1)},
    )


def test_subtree_ops_collects_body_leaves() -> None:
    """subtree_ops walks to every BodyLeaf under the given subtree."""
    node = LoopNode(
        dim_id="d0",
        trip_count=4,
        role=AxisRole.PARALLEL,
        children=[_leaf(7), LoopNode(dim_id="d1", trip_count=2, role=AxisRole.PARALLEL, children=[_leaf(9)])],
    )
    assert subtree_ops(node) == frozenset({7, 9})


def test_subtree_ops_on_leaf_returns_singleton() -> None:
    """Passing a BodyLeaf directly returns just its op_idx."""
    assert subtree_ops(_leaf(3)) == frozenset({3})


def test_subtree_reads_writes_unions_per_op_edges() -> None:
    """subtree_reads/subtree_writes union per-op maps over all leaves."""
    dep = _dep_two_ops(r0={"a"}, w0={"b"}, r1={"c"}, w1={"d"})
    node = LoopNode(
        dim_id="d0",
        trip_count=4,
        role=AxisRole.PARALLEL,
        children=[_leaf(0), _leaf(1)],
    )
    assert subtree_reads(node, dep) == frozenset({"a", "c"})
    assert subtree_writes(node, dep) == frozenset({"b", "d"})


def test_commutes_accepts_disjoint_pair() -> None:
    """Two subtrees touching disjoint tensor sets commute."""
    dep = _dep_two_ops(r0={"a"}, w0={"b"}, r1={"c"}, w1={"d"})
    assert commutes(_singleton_loop(0), _singleton_loop(1), dep) is True


def test_commutes_rejects_raw_edge() -> None:
    """RAW: a writes X, b reads X — does not commute."""
    dep = _dep_two_ops(r0=set(), w0={"x"}, r1={"x"}, w1=set())
    assert commutes(_singleton_loop(0), _singleton_loop(1), dep) is False


def test_commutes_rejects_war_edge() -> None:
    """WAR: a reads X, b writes X — does not commute."""
    dep = _dep_two_ops(r0={"x"}, w0=set(), r1=set(), w1={"x"})
    assert commutes(_singleton_loop(0), _singleton_loop(1), dep) is False


def test_commutes_rejects_waw_edge() -> None:
    """WAW: a writes X, b writes X — does not commute.

    Note this case cannot arise from a real OpGraph (SSA forbids
    duplicate writes, which ``build_dep_graph`` enforces), but the
    subtree ``commutes`` predicate must still reject it for safety
    under hand-built test forests and future rewrites that might
    introduce mutation.
    """
    dep = _dep_two_ops(r0=set(), w0={"x"}, r1=set(), w1={"x"})
    assert commutes(_singleton_loop(0), _singleton_loop(1), dep) is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/codegen/test_dep_graph.py -v
```

Expected: FAIL with `ImportError: cannot import name 'subtree_ops'`.

- [ ] **Step 3: Implement the helpers**

Append to `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/dep_graph.py`:

```python
from nkigym.codegen.loop_forest import BodyLeaf, LoopNode


def subtree_ops(node: "LoopNode | BodyLeaf") -> frozenset[int]:
    """Return the set of ``op_idx`` values appearing in ``BodyLeaf`` descendants of ``node``.

    A ``BodyLeaf`` contributes its own ``op_idx``; a ``LoopNode``
    contributes the union over its children.
    """
    collected: set[int] = set()
    _collect_op_idx(node, collected)
    return frozenset(collected)


def _collect_op_idx(node: "LoopNode | BodyLeaf", into: set[int]) -> None:
    """Recursive helper for :func:`subtree_ops`."""
    if isinstance(node, BodyLeaf):
        into.add(node.op_idx)
        return
    for child in node.children:
        _collect_op_idx(child, into)


def subtree_reads(node: "LoopNode | BodyLeaf", dep: DepGraph) -> frozenset[str]:
    """Return the union of ``dep.reads`` over every ``BodyLeaf`` under ``node``."""
    result: set[str] = set()
    for op_idx in subtree_ops(node):
        result |= dep.reads.get(op_idx, frozenset())
    return frozenset(result)


def subtree_writes(node: "LoopNode | BodyLeaf", dep: DepGraph) -> frozenset[str]:
    """Return the union of ``dep.writes`` over every ``BodyLeaf`` under ``node``."""
    result: set[str] = set()
    for op_idx in subtree_ops(node):
        result |= dep.writes.get(op_idx, frozenset())
    return frozenset(result)


def commutes(a: "LoopNode | BodyLeaf", b: "LoopNode | BodyLeaf", dep: DepGraph) -> bool:
    """Return ``True`` iff subtrees ``a`` and ``b`` share no RAW, WAR, or WAW edge.

    Two subtrees commute when the respective read/write sets are
    pair-wise disjoint across all three conflict flavours:

    * RAW: ``writes(a) ∩ reads(b)``
    * WAR: ``reads(a) ∩ writes(b)``
    * WAW: ``writes(a) ∩ writes(b)``
    """
    wa = subtree_writes(a, dep)
    ra = subtree_reads(a, dep)
    wb = subtree_writes(b, dep)
    rb = subtree_reads(b, dep)
    return not (wa & rb) and not (ra & wb) and not (wa & wb)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test/codegen/test_dep_graph.py -v
```

Expected: PASS (all thirteen tests).

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/dep_graph.py test/codegen/test_dep_graph.py
git commit -m "codegen: add DepGraph subtree helpers"
```

---

## Task 4: Wire `DepGraph` into `OpGraph` and `parse_and_resolve`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/graph.py`
- Modify: `test/codegen/test_graph.py`

- [ ] **Step 1: Write the failing test**

Append to `/home/ubuntu/nki-autotune/test/codegen/test_graph.py`:

```python
def test_parse_and_resolve_attaches_dep_graph() -> None:
    """parse_and_resolve builds and attaches a DepGraph to the returned OpGraph."""
    from nkigym.codegen.dep_graph import DepGraph
    from nkigym.codegen.graph import parse_and_resolve

    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    g = parse_and_resolve(_matmul_func, specs)
    assert isinstance(g.dep, DepGraph)
    """Load(lhs) at idx=0 reads lhs, writes lhs_sbuf."""
    assert g.dep.reads[0] == frozenset({"lhs"})
    assert g.dep.writes[0] == frozenset({"lhs_sbuf"})
    """Matmul at idx=3 reads lhs_T and rhs_sbuf, writes prod."""
    assert g.dep.reads[3] == frozenset({"lhs_T", "rhs_sbuf"})
    assert g.dep.writes[3] == frozenset({"prod"})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/codegen/test_graph.py -v -k test_parse_and_resolve_attaches_dep_graph
```

Expected: FAIL with `AttributeError: 'OpGraph' object has no attribute 'dep'`.

- [ ] **Step 3: Modify `OpGraph` to carry `dep`**

Edit `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/graph.py`.

At the top of the file, add an import for `DepGraph`:

```python
from nkigym.codegen.dep_graph import DepGraph, build_dep_graph
```

This import lives near the other `from ...` imports at the top. Place it *after* the `numpy` / `NKIOp` imports so the ordering is alphabetical-by-module within the nkigym block.

Extend the `OpGraph` dataclass (around line 123–147) by adding the `dep` field:

```python
@dataclass
class OpGraph:
    """Read-only resolved view of an ``f_nkigym`` function.

    Attributes:
        func_name: Function name (lands on the emitted kernel).
        param_names: Kernel parameters in signature order.
        return_name: Tensor name of the return value (the ``NKIStore``
            output).
        tensors: All named tensors, keyed by name.
        dims: All dims, keyed by dim id.
        ops: Parsed ops in source order.
        per_op_attrs: Per-op annotation side-table keyed by
            ``ParsedOp.idx``. Empty by default — reserved for future
            passes like ``propagate_compute_skip``.
        dep: Op-level dependency graph derived from ``ops`` and
            ``tensors``. Built once by ``parse_and_resolve``.
    """

    func_name: str
    param_names: list[str]
    return_name: str
    tensors: dict[str, Tensor]
    dims: dict[str, DimInfo]
    ops: list[ParsedOp]
    per_op_attrs: dict[int, dict[str, Any]] = field(default_factory=dict)
    dep: DepGraph = field(default_factory=lambda: DepGraph(producer={}, consumers={}, reads={}, writes={}))
```

In `parse_and_resolve` (around line 362–371), update the return to include `dep`:

```python
    return OpGraph(
        func_name=unwrapped.__name__,
        param_names=param_names,
        return_name=return_name,
        tensors=tensors,
        dims=dims,
        ops=ops,
        per_op_attrs={},
        dep=build_dep_graph(ops, tensors),
    )
```

- [ ] **Step 4: Run targeted test to verify it passes**

```bash
pytest test/codegen/test_graph.py -v -k test_parse_and_resolve_attaches_dep_graph
```

Expected: PASS.

- [ ] **Step 5: Run the full test_graph.py suite to confirm no regressions**

```bash
pytest test/codegen/test_graph.py -v
```

Expected: all existing tests still pass.

- [ ] **Step 6: Check for circular imports in dep_graph.py**

`dep_graph.py` currently imports from `graph.py` (for `ParsedOp`) and from `loop_forest.py` (for `BodyLeaf`/`LoopNode`). `graph.py` now imports from `dep_graph.py`. Verify the module loads cleanly:

```bash
python -c "from nkigym.codegen.graph import parse_and_resolve; from nkigym.codegen.dep_graph import build_dep_graph; print('ok')"
```

Expected: `ok`.

If this fails with `ImportError: cannot import name 'ParsedOp' ...` due to circular import, resolve by moving the `ParsedOp` import in `dep_graph.py` to a local import inside `build_dep_graph` (keep the `LoopNode`/`BodyLeaf` top-level since they're needed by the subtree helpers).

- [ ] **Step 7: Commit**

```bash
git add nkigym/src/nkigym/codegen/graph.py test/codegen/test_graph.py
git commit -m "codegen: attach DepGraph to OpGraph in parse_and_resolve"
```

---

## Task 5: Widen `FuseLoops.is_legal` to accept non-adjacent boundaries

**Files:**
- Modify: `nkigym/src/nkigym/tune/fuse_loops.py`
- Modify: `test/codegen/test_tune.py`

- [ ] **Step 1: Write the failing test (topological-adjacency acceptance)**

Append to `/home/ubuntu/nki-autotune/test/codegen/test_tune.py`:

```python
def test_is_legal_accepts_topologically_adjacent_pair_with_independent_intervening() -> None:
    """A pair (i, j) with j > i+1 is legal when every intervening sibling commutes with both endpoints.

    On the canonical rmsnorm+matmul forest, siblings 2 (ActivationReduce
    writing sum_sq from lhs_sbuf) and 4 (TensorScalar writing lhs_rms
    from lhs_sbuf + rms_inv) share d0 PARALLEL. Sibling 3 (Activation
    writing rms_inv from sum_sq) reads sum_sq and writes rms_inv — the
    Activation depends on ActivationReduce (RAW via sum_sq), so the
    pair (2, 4) is NOT topologically adjacent. Use a simpler hand-built
    forest to exercise the acceptance path.
    """
    from nkigym.codegen.dep_graph import DepGraph
    from nkigym.codegen.graph import OpGraph

    dep = DepGraph(
        producer={"a": 0, "b": 1, "c": 2, "d_out": 2},
        consumers={"a": (2,), "b": (), "c": ()},
        reads={0: frozenset(), 1: frozenset(), 2: frozenset({"a"})},
        writes={0: frozenset({"a"}), 1: frozenset({"b"}), 2: frozenset({"d_out"})},
    )
    og = OpGraph(func_name="f", param_names=[], return_name="d_out", tensors={}, dims={}, ops=[], dep=dep)
    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=2)]),
    ]
    atom = FuseLoops(path=(), boundary=(0, 2), dim_id="d0")
    assert atom.is_legal(og, forest) is True


def test_is_legal_rejects_topologically_non_adjacent_pair() -> None:
    """A pair (i, j) with an intervening sibling that depends on i is illegal.

    Ops: 0 writes a. 1 reads a, writes b. 2 reads a, writes c.
    Sibling 1 has a RAW edge with sibling 0, so sibling 1 cannot pass
    the producer (sibling 0) to move left. Fuse (0, 2) must be rejected.
    """
    from nkigym.codegen.dep_graph import DepGraph
    from nkigym.codegen.graph import OpGraph

    dep = DepGraph(
        producer={"a": 0, "b": 1, "c": 2},
        consumers={"a": (1, 2), "b": (), "c": ()},
        reads={0: frozenset(), 1: frozenset({"a"}), 2: frozenset({"a"})},
        writes={0: frozenset({"a"}), 1: frozenset({"b"}), 2: frozenset({"c"})},
    )
    og = OpGraph(func_name="f", param_names=[], return_name="c", tensors={}, dims={}, ops=[], dep=dep)
    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=2)]),
    ]
    atom = FuseLoops(path=(), boundary=(0, 2), dim_id="d0")
    assert atom.is_legal(og, forest) is False


def test_is_legal_rejects_boundary_with_role_mismatch_on_endpoints() -> None:
    """Topological-adjacency check does not override the three-field rule.

    Siblings 0 (PARALLEL) and 2 (ACCUMULATION) on d0 with an independent
    sibling 1 in between. The pair fails the role check regardless of
    topology."""
    from nkigym.codegen.dep_graph import DepGraph
    from nkigym.codegen.graph import OpGraph

    dep = DepGraph(
        producer={},
        consumers={},
        reads={0: frozenset(), 1: frozenset(), 2: frozenset()},
        writes={0: frozenset(), 1: frozenset(), 2: frozenset()},
    )
    og = OpGraph(func_name="f", param_names=[], return_name="x", tensors={}, dims={}, ops=[], dep=dep)
    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)]),
        LoopNode("d0", 4, AxisRole.ACCUMULATION, [BodyLeaf(op_idx=2)]),
    ]
    atom = FuseLoops(path=(), boundary=(0, 2), dim_id="d0")
    assert atom.is_legal(og, forest) is False
```

- [ ] **Step 2: Update existing tests that construct `FuseLoops(path, (i, i+1), dim_id)` but pass `op_graph=None` through to `is_legal`**

The existing tests call `.is_legal(None, forest)` for hand-built forests without a real OpGraph. The generalized legality rule still allows `None` when the boundary is literally adjacent (no intervening siblings → no topological check needed). Confirm by searching:

```bash
grep -n "is_legal(None" /home/ubuntu/nki-autotune/test/codegen/test_tune.py
```

Every existing `is_legal(None, forest)` call uses `boundary=(i, i+1)` — leave them alone. Implementation in Step 3 must short-circuit the topological check when `j == i+1` so `op_graph=None` remains valid for adjacent pairs.

- [ ] **Step 3: Run tests to verify the new ones fail**

```bash
pytest test/codegen/test_tune.py -v -k "topologically or role_mismatch_on_endpoints"
```

Expected: the three new tests FAIL (likely with incorrect rejection or incorrect acceptance depending on current logic).

- [ ] **Step 4: Widen `FuseLoops.is_legal`**

Edit `/home/ubuntu/nki-autotune/nkigym/src/nkigym/tune/fuse_loops.py`. Replace the `is_legal` method (lines 48–54):

```python
    def is_legal(self, op_graph: OpGraph, forest: LoopForest) -> bool:
        """Return ``True`` when the atom can be applied at ``(path, boundary)``.

        Three layers:

        1. Resolve ``path``; bail if stale.
        2. Three-field match on the endpoints: same ``dim_id``, same
           ``trip_count``, both ``PARALLEL``.
        3. Topological adjacency: every sibling strictly between ``i``
           and ``j`` must commute with BOTH endpoints — so it can be
           pushed to the left of the producer without breaking any
           RAW / WAR / WAW edge. Skipped when ``j == i + 1``.

        Layer 3 consults ``op_graph.dep``; when ``j == i + 1`` the
        topological check is vacuous and ``op_graph`` may be ``None``
        (supported for hand-built test forests).
        """
        siblings = _resolve_siblings(forest, self.path)
        if siblings is None:
            return False
        if not _pair_is_fusable(siblings, self.boundary, self.dim_id):
            return False
        i, j = self.boundary
        if j == i + 1:
            return True
        return _intervening_siblings_commute(op_graph, siblings, i, j)
```

Add a new helper near the bottom of the file (above `_merge_pair` or just after `_pair_is_fusable`):

```python
def _intervening_siblings_commute(
    op_graph: OpGraph,
    siblings: list[LoopNode | BodyLeaf],
    i: int,
    j: int,
) -> bool:
    """Return ``True`` when every sibling strictly between ``i`` and ``j``
    commutes with both endpoints.

    Consults ``op_graph.dep`` to form RAW/WAR/WAW judgments at
    subtree granularity. Callers must have already verified the
    three-field match on the endpoints; this helper is only concerned
    with the movement legality of the intervening siblings.
    """
    from nkigym.codegen.dep_graph import commutes

    dep = op_graph.dep
    producer = siblings[i]
    consumer = siblings[j]
    for k in range(i + 1, j):
        survivor = siblings[k]
        if not commutes(survivor, producer, dep):
            return False
        if not commutes(survivor, consumer, dep):
            return False
    return True
```

`_pair_is_fusable` stays as-is; the existing function already enforces the three-field rule. The only shape change to it is the old rejection of `j != i + 1` in line 116 — relax that guard.

Replace `_pair_is_fusable` (around lines 108–129):

```python
def _pair_is_fusable(siblings: list[LoopNode | BodyLeaf], boundary: tuple[int, int], dim_id: str | None) -> bool:
    """Check the three-field fusion rule on a specific pair.

    When ``dim_id`` is ``None`` the check accepts any shared dim
    (enumerator path). When ``dim_id`` is specified (``is_legal``
    path) the pair must also match it. The pair is identified by
    ``boundary = (i, j)`` with ``i < j``; non-adjacent pairs are
    accepted here — the topological-adjacency check lives in
    :func:`_intervening_siblings_commute`.
    """
    i, j = boundary
    if not (0 <= i < j < len(siblings)):
        return False
    a = siblings[i]
    b = siblings[j]
    if not isinstance(a, LoopNode) or not isinstance(b, LoopNode):
        return False
    if dim_id is not None and (a.dim_id != dim_id or b.dim_id != dim_id):
        return False
    return (
        a.dim_id == b.dim_id
        and a.role == AxisRole.PARALLEL
        and b.role == AxisRole.PARALLEL
        and a.trip_count == b.trip_count
    )
```

- [ ] **Step 5: Run the new tests to verify they pass**

```bash
pytest test/codegen/test_tune.py -v -k "topologically or role_mismatch_on_endpoints"
```

Expected: PASS.

- [ ] **Step 6: Run the full `test_tune.py` suite**

```bash
pytest test/codegen/test_tune.py -v
```

Expected: all existing tests still pass. Note the `test_is_legal_rejects_non_adjacent_boundary` test at line 82 — previously that atom was rejected solely because `j != i + 1`; now it's rejected because sibling 3 (Activation) has a RAW dependency on sibling 2 (ActivationReduce) via `sum_sq`. The assertion still holds; the rejection path is now topological instead of structural.

- [ ] **Step 7: Commit**

```bash
git add nkigym/src/nkigym/tune/fuse_loops.py test/codegen/test_tune.py
git commit -m "tune: widen FuseLoops.is_legal to topological adjacency"
```

---

## Task 6: Rewrite `FuseLoops.apply` to handle intervening siblings

**Files:**
- Modify: `nkigym/src/nkigym/tune/fuse_loops.py`
- Modify: `test/codegen/test_tune.py`

- [ ] **Step 1: Write the failing tests**

Append to `/home/ubuntu/nki-autotune/test/codegen/test_tune.py`:

```python
def test_apply_topological_fuse_pushes_survivors_left_and_lands_at_consumer_slot() -> None:
    """Fusing (0, 2) with an independent sibling at 1:
       forest goes from [A, B, C] to [B, fused(A ‖ C)].
    """
    from nkigym.codegen.dep_graph import DepGraph
    from nkigym.codegen.graph import OpGraph

    dep = DepGraph(
        producer={"a": 0, "b": 1, "c": 2},
        consumers={"a": (2,), "b": (), "c": ()},
        reads={0: frozenset(), 1: frozenset(), 2: frozenset({"a"})},
        writes={0: frozenset({"a"}), 1: frozenset({"b"}), 2: frozenset({"c"})},
    )
    og = OpGraph(func_name="f", param_names=[], return_name="c", tensors={}, dims={}, ops=[], dep=dep)
    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0, phase="A")]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1, phase="B")]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=2, phase="C")]),
    ]
    atom = FuseLoops(path=(), boundary=(0, 2), dim_id="d0")
    assert atom.is_legal(og, forest) is True
    _, new_forest = atom.apply(og, forest)
    assert len(new_forest) == 2
    """B slid left of the former producer position."""
    assert isinstance(new_forest[0], LoopNode)
    assert new_forest[0].children[0].phase == "B"
    """Fused nest lands at consumer slot with producer-body first."""
    assert isinstance(new_forest[1], LoopNode)
    assert new_forest[1].dim_id == "d0"
    assert new_forest[1].role is AxisRole.PARALLEL
    assert [child.phase for child in new_forest[1].children] == ["A", "C"]


def test_apply_topological_fuse_preserves_survivor_relative_order() -> None:
    """Two intervening siblings keep their relative order in the survivor list.

    Forest: [A, B, C, D] where A (writes a), D (reads a), B and C
    both independent. Fuse (0, 3) pushes [B, C] left of A's position
    preserving order — result: [B, C, fused(A ‖ D)].
    """
    from nkigym.codegen.dep_graph import DepGraph
    from nkigym.codegen.graph import OpGraph

    dep = DepGraph(
        producer={"a": 0, "b": 1, "c": 2, "d_out": 3},
        consumers={"a": (3,)},
        reads={0: frozenset(), 1: frozenset(), 2: frozenset(), 3: frozenset({"a"})},
        writes={0: frozenset({"a"}), 1: frozenset({"b"}), 2: frozenset({"c"}), 3: frozenset({"d_out"})},
    )
    og = OpGraph(func_name="f", param_names=[], return_name="d_out", tensors={}, dims={}, ops=[], dep=dep)
    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0, phase="A")]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1, phase="B")]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=2, phase="C")]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=3, phase="D")]),
    ]
    atom = FuseLoops(path=(), boundary=(0, 3), dim_id="d0")
    assert atom.is_legal(og, forest) is True
    _, new_forest = atom.apply(og, forest)
    phases = [n.children[0].phase for n in new_forest]
    assert phases[:2] == ["B", "C"]
    """Fused nest at new index 2 holds A then D."""
    assert [c.phase for c in new_forest[2].children] == ["A", "D"]


def test_apply_literal_adjacent_fuse_unchanged_behaviour() -> None:
    """For j == i+1 the generalized apply behaves identically to the previous implementation.

    Explicit regression: fused nest lands at j's slot with [*A.children,
    *B.children]; no siblings moved; length shrinks by one.
    """
    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0, phase="A0")]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1, phase="B0")]),
    ]
    atom = FuseLoops(path=(), boundary=(0, 1), dim_id="d0")
    _, new_forest = atom.apply(None, forest)
    assert len(new_forest) == 1
    merged = new_forest[0]
    assert isinstance(merged, LoopNode)
    assert [c.phase for c in merged.children] == ["A0", "B0"]
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
pytest test/codegen/test_tune.py -v -k "topological_fuse or literal_adjacent_fuse_unchanged"
```

Expected: `topological_fuse_*` tests FAIL because the current `apply` / `_merge_pair` / `_rewrite_forest` pipeline only handles `(i, i+1)`. The `literal_adjacent_fuse_unchanged_behaviour` test should PASS with current code (regression baseline).

- [ ] **Step 3: Rewrite `apply` / `_rewrite_forest` / `_merge_pair`**

Edit `/home/ubuntu/nki-autotune/nkigym/src/nkigym/tune/fuse_loops.py`. Replace `apply` (lines 56–64):

```python
    def apply(self, op_graph: OpGraph, forest: LoopForest) -> tuple[OpGraph, LoopForest]:
        """Fuse the pair at ``(path, boundary)``; consumer absorbs producer.

        Siblings strictly between ``i`` and ``j`` slide to the left of
        ``i``'s original position, keeping their relative order. The
        fused ``LoopNode`` lands at ``j``'s original slot. Its children
        are ``producer.children ++ consumer.children`` — producer body
        first, preserving the RAW edge. The fused loop inherits the
        consumer's ``name``. ``op_graph`` passes through unchanged.
        """
        new_forest = _rewrite_forest(forest, self.path, self.boundary, self.dim_id)
        return op_graph, new_forest
```

Replace `_rewrite_forest` and `_merge_pair` (around lines 132–172):

```python
def _rewrite_forest(
    forest: LoopForest, path: tuple[int, ...], boundary: tuple[int, int], dim_id: str
) -> LoopForest:
    """Return a new forest with the pair at ``(path, boundary)`` merged.

    Structural sharing: subtrees outside the rewrite site are passed
    through by reference, not deep-copied.
    """
    if not path:
        return _apply_fuse_in_siblings(forest, boundary, dim_id)
    idx, rest = path[0], path[1:]
    parent = forest[idx]
    assert isinstance(parent, LoopNode)
    new_children = _rewrite_forest(parent.children, rest, boundary, dim_id)
    new_parent = LoopNode(
        dim_id=parent.dim_id,
        trip_count=parent.trip_count,
        role=parent.role,
        children=new_children,
        reduce_op=parent.reduce_op,
        name=parent.name,
    )
    return [*forest[:idx], new_parent, *forest[idx + 1 :]]


def _apply_fuse_in_siblings(
    siblings: list[LoopNode | BodyLeaf], boundary: tuple[int, int], dim_id: str
) -> list[LoopNode | BodyLeaf]:
    """Merge one pair inside ``siblings`` — producer absorbed by consumer.

    Layout after apply:
        siblings[:i] ++ survivors ++ [fused] ++ siblings[j+1:]

    where ``survivors = siblings[i+1 : j]`` keeps the original
    relative order. The fused ``LoopNode`` lands at ``j``'s original
    slot; its children are
    ``producer.children ++ consumer.children``.

    For ``j == i + 1`` ``survivors`` is empty and the output matches
    the literal-adjacent case byte-for-byte.
    """
    i, j = boundary
    producer = siblings[i]
    consumer = siblings[j]
    assert isinstance(producer, LoopNode) and isinstance(consumer, LoopNode)
    survivors = list(siblings[i + 1 : j])
    merged = LoopNode(
        dim_id=dim_id,
        trip_count=consumer.trip_count,
        role=AxisRole.PARALLEL,
        children=[*producer.children, *consumer.children],
        name=consumer.name,
    )
    return [*siblings[:i], *survivors, merged, *siblings[j + 1 :]]
```

Note: `_merge_pair` is removed (folded into `_apply_fuse_in_siblings`). The renaming matches the broader semantics: we're merging a pair *within a siblings list*, not "merging a pair" in isolation.

- [ ] **Step 4: Run the new tests to verify they pass**

```bash
pytest test/codegen/test_tune.py -v -k "topological_fuse or literal_adjacent_fuse_unchanged"
```

Expected: PASS.

- [ ] **Step 5: Run the full `test_tune.py` suite**

```bash
pytest test/codegen/test_tune.py -v
```

Expected: all tests pass. The existing `test_apply_*` cases (literal-adjacent) are unchanged.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/tune/fuse_loops.py test/codegen/test_tune.py
git commit -m "tune: rewrite FuseLoops.apply for topological-adjacent fuse"
```

---

## Task 7: Update `enumerate_fusion_atoms` signature and logic

**Files:**
- Modify: `nkigym/src/nkigym/tune/fuse_loops.py`
- Modify: `nkigym/src/nkigym/tune/batch.py`
- Modify: `test/codegen/test_tune.py`
- Modify: `test/codegen/test_batch.py` (stub lambda signatures)

- [ ] **Step 1: Write the failing tests**

Append to `/home/ubuntu/nki-autotune/test/codegen/test_tune.py`:

```python
def test_enumerate_emits_topological_non_adjacent_pair_when_independent_sibling_in_between() -> None:
    """Enumerator emits (0, 2) when sibling 1 is independent of 0 and 2."""
    from nkigym.codegen.dep_graph import DepGraph
    from nkigym.codegen.graph import OpGraph

    dep = DepGraph(
        producer={"a": 0, "b": 1, "c": 2},
        consumers={"a": (2,)},
        reads={0: frozenset(), 1: frozenset(), 2: frozenset({"a"})},
        writes={0: frozenset({"a"}), 1: frozenset({"b"}), 2: frozenset({"c"})},
    )
    og = OpGraph(func_name="f", param_names=[], return_name="c", tensors={}, dims={}, ops=[], dep=dep)
    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=2)]),
    ]
    atoms = enumerate_fusion_atoms(og, forest)
    root_atoms = [a for a in atoms if a.path == ()]
    """(0, 1), (1, 2), and (0, 2) are all legal."""
    boundaries = {a.boundary for a in root_atoms}
    assert (0, 1) in boundaries
    assert (1, 2) in boundaries
    assert (0, 2) in boundaries


def test_enumerate_omits_pair_blocked_by_intervening_dependency() -> None:
    """When sibling 1 depends on sibling 0 (RAW), (0, 2) is omitted."""
    from nkigym.codegen.dep_graph import DepGraph
    from nkigym.codegen.graph import OpGraph

    dep = DepGraph(
        producer={"a": 0, "b": 1, "c": 2},
        consumers={"a": (1, 2)},
        reads={0: frozenset(), 1: frozenset({"a"}), 2: frozenset({"a"})},
        writes={0: frozenset({"a"}), 1: frozenset({"b"}), 2: frozenset({"c"})},
    )
    og = OpGraph(func_name="f", param_names=[], return_name="c", tensors={}, dims={}, ops=[], dep=dep)
    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=2)]),
    ]
    atoms = enumerate_fusion_atoms(og, forest)
    boundaries = {a.boundary for a in atoms if a.path == ()}
    assert (0, 2) not in boundaries
    """The literal-adjacent atoms remain legal — (0, 1) checks only the
    three-field rule; (1, 2) the same. Both independent of the
    intervening-sibling rule."""
    assert (0, 1) in boundaries
    assert (1, 2) in boundaries
```

Also update the existing `test_enumerate_fusion_atoms_rmsnorm_matmul_canonical_has_sensible_atoms` test signature expectation: after this task, `enumerate_fusion_atoms` takes `(op_graph, forest)`. Update the test body at lines 211–224:

Find:
```python
def test_enumerate_fusion_atoms_rmsnorm_matmul_canonical_has_sensible_atoms() -> None:
    """Canonical rmsnorm+matmul yields atoms at boundaries where both roots are PARALLEL on same dim."""
    g, forest = _canonical_rmsnorm_matmul()
    atoms = enumerate_fusion_atoms(forest)
```

Replace with:
```python
def test_enumerate_fusion_atoms_rmsnorm_matmul_canonical_has_sensible_atoms() -> None:
    """Canonical rmsnorm+matmul yields atoms at boundaries where both roots are PARALLEL on same dim."""
    g, forest = _canonical_rmsnorm_matmul()
    atoms = enumerate_fusion_atoms(g, forest)
```

And update the expected root atoms: with the topological rule, new pairs become legal. Op indices: 0=Load(lhs), 1=Load(rhs), 2=ActivationReduce(reads lhs_sbuf), 3=Activation(reads sum_sq), 4=TensorScalar(reads lhs_sbuf, rms_inv), 5=Transpose(reads lhs_rms), 6=Matmul(reads lhs_T, rhs_sbuf), 7=Store(reads prod).

Topologically-adjacent root pairs on `d0`:

- `(0, 2)`: Load(lhs) ↔ ActivationReduce. Sibling 1 (Load(rhs)) is independent → legal.
- `(2, 3)`, `(3, 4)`, `(6, 7)`: literal-adjacent, previously legal, still legal.
- `(0, 4)`: Load(lhs) ↔ TensorScalar. Siblings 1, 2, 3 intervene. Sibling 1 (Load(rhs)) is independent of both endpoints. Sibling 2 (ActivationReduce) reads lhs_sbuf (RAW with 0) — blocks.
- `(2, 6)`: ActivationReduce ↔ Matmul. ActivationReduce's dim_id on d0 matches Matmul's outer d0. But Matmul's outer d0 is M (mapped from lhs_T via Transpose). The Transpose (sibling 5) reads lhs_rms (written by TensorScalar, sibling 4) — dependent. Too many blockers.

Expect at minimum: `{(0, 2), (2, 3), (3, 4), (6, 7)}` on d0 at root. Replace the assertion:

Find:
```python
    root_atoms = [(a.boundary, a.dim_id) for a in atoms if a.path == ()]
    assert ((2, 3), "d0") in root_atoms
    assert ((3, 4), "d0") in root_atoms
    assert ((6, 7), "d0") in root_atoms
    """Exactly three root-level atoms."""
    assert len(root_atoms) == 3
```

Replace with:
```python
    root_atoms = [(a.boundary, a.dim_id) for a in atoms if a.path == ()]
    """Literal-adjacent fuses on d0 remain; the topological generalisation
    adds (0, 2) — Load(lhs) ↔ ActivationReduce with independent Load(rhs)
    in between."""
    assert ((0, 2), "d0") in root_atoms
    assert ((2, 3), "d0") in root_atoms
    assert ((3, 4), "d0") in root_atoms
    assert ((6, 7), "d0") in root_atoms
```

Drop the exact-count assertion — it's no longer stable under the topological generalisation and the spec does not require it. Presence of the expected atoms is the contract.

Similarly patch the other `enumerate_fusion_atoms(forest)` callers in the file. Run:

```bash
grep -n "enumerate_fusion_atoms(" /home/ubuntu/nki-autotune/test/codegen/test_tune.py
```

Replace every `enumerate_fusion_atoms(forest)` with `enumerate_fusion_atoms(None, forest)` — the `None` is the `op_graph` argument, and the `_intervening_siblings_commute` helper only consults it when a pair has intervening siblings. For test cases with 2-element forests or that iterate only `(i, i+1)` pairs, the check never runs and `None` is accepted. **Exception:** for the 3-sibling forest test `test_enumerate_finds_nested_sibling_pairs` (around line 199), the forest has only one outer node with two inner siblings — still `(i, i+1)` boundaries only, so `None` works there.

- [ ] **Step 2: Update `enumerate_fusion_atoms` signature**

Edit `/home/ubuntu/nki-autotune/nkigym/src/nkigym/tune/fuse_loops.py`. Replace the enumerator + recursive helper (around lines 67–88):

```python
def enumerate_fusion_atoms(op_graph: OpGraph | None, forest: LoopForest) -> list[FuseLoops]:
    """Return every legal :class:`FuseLoops` atom in ``forest``.

    Walks the forest recursively: at every children list (the forest
    itself plus every ``LoopNode.children``), enumerates every pair
    ``(i, j)`` with ``0 ≤ i < j < len(siblings)`` and emits one atom
    per pair that passes both the three-field match and the
    topological-adjacency check.

    Args:
        op_graph: Used for the topological-adjacency check when a pair
            has intervening siblings. May be ``None`` on hand-built
            test forests that only exercise literal-adjacent pairs;
            in that case a pair with ``j > i + 1`` is conservatively
            rejected.
        forest: The forest to enumerate over.

    Returns:
        List of atoms in depth-first order.
    """
    atoms: list[FuseLoops] = []
    _collect(op_graph, forest, path=(), atoms=atoms)
    return atoms


def _collect(
    op_graph: OpGraph | None,
    siblings: list[LoopNode | BodyLeaf],
    path: tuple[int, ...],
    atoms: list[FuseLoops],
) -> None:
    """Recursive helper for :func:`enumerate_fusion_atoms`."""
    n = len(siblings)
    for i in range(n):
        for j in range(i + 1, n):
            if not _pair_is_fusable(siblings, (i, j), dim_id=None):
                continue
            if j > i + 1:
                if op_graph is None:
                    continue
                if not _intervening_siblings_commute(op_graph, siblings, i, j):
                    continue
            a = siblings[i]
            assert isinstance(a, LoopNode)
            atoms.append(FuseLoops(path=path, boundary=(i, j), dim_id=a.dim_id))
    for idx, child in enumerate(siblings):
        if isinstance(child, LoopNode):
            _collect(op_graph, child.children, path=path + (idx,), atoms=atoms)
```

- [ ] **Step 3: Update both call sites in `batch.py`**

Edit `/home/ubuntu/nki-autotune/nkigym/src/nkigym/tune/batch.py`.

Find (around line 55):

```python
    frontier: dict[int, list[KernelRewrite]] = {h0: enumerate_fusion_atoms(forest) + enumerate_reorder_atoms(forest)}
```

Replace with:

```python
    frontier: dict[int, list[KernelRewrite]] = {
        h0: enumerate_fusion_atoms(op_graph, forest) + enumerate_reorder_atoms(forest)
    }
```

Find (around line 75):

```python
    new_atoms = enumerate_fusion_atoms(new_f) + enumerate_reorder_atoms(new_f)
```

Replace with:

```python
    new_atoms = enumerate_fusion_atoms(new_og, new_f) + enumerate_reorder_atoms(new_f)
```

- [ ] **Step 4: Update the `test_batch.py` monkeypatch lambdas**

Edit `/home/ubuntu/nki-autotune/test/codegen/test_batch.py`. Find (around line 54):

```python
    monkeypatch.setattr(batch_mod, "enumerate_fusion_atoms", lambda f: [])
```

Replace with:

```python
    monkeypatch.setattr(batch_mod, "enumerate_fusion_atoms", lambda og, f: [])
```

Check for other occurrences:

```bash
grep -n "enumerate_fusion_atoms" /home/ubuntu/nki-autotune/test/codegen/test_batch.py
```

Update any additional stub to match the new `(op_graph, forest)` signature.

- [ ] **Step 5: Run all affected tests**

```bash
pytest test/codegen/test_tune.py test/codegen/test_batch.py -v
```

Expected: all tests pass, including the two new enumerator tests.

- [ ] **Step 6: Run the full codegen test suite**

```bash
pytest test/codegen/ -v
```

Expected: every test passes.

- [ ] **Step 7: Commit**

```bash
git add nkigym/src/nkigym/tune/fuse_loops.py nkigym/src/nkigym/tune/batch.py \
    test/codegen/test_tune.py test/codegen/test_batch.py
git commit -m "tune: generalize enumerate_fusion_atoms to non-adjacent pairs"
```

---

## Task 8: CPU-sim correctness gate on the generalised fuse

**Files:**
- Create: `test/codegen/test_fuse_loops_cpu_sim.py`

- [ ] **Step 1: Write the failing tests**

Create `/home/ubuntu/nki-autotune/test/codegen/test_fuse_loops_cpu_sim.py`:

```python
"""CPU-sim correctness gate for topologically-adjacent FuseLoops.

Renders each post-fuse kernel to source via the existing codegen
pipeline, runs it through ``nki.simulate``, and compares against the
numpy reference at fp32 tolerance. The standard nkigym validation
contract: elementwise ``atol=rtol=1e-5`` for K≤4096 reducers.
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
from nkigym.tune.fuse_loops import FuseLoops

try:
    import nki
except ImportError:
    nki = None

_SEED = 0
_ATOL = 1e-5
_RTOL = 1e-5


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
    assert np.allclose(actual, expected, atol=_ATOL, rtol=_RTOL), (
        f"CPU-sim mismatch: max_abs_diff={float(np.abs(actual - expected).max()):.3e}"
    )


@pytest.mark.skipif(nki is None, reason="nki runtime not available")
def test_matmul_lhsT_rhs_topological_fuse_cpu_sim_matches_numpy() -> None:
    """Apply FuseLoops(path=(), boundary=(0, 2), dim_id='d0') — Load(lhs) ↔ Transpose —
    on a pure matmul. Sibling 1 (Load(rhs)) is independent; fuse is legal.
    """
    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 128), "bfloat16")}
    g = parse_and_resolve(_matmul_lhsT_rhs, specs)
    forest = build_canonical_forest(g)
    atom = FuseLoops(path=(), boundary=(0, 2), dim_id="d0")
    assert atom.is_legal(g, forest) is True
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
    from nkigym.tune.fuse_loops import enumerate_fusion_atoms

    g = parse_and_resolve(_rmsnorm_matmul, _SPECS)
    forest = build_canonical_forest(g)
    atom0 = FuseLoops(path=(), boundary=(0, 2), dim_id="d0")
    _, forest1 = atom0.apply(g, forest)
    next_atoms = [
        a for a in enumerate_fusion_atoms(g, forest1)
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
    from nkigym.tune.fuse_loops import enumerate_fusion_atoms

    g = parse_and_resolve(_rmsnorm_matmul, _SPECS)
    forest = build_canonical_forest(g)
    atoms = enumerate_fusion_atoms(g, forest)
    root_atoms = {a.boundary for a in atoms if a.path == ()}
    assert (2, 4) not in root_atoms
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest test/codegen/test_fuse_loops_cpu_sim.py -v
```

Expected: if `nki` is available, the three CPU-sim tests run and pass (Task 7 already landed the mechanics; this suite is a new independent verification). If any fail, investigate — likely indicates a subtle bug in `_apply_fuse_in_siblings` that the unit tests didn't catch.

If the tests fail due to `nki` not being importable at test-collection time, the skip marker handles it. The negative-control test (`test_enumerator_refuses_pair_blocked_by_raw_intervening_dependency`) does not depend on `nki` and must pass unconditionally.

- [ ] **Step 3: If any CPU-sim test fails**, diagnose via the rendered source

```bash
pytest test/codegen/test_fuse_loops_cpu_sim.py -v -k rmsnorm_matmul_topological -s
```

Add a temporary `print(source)` in the failing test to inspect the rendered kernel. Most likely failure modes:

- `name` on the merged `LoopNode` not preserved correctly → renderer emits a stale loop variable.
- Survivors slid left of `i`'s position but their subtrees share a buffer with the now-fused producer body → indicates the `commutes` check missed an edge.

Fix by cross-checking `_apply_fuse_in_siblings` against the design doc §3.3.

- [ ] **Step 4: Run the full codegen suite**

```bash
pytest test/codegen/ -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add test/codegen/test_fuse_loops_cpu_sim.py
git commit -m "test: CPU-sim correctness gate for topological FuseLoops"
```

---

## Task 9: End-to-end run on `examples/rmsnorm_matmul.py`

This is the acceptance gate. Runs the full `nkigym_compile` pipeline
(synthesis + initial_codegen + batch tune with 100 kernels) against
the generalised fuse and confirms no regressions.

**Files:**
- Read: `examples/rmsnorm_matmul.py`
- Read: `/home/ubuntu/cache/rmsnorm_matmul_compile/results.json` (after run)
- Read: `/home/ubuntu/cache/rmsnorm_matmul_compile/kernel_tuned_*.py` (after run)

- [ ] **Step 1: Run the example**

```bash
cd /home/ubuntu/nki-autotune
source ~/venvs/kernel-env/bin/activate
python examples/rmsnorm_matmul.py
```

Expected runtime: ~2–5 minutes on the kernel gym (three-host profile stage). Expected terminal output: the example prints `[rmsnorm_matmul] canonical kernel: ...` and `[rmsnorm_matmul] results.json: ...` paths on completion. No exceptions raised during CPU-sim gates.

If the run fails mid-way with a CPU-sim assertion error, stop and inspect the failing `kernel_tuned_*.py`. Do not continue past step 1 without a clean end-to-end run.

- [ ] **Step 2: Verify all sampled kernels passed CPU-sim**

```bash
python -c "
import json
from pathlib import Path
results = json.loads(Path('/home/ubuntu/cache/rmsnorm_matmul_compile/results.json').read_text())
failures = [r for r in results if r.get('cpu_sim_failed')]
print(f'total: {len(results)}, cpu_sim_failures: {len(failures)}')
assert not failures, f'cpu_sim failures: {[r[\"kernel_name\"] for r in failures]}'
"
```

Expected: `cpu_sim_failures: 0`. Any non-zero failure count signals the generalised enumerator emitted an illegal state that reached the CPU-sim gate.

- [ ] **Step 3: Confirm at least one kernel shows the topological fuse**

The `lhs_load ↔ rmsnorm` topological fuse merges sibling 0 (`nisa.dma_copy` for `sbuf_lhs`) into the outer-`d0_0` loop of the rmsnorm block. Rendered form: the kernel has *no separate* `for i_d0_0 in range(16):` wrapping a single `nisa.dma_copy` to `sbuf_lhs`; instead that dma_copy lives inside the rmsnorm's outer loop.

```bash
cd /home/ubuntu/cache/rmsnorm_matmul_compile
for f in kernel_tuned_*.py; do
    if grep -q "nisa.dma_copy(dst=sbuf_lhs" "$f"; then
        lhs_copy_count=$(grep -c "^    for i_d0_0 in range(16):$" "$f")
        if [ "$lhs_copy_count" -lt 3 ]; then
            echo "candidate $f: top-level i_d0_0 loops = $lhs_copy_count"
        fi
    fi
done | head -5
```

Expected: at least one candidate file printed. If no candidate appears, the topological fuse atom exists but the sampler never drew it in 100 kernels. Re-run with a different seed to confirm, then investigate — a stochastic sampler should land at least one such kernel in a pool of 100 (the atom is one of only ~4–6 root-level d0 atoms; probability of never sampling across 100 draws is negligible).

Manually inspect one candidate:

```bash
head -50 /home/ubuntu/cache/rmsnorm_matmul_compile/<candidate>.py
```

Verify by eye that the `nisa.dma_copy(dst=sbuf_lhs, ...)` call appears inside the rmsnorm outer loop (not as its own top-level `for i_d0_0 in range(16)` block).

- [ ] **Step 4: Sanity check MFU regression floor**

```bash
python -c "
import json
from pathlib import Path
results = json.loads(Path('/home/ubuntu/cache/rmsnorm_matmul_compile/results.json').read_text())
mfus = [r.get('mfu_estimated_percent') for r in results if r.get('mfu_estimated_percent') is not None]
print(f'count with mfu: {len(mfus)}, max: {max(mfus):.2f}%, median: {sorted(mfus)[len(mfus)//2]:.2f}%')
"
```

Expected: max MFU in the 75–82% range (current SOTA is 79.09%/80.57%). A max below ~65% signals a regression — investigate. A max in the expected range is acceptance.

Full MFU sweeps and perf wins from the topological fuse are a separate exercise; this step only catches catastrophic regressions.

- [ ] **Step 5: Commit the `examples/rmsnorm_matmul.py` change if any**

The existing file already has the right shape — nothing should need editing. If git shows `M examples/rmsnorm_matmul.py` from a prior local change (visible in the pre-task `git status`), inspect:

```bash
git diff examples/rmsnorm_matmul.py
```

If the diff is unrelated to this milestone, restore the file (`git restore examples/rmsnorm_matmul.py`). If it represents deliberate local work, leave it untouched — this plan does not modify the example script.

- [ ] **Step 6: Tag the acceptance gate**

No formal tag needed, but create a summary commit for the milestone:

```bash
git log --oneline origin/dev_1..HEAD
```

Expected: six or seven commits from Tasks 1–8 (one per step-5 commit).

---

## Task 10: Documentation and learnings update

**Files:**
- Modify: `/home/ubuntu/nki-autotune/.claude/rules/learnings.md`

- [ ] **Step 1: Add a one-line entry under "Architecture"**

Open `/home/ubuntu/nki-autotune/.claude/rules/learnings.md` and add a new bullet under the **Architecture** section (alongside the existing `FuseLoops`-related entries). The entry must be compact — match the surrounding one-line style and include the date.

Run `TZ='America/New_York' date +'%Y-%m-%d %H:%M ET'` to get the current Eastern-time timestamp, then insert the entry using that stamp:

```markdown
- **Topological `FuseLoops`**: `boundary=(i, j)` with `j > i+1` accepted when every intervening sibling commutes (no RAW/WAR/WAW) with both endpoints. Dep info via `OpGraph.dep: DepGraph` (producer/consumers/reads/writes; `operand_names ∩ OPERAND_AXES` vs `output_names`). Fuse = "consumer absorbs producer": survivors slide left, fused nest at consumer slot, children = `producer.children ++ consumer.children`, merged `name` from consumer. `enumerate_fusion_atoms(op_graph, forest)`. *(<timestamp>)*
```

Keep the entry under 500 characters, matching the existing style.

Update the "Last updated" line at the bottom:

Find:
```
*Last updated: 2026-05-06 01:25 ET*
```

Replace with the current timestamp.

- [ ] **Step 2: Commit**

```bash
git add .claude/rules/learnings.md
git commit -m "learnings: topological FuseLoops + DepGraph"
```

---

## Verification

After all tasks complete, run the full test suite once more to confirm clean state:

```bash
cd /home/ubuntu/nki-autotune
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/ -v
```

Expected: all green, no skips beyond the `nki`-availability guard on `test_fuse_loops_cpu_sim.py` (which should NOT skip in the primary environment).

Then confirm the end-to-end example is still green:

```bash
python examples/rmsnorm_matmul.py
```

Expected: runs to completion; `results.json` shows CPU-sim pass rate 100%, at least one topological-fuse candidate kernel, MFU in the expected range.

---

## Self-Review Notes

- Each task is a single conceptual change with its own commit.
- Every test has real code, not stubs.
- Every step has an exact command with expected output.
- Types (`OpGraph`, `DepGraph`, `FuseLoops`) are consistent across tasks.
- `enumerate_fusion_atoms` signature change flows through both `fuse_loops.py` and `batch.py`, with downstream test stubs updated in the same task.
- `op_graph=None` accepted for literal-adjacent pairs only — explicitly documented in the enumerator's docstring and in Task 7 Step 1 notes.
- Task 8 is the correctness gate; Task 9 is the acceptance gate. Neither is optional.
