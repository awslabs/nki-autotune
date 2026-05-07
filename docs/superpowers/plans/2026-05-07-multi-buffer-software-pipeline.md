# Multi-Buffer + Software Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the renderer's cross-loopnest buffer bloat bug and add two orthogonal rewrite atoms (`MultiBuffer`, `SoftwarePipeline`) to the tune stage so sampled kernels can shrink intra-loopnest buffers (fixing SBUF OOM) and overlap producer-consumer pairs via software pipelining.

**Architecture:** Buffer size is factored as `required_tiles * buffer_degree`. `required_tiles(tensor, d)` is derived at render time from `num_tiles(d) / lca_trip_product(tensor, d, forest, dep)` — fusion automatically shrinks intra-loopnest buffers. `buffer_degree[dim]` is a new persisted field on `Tensor`, defaulting to 1, controlled by the `MultiBuffer` atom. `LoopNode.pipeline_depth` is a new persisted field, defaulting to 1, controlled by `SoftwarePipeline`; when > 1, the renderer emits prologue + skewed body + epilogue with per-leaf iteration offsets. The frontier sampler gains a `hash_state(op_graph, forest)` hash covering both forest and per-tensor `buffer_degree`.

**Tech Stack:** Python 3.12, pytest, `black`, `isort`, `pyright`. Kernel venv at `~/venvs/kernel-env/bin/activate`. Tests live in `/home/ubuntu/nki-autotune/test/codegen/`; source in `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/` and `nkigym/src/nkigym/tune/`.

**Design doc:** `docs/superpowers/specs/2026-05-07-multi-buffer-software-pipeline-design.md`

---

## File Structure

New files:

- `test/codegen/_rmsnorm_matmul_fixture.py` — shared test fixture exposing `f_nkigym`, `INPUT_SPECS`, `f_numpy` for the rmsnorm+matmul example. Already created ahead of Task 1 because `examples/rmsnorm_matmul.py` defines these names only under `if __name__ == "__main__":` and tests cannot import from there. Every test below imports via `from test.codegen._rmsnorm_matmul_fixture import f_nkigym, INPUT_SPECS` (or `..., f_numpy`).
- `nkigym/src/nkigym/tune/multi_buffer.py` — `MultiBuffer` dataclass, `is_legal`, `apply`, `enumerate_multi_buffer_atoms`.
- `nkigym/src/nkigym/tune/software_pipeline.py` — `SoftwarePipeline` dataclass, `is_legal`, `apply`, `enumerate_software_pipeline_atoms`, `assign_stages`.
- `test/codegen/test_multi_buffer_unit.py` — unit tests for MultiBuffer atom.
- `test/codegen/test_software_pipeline_unit.py` — unit tests for SoftwarePipeline atom.
- `test/codegen/test_render_derivation.py` — renderer tests for `required_tiles` derivation + the two new atoms' emission.
- `test/codegen/test_multi_buffer_cpu_sim.py` — CPU-sim correctness gate on fusion + multi-buffer + pipeline stacking.

Modified files:

- `nkigym/src/nkigym/codegen/graph.py` — `Tensor` gains `buffer_degree: dict[str, int]`; `_finalize_tensors` populates default `{d: 1 for d in tensor.dim_ids}`.
- `nkigym/src/nkigym/codegen/loop_forest.py` — `LoopNode` gains `pipeline_depth: int = 1`; `_canonical_key` includes it; `hash_forest` renamed `hash_state` taking `(op_graph, forest)`; resolver helpers gain no changes.
- `nkigym/src/nkigym/codegen/render.py` — new `required_tiles` helper; `_sbuf_shape` and `_slot_expr` consult derived + stored buffer state; new `assign_stages` helper; `_emit_node` handles `pipeline_depth > 1`.
- `nkigym/src/nkigym/tune/batch.py` — `enumerate_pool` lists union four atom kinds; pool keyed by `hash_state`.
- `nkigym/src/nkigym/tune/__init__.py` — re-exports for new atoms.
- `test/codegen/test_loop_forest.py` — `hash_forest` → `hash_state` call-site updates.
- `test/codegen/test_batch.py` — `hash_forest` → `hash_state` call-site updates; extended pool assertions.

Unchanged:

- `nkigym/src/nkigym/codegen/dep_graph.py`
- `nkigym/src/nkigym/tune/fuse_loops.py`
- `nkigym/src/nkigym/tune/reorder_loops.py`
- `nkigym/src/nkigym/tune/stage.py`

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

## Task 1: Add `buffer_degree` field to `Tensor`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/graph.py` (Tensor dataclass; `_finalize_tensors`)
- Modify: `test/codegen/test_graph.py`

- [ ] **Step 1: Write the failing test**

Append to `/home/ubuntu/nki-autotune/test/codegen/test_graph.py`:

```python
def test_tensor_default_buffer_degree_is_one_per_dim() -> None:
    """Every tensor in a parsed OpGraph defaults buffer_degree[d]=1 for every d in dim_ids."""
    from test.codegen._rmsnorm_matmul_fixture import f_nkigym, INPUT_SPECS
    from nkigym.codegen.graph import parse_and_resolve

    graph = parse_and_resolve(f_nkigym, INPUT_SPECS)
    for tensor in graph.tensors.values():
        assert set(tensor.buffer_degree.keys()) == set(tensor.dim_ids), (
            f"Tensor {tensor.name!r}: buffer_degree keys {set(tensor.buffer_degree)} "
            f"!= dim_ids {set(tensor.dim_ids)}"
        )
        assert all(deg == 1 for deg in tensor.buffer_degree.values()), (
            f"Tensor {tensor.name!r}: non-default buffer_degree {tensor.buffer_degree}"
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ubuntu/nki-autotune
pytest test/codegen/test_graph.py::test_tensor_default_buffer_degree_is_one_per_dim -v
```

Expected: FAIL with `AttributeError: 'Tensor' object has no attribute 'buffer_degree'`.

- [ ] **Step 3: Add the field**

Modify `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/graph.py` at the `Tensor` dataclass (around line 31-49):

```python
@dataclass
class Tensor:
    """Named tensor appearing in the ``f_nkigym`` body.

    Attributes:
        name: Source-level variable name (e.g. ``"lhs"`` or ``"rms_inv"``).
        dim_ids: Concrete dim ids in operand order (e.g. ``("d0", "d1")``).
        shape: Element sizes aligned with ``dim_ids``.
        dtype: Element dtype (e.g. ``"bfloat16"``, ``"float32"``).
        origin: Lineage role — ``"param"`` (HBM kernel input),
            ``"intermediate"`` (SBUF handoff), or ``"return"`` (final
            op output).
        buffer_degree: Multi-buffer degree per dim. ``{d: 1 for d in
            dim_ids}`` by default. ``MultiBuffer`` atom mutates this
            field; the renderer consults it when sizing SBUF/HBM
            allocations and when building slot expressions.
    """

    name: str
    dim_ids: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    origin: TensorOrigin
    buffer_degree: dict[str, int] = field(default_factory=dict)
```

Populate it in `_finalize_tensors` (around line 524-538):

```python
def _finalize_tensors(
    scratch: dict[str, _ScratchTensor], param_names: list[str], return_name: str, raws: list[_ParsedOpRaw]
) -> dict[str, Tensor]:
    """Convert scratch tensors to read-only ``Tensor``s, tagging origin."""
    out: dict[str, Tensor] = {}
    for name, st in scratch.items():
        if name in param_names:
            origin: TensorOrigin = "param"
        elif name == return_name:
            origin = "return"
        else:
            origin = "intermediate"
        dim_ids = tuple(st.dim_ids)
        out[name] = Tensor(
            name=name,
            dim_ids=dim_ids,
            shape=tuple(st.shape),
            dtype=st.dtype,
            origin=origin,
            buffer_degree={d: 1 for d in dim_ids},
        )
    _ = raws
    return out
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/codegen/test_graph.py::test_tensor_default_buffer_degree_is_one_per_dim -v
```

Expected: PASS.

- [ ] **Step 5: Run the full codegen suite to confirm no regressions**

```bash
pytest test/codegen/ -x
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/codegen/graph.py test/codegen/test_graph.py
git commit -m "codegen: add buffer_degree field to Tensor (default 1 per dim)"
```

---

## Task 2: Add `pipeline_depth` field to `LoopNode`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/loop_forest.py`
- Modify: `test/codegen/test_loop_forest.py`

- [ ] **Step 1: Write the failing test**

Append to `/home/ubuntu/nki-autotune/test/codegen/test_loop_forest.py`:

```python
def test_loop_node_pipeline_depth_defaults_to_one() -> None:
    """Every LoopNode built by build_canonical_forest has pipeline_depth=1."""
    from test.codegen._rmsnorm_matmul_fixture import f_nkigym, INPUT_SPECS
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, build_canonical_forest

    graph = parse_and_resolve(f_nkigym, INPUT_SPECS)
    forest = build_canonical_forest(graph)

    def walk(node):
        if isinstance(node, BodyLeaf):
            return
        assert node.pipeline_depth == 1, (
            f"LoopNode(dim_id={node.dim_id!r}) has pipeline_depth={node.pipeline_depth}"
        )
        for child in node.children:
            walk(child)

    for tree in forest:
        walk(tree)


def test_canonical_key_includes_pipeline_depth() -> None:
    """Two LoopNodes differing only in pipeline_depth produce distinct canonical keys."""
    from nkigym.codegen.loop_forest import LoopNode, _canonical_key
    from nkigym.ops.base import AxisRole

    a = LoopNode(dim_id="d0", trip_count=16, role=AxisRole.PARALLEL, children=[], pipeline_depth=1)
    b = LoopNode(dim_id="d0", trip_count=16, role=AxisRole.PARALLEL, children=[], pipeline_depth=2)
    assert _canonical_key(a) != _canonical_key(b)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/codegen/test_loop_forest.py::test_loop_node_pipeline_depth_defaults_to_one -v
pytest test/codegen/test_loop_forest.py::test_canonical_key_includes_pipeline_depth -v
```

Expected: FAIL — `TypeError: LoopNode.__init__() got an unexpected keyword argument 'pipeline_depth'` for the second; the first will fail on the default-check assert too.

- [ ] **Step 3: Add the field and extend `_canonical_key`**

Modify `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/loop_forest.py` — `LoopNode` dataclass (around lines 33-66):

```python
@dataclass
class LoopNode:
    """A single loop at one tree depth.

    Attributes:
        dim_id: Concrete ``OpGraph.dims`` key this loop iterates.
        trip_count: Iteration count (``num_tiles(d)`` for a "block" tier,
            ``1`` for a "tile" tier in the 2N-per-dim canonical form;
            any divisor of ``num_tiles(d)`` under structural transforms).
        role: ``AxisRole`` for this op's use of ``dim_id``. After
            fusion the merged ``LoopNode``'s role is ``PARALLEL`` by
            construction (fusion requires both sides PARALLEL).
        children: Nested ``LoopNode``s and/or terminal ``BodyLeaf``s, in
            emission order.
        reduce_op: Reducer name for ACCUMULATION loops (``"add"``,
            ``"max"``, ...). ``None`` for PARALLEL / SEQUENTIAL loops.
            Used by :class:`ReorderLoops` to detect associative-
            compatible ACC×ACC swaps.
        name: Loop variable name emitted in the rendered ``for`` header.
            Populated by :func:`build_canonical_forest` as ``f"i_{dim_id}_{k}"``
            where ``k`` counts same-dim ancestors outermost→innermost at
            build time. Preserved verbatim across :class:`FuseLoops` and
            :class:`ReorderLoops` so loop identity survives swaps. ``None``
            on raw test forests; the renderer falls back to a
            position-based name when unset.
        pipeline_depth: Software-pipeline depth for this loop. ``1``
            means un-pipelined (current behaviour). ``> 1`` means the
            renderer emits ``depth-1`` prologue iters + skewed body +
            ``depth-1`` epilogue iters, with per-leaf iteration offsets
            derived from the subtree's op dep graph. Default ``1``.
    """

    dim_id: str
    trip_count: int
    role: AxisRole
    children: "list[LoopNode | BodyLeaf]" = field(default_factory=list)
    reduce_op: str | None = None
    name: str | None = None
    pipeline_depth: int = 1
```

Extend `_canonical_key` (around lines 334-350):

```python
def _canonical_key(node: "LoopNode | BodyLeaf") -> tuple:
    """Recursive structural key for a node.

    Two nodes (and their subtrees) produce equal keys iff they have the
    same tree shape, dim_ids, trip counts, roles, reduce_ops,
    pipeline_depths, and leaf op_idx / phase tags.
    """
    if isinstance(node, BodyLeaf):
        return ("leaf", node.op_idx, node.phase)
    return (
        "node",
        node.dim_id,
        node.trip_count,
        node.role.value,
        node.reduce_op,
        node.pipeline_depth,
        tuple(_canonical_key(c) for c in node.children),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test/codegen/test_loop_forest.py -v
```

Expected: PASS on both new tests; existing loop-forest tests still PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/loop_forest.py test/codegen/test_loop_forest.py
git commit -m "codegen: add pipeline_depth field to LoopNode (default 1)"
```

---

## Task 3: Rename `hash_forest` → `hash_state(op_graph, forest)` folding `buffer_degree`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/loop_forest.py`
- Modify: `nkigym/src/nkigym/tune/batch.py`
- Modify: `test/codegen/test_loop_forest.py`
- Modify: `test/codegen/test_batch.py`

- [ ] **Step 1: Write the failing test**

Append to `/home/ubuntu/nki-autotune/test/codegen/test_loop_forest.py`:

```python
def test_hash_state_distinguishes_buffer_degree() -> None:
    """hash_state differs when a tensor's buffer_degree changes, forest fixed."""
    from copy import deepcopy
    from test.codegen._rmsnorm_matmul_fixture import f_nkigym, INPUT_SPECS
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.codegen.loop_forest import build_canonical_forest, hash_state

    graph_a = parse_and_resolve(f_nkigym, INPUT_SPECS)
    forest = build_canonical_forest(graph_a)
    graph_b = deepcopy(graph_a)
    """Pick any intermediate tensor with a dim."""
    tname = next(t.name for t in graph_b.tensors.values() if t.origin == "intermediate" and t.dim_ids)
    some_dim = graph_b.tensors[tname].dim_ids[0]
    graph_b.tensors[tname].buffer_degree[some_dim] = 2

    assert hash_state(graph_a, forest) != hash_state(graph_b, forest)


def test_hash_state_stable_under_self_move() -> None:
    """Setting buffer_degree to its current value leaves hash_state unchanged."""
    from copy import deepcopy
    from test.codegen._rmsnorm_matmul_fixture import f_nkigym, INPUT_SPECS
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.codegen.loop_forest import build_canonical_forest, hash_state

    graph_a = parse_and_resolve(f_nkigym, INPUT_SPECS)
    forest = build_canonical_forest(graph_a)
    graph_b = deepcopy(graph_a)
    for t in graph_b.tensors.values():
        for d in t.dim_ids:
            t.buffer_degree[d] = t.buffer_degree[d]

    assert hash_state(graph_a, forest) == hash_state(graph_b, forest)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/codegen/test_loop_forest.py::test_hash_state_distinguishes_buffer_degree -v
```

Expected: FAIL with `ImportError: cannot import name 'hash_state'`.

- [ ] **Step 3: Implement `hash_state` in `loop_forest.py`**

Replace `hash_forest` in `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/loop_forest.py` (around lines 353-364) with:

```python
def hash_state(op_graph: OpGraph, forest: LoopForest) -> int:
    """Return a deterministic structural hash of the tune-stage state.

    The tune stage's state consists of the current ``op_graph`` (for
    tensor ``buffer_degree`` maps mutated by ``MultiBuffer`` atoms) and
    the current ``forest`` (mutated by ``FuseLoops``, ``ReorderLoops``,
    and ``SoftwarePipeline`` atoms). The hash folds both so identical
    states collide and distinct states (including self-moves) don't.

    Args:
        op_graph: Current ``OpGraph`` (contains the mutated tensors).
        forest: Current ``LoopForest``.

    Returns:
        Integer hash. Suitable as dict key for pool dedup.
    """
    forest_key = tuple(_canonical_key(e) for e in forest)
    tensor_key = tuple(
        (t.name, tuple(sorted(t.buffer_degree.items())))
        for t in op_graph.tensors.values()
    )
    return hash((forest_key, tensor_key))
```

Import `OpGraph` at the top of `loop_forest.py`:

```python
from nkigym.codegen.graph import OpGraph, ParsedOp
```

(Already importing `ParsedOp` — extend the line.)

- [ ] **Step 4: Update `batch.py` call sites**

Replace both `hash_forest(...)` usages in `/home/ubuntu/nki-autotune/nkigym/src/nkigym/tune/batch.py`:

```python
from nkigym.codegen.loop_forest import LoopForest, hash_state
```

In `enumerate_pool`:

```python
h0 = hash_state(op_graph, forest)
pool: dict[int, tuple[OpGraph, LoopForest]] = {h0: (op_graph, forest)}
frontier: dict[int, list[KernelRewrite]] = {h0: enumerate_fusion_atoms(op_graph, forest) + enumerate_reorder_atoms(forest)}
```

And later:

```python
h_new = hash_state(new_og, new_f)
```

- [ ] **Step 5: Update test call-sites**

Grep for stale uses:

```bash
grep -rn "hash_forest" test/ nkigym/src/
```

Replace each test's `hash_forest(forest)` call with `hash_state(op_graph, forest)`. For any test that did not previously need `op_graph` in scope, reconstruct via `parse_and_resolve`. Confirm by rerunning the suite.

- [ ] **Step 6: Run tests**

```bash
pytest test/codegen/ -x
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add nkigym/src/nkigym/codegen/loop_forest.py nkigym/src/nkigym/tune/batch.py test/codegen/
git commit -m "codegen: rename hash_forest -> hash_state; fold tensor buffer_degree"
```

---

## Task 4: Add `required_tiles` derivation helper

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py`
- Create: `test/codegen/test_render_derivation.py`

- [ ] **Step 1: Write the failing test**

Create `/home/ubuntu/nki-autotune/test/codegen/test_render_derivation.py`:

```python
"""Tests for render-time derivations (required_tiles) and new atoms' emission."""

from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.loop_forest import build_canonical_forest


def test_required_tiles_cross_loopnest_returns_num_tiles() -> None:
    """Starting state: every intermediate is cross-loopnest, so required_tiles = num_tiles."""
    from test.codegen._rmsnorm_matmul_fixture import f_nkigym, INPUT_SPECS
    from nkigym.codegen.render import required_tiles

    graph = parse_and_resolve(f_nkigym, INPUT_SPECS)
    forest = build_canonical_forest(graph)
    for tensor in graph.tensors.values():
        if tensor.origin != "intermediate":
            continue
        for d in tensor.dim_ids:
            want = graph.dims[d].num_tiles
            got = required_tiles(tensor, d, graph, forest)
            assert got == want, (
                f"Tensor {tensor.name!r} dim {d!r}: required_tiles={got}, num_tiles={want}"
            )


def test_required_tiles_intra_loopnest_returns_one() -> None:
    """After fusing AR + activation under a shared d0, sbuf_squared_sum's d0
    required_tiles drops to 1."""
    from test.codegen._rmsnorm_matmul_fixture import f_nkigym, INPUT_SPECS
    from nkigym.codegen.render import required_tiles
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode
    from nkigym.ops.base import AxisRole

    graph = parse_and_resolve(f_nkigym, INPUT_SPECS)

    """Hand-build a tiny forest with AR's reduce_close leaf and the
    activation leaf under the same d0 LoopNode. We only need these two
    ops under the shared d0; other ops can stay in their canonical
    loopnests in the forest to the side."""
    reduce_close_idx = next(
        op.idx for op in graph.ops if op.op_cls.__name__ == "NKIActivationReduce"
    )
    activation_idx = next(
        op.idx for op in graph.ops if op.op_cls.__name__ == "NKIActivation"
    )
    fused = LoopNode(
        dim_id="d0",
        trip_count=graph.dims["d0"].num_tiles,
        role=AxisRole.PARALLEL,
        children=[
            BodyLeaf(op_idx=reduce_close_idx, phase="reduce_close"),
            BodyLeaf(op_idx=activation_idx, phase="main"),
        ],
    )
    forest = [fused]

    sum_sq_name = graph.ops[reduce_close_idx].output_names[0]
    tensor = graph.tensors[sum_sq_name]
    got = required_tiles(tensor, "d0", graph, forest)
    assert got == 1, f"required_tiles(sum_sq, d0) = {got}, expected 1"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/codegen/test_render_derivation.py -v
```

Expected: FAIL — `ImportError: cannot import name 'required_tiles' from 'nkigym.codegen.render'`.

- [ ] **Step 3: Implement `required_tiles`**

Add to `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/render.py` near the top of the module (after the imports):

```python
def required_tiles(tensor: "Tensor", dim_id: str, op_graph: "OpGraph", forest: "LoopForest") -> int:
    """Return the minimum tile count along ``dim_id`` that ``tensor`` must hold.

    Derived by walking the forest: find the LCA of ``tensor``'s producer
    and all consumers, then take
    ``num_tiles(dim_id) / product_of_dim_id_trips_above_lca``. For
    cross-loopnest tensors (LCA is the forest root) the product is 1
    and ``required_tiles`` equals ``num_tiles(dim_id)``. For fully
    intra-loopnest tensors (LCA below all ``dim_id``-iterating
    ancestors) the product equals ``num_tiles(dim_id)`` and
    ``required_tiles`` is ``1``.

    Parameter and return tensors — which live in HBM and are not
    tile-decomposed — return ``op_graph.dims[dim_id].num_tiles``
    unchanged.

    Raises:
        ValueError: Product of ancestor trip counts does not divide
            ``num_tiles(dim_id)`` (unsupported forest shape).
    """
    num_t = op_graph.dims[dim_id].num_tiles
    if tensor.origin in ("param", "return"):
        return num_t
    paths = _find_access_paths(tensor.name, op_graph, forest)
    if not paths:
        return num_t
    lca = _lowest_common_ancestor(paths)
    prod = 1
    for node in lca:
        if isinstance(node, LoopNode) and node.dim_id == dim_id:
            prod *= node.trip_count
    if num_t % prod != 0:
        raise ValueError(
            f"Tensor {tensor.name!r} dim {dim_id!r}: ancestor trip product {prod} "
            f"does not divide num_tiles {num_t}"
        )
    return num_t // prod


def _find_access_paths(
    tensor_name: str, op_graph: "OpGraph", forest: "LoopForest"
) -> "list[list[LoopNode | BodyLeaf]]":
    """Return root-to-leaf node paths for every BodyLeaf that reads or writes ``tensor_name``.

    Each path is a list of ancestor nodes ending in the ``BodyLeaf``.
    The producer (unique by SSA) + all consumers are collected via
    ``op_graph.dep``.
    """
    producer_idx = op_graph.dep.producer.get(tensor_name)
    consumer_idxs = op_graph.dep.consumers.get(tensor_name, ())
    targets = set(consumer_idxs)
    if producer_idx is not None:
        targets.add(producer_idx)
    paths: list[list[LoopNode | BodyLeaf]] = []

    def walk(node: "LoopNode | BodyLeaf", stack: list["LoopNode | BodyLeaf"]) -> None:
        stack.append(node)
        if isinstance(node, BodyLeaf):
            if node.op_idx in targets:
                paths.append(list(stack))
        else:
            for child in node.children:
                walk(child, stack)
        stack.pop()

    for root in forest:
        walk(root, [])
    return paths


def _lowest_common_ancestor(
    paths: "list[list[LoopNode | BodyLeaf]]",
) -> "list[LoopNode | BodyLeaf]":
    """Return the longest common prefix of root-to-leaf paths."""
    if not paths:
        return []
    common = paths[0]
    for p in paths[1:]:
        new_len = 0
        for a, b in zip(common, p):
            if a is b:
                new_len += 1
            else:
                break
        common = common[:new_len]
    return common
```

Note: the `BodyLeaf` / `LoopNode` / `LoopForest` / `Tensor` / `OpGraph` imports at render.py's top need to cover these. Ensure the following are imported (update the existing import line):

```python
from nkigym.codegen.graph import OpGraph, Tensor
from nkigym.codegen.loop_forest import BodyLeaf, LoopForest, LoopNode, build_canonical_forest
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test/codegen/test_render_derivation.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/render.py test/codegen/test_render_derivation.py
git commit -m "codegen: add required_tiles derivation on forest+dep"
```

---

## Task 5: Wire `required_tiles * buffer_degree` through `_sbuf_shape`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py`
- Modify: `test/codegen/test_render_derivation.py`

- [ ] **Step 1: Write the failing test**

Append to `/home/ubuntu/nki-autotune/test/codegen/test_render_derivation.py`:

```python
def test_sbuf_allocation_shrinks_after_fusion_of_ar_and_activation() -> None:
    """After fusing AR + activation under shared d0, sbuf_squared_sum allocates (128, 1, 1)."""
    from test.codegen._rmsnorm_matmul_fixture import f_nkigym, INPUT_SPECS
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, build_canonical_forest
    from nkigym.codegen.render import render
    from nkigym.ops.base import AxisRole

    graph = parse_and_resolve(f_nkigym, INPUT_SPECS)
    canonical = build_canonical_forest(graph)

    """Replace the AR tree and the activation tree in the canonical
    forest with a single fused d0 LoopNode that contains AR's
    reduce_close leaf + activation's main leaf directly. This is a
    synthetic forest for testing the derivation only — a proper fusion
    via FuseLoops atoms lands in Task 10."""
    ar_idx = next(i for i, op in enumerate(graph.ops) if op.op_cls.__name__ == "NKIActivationReduce")
    act_idx = next(i for i, op in enumerate(graph.ops) if op.op_cls.__name__ == "NKIActivation")
    fused = LoopNode(
        dim_id="d0",
        trip_count=graph.dims["d0"].num_tiles,
        role=AxisRole.PARALLEL,
        children=[
            BodyLeaf(op_idx=graph.ops[ar_idx].idx, phase="reduce_close"),
            BodyLeaf(op_idx=graph.ops[act_idx].idx, phase="main"),
        ],
        name="i_d0_0_fused",
    )
    """Splice: keep other roots, drop the AR and activation trees, add the fused node."""
    new_forest = []
    for idx, root in enumerate(canonical):
        if idx in (ar_idx, act_idx):
            continue
        new_forest.append(root)
    new_forest.insert(ar_idx, fused)

    sum_sq_name = graph.ops[ar_idx].output_names[0]
    src = render(graph, new_forest)
    expected_alloc = f"sbuf_{sum_sq_name} = nl.ndarray((128, 1, 1), dtype=nl.float32, buffer=nl.sbuf)"
    assert expected_alloc in src, f"Did not find expected allocation; source:\n{src}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/codegen/test_render_derivation.py::test_sbuf_allocation_shrinks_after_fusion_of_ar_and_activation -v
```

Expected: FAIL — allocation still `(128, 16, 1)`.

- [ ] **Step 3: Update `_sbuf_shape`**

Modify `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/render.py` `_sbuf_shape` (around lines 131-144):

```python
def _sbuf_shape(
    tensor: "Tensor", op_graph: "OpGraph", forest: "LoopForest"
) -> tuple[int, int, int]:
    """Compute 3D SBUF shape ``(p_tile, total_slots_P, num_f_tiles * f_tile)``.

    ``total_slots_P = required_tiles(P) * buffer_degree[P]``. Free axis
    still spans the full tile count for now — free-axis multi-buffer is
    out of scope.

    1D tensors collapse the free axis to a single element.
    """
    if not tensor.dim_ids:
        raise ValueError(f"Tensor {tensor.name!r} has no dims")
    p_axis = tensor.dim_ids[0]
    p_info = op_graph.dims[p_axis]
    p_required = required_tiles(tensor, p_axis, op_graph, forest)
    p_total = p_required * tensor.buffer_degree[p_axis]
    if len(tensor.dim_ids) == 1:
        return (p_info.tile_size, p_total, 1)
    f_axis = tensor.dim_ids[1]
    f_info = op_graph.dims[f_axis]
    return (p_info.tile_size, p_total, f_info.num_tiles * f_info.tile_size)
```

Update the call-site in `_emit_sbuf_allocations` (around line 122) to pass the forest — which means `_emit_sbuf_allocations` also needs the forest. Threaded through `render`:

```python
def _emit_sbuf_allocations(w: _Writer, op_graph: OpGraph, forest: LoopForest) -> None:
    """Allocate one SBUF buffer per intermediate, then per op-local buffer.

    Kernel inputs live in HBM (consumed by ``NKILoad``) and the return
    tensor lives in HBM (written by ``NKIStore``). The store emitter
    reads from its data-operand's SBUF buffer directly, so the return
    has no SBUF mirror and is skipped here. Op-local buffers come after
    cross-nest tensors, in op-index order, sized at single-iteration
    scope (``(tile_size, 1, free_extent)``).
    """
    for name, tensor in op_graph.tensors.items():
        if tensor.origin in ("param", "return"):
            continue
        shape = _sbuf_shape(tensor, op_graph, forest)
        w.line(f"{_sbuf_name(name)} = nl.ndarray({shape}, dtype=nl.{tensor.dtype}, buffer=nl.sbuf)")
    for op in op_graph.ops:
        for buf in op.op_local_buffers.values():
            nl_buffer = "nl.sbuf" if buf.location == "sbuf" else "nl.psum"
            w.line(f"{buf.emitted_name} = nl.ndarray({buf.shape}, dtype=nl.{buf.dtype}, buffer={nl_buffer})")
    w.line()
```

And thread it through `render` (around line 58-77):

```python
def render(op_graph: OpGraph, forest: LoopForest | None = None) -> str:
    """Render ``op_graph`` to NKI kernel source via the forest walker.

    When ``forest`` is ``None``, a canonical forest is built from
    ``op_graph`` — matches today's default behaviour. Callers with a
    transformed forest (e.g. after fusion rewrites) pass it explicitly.
    """
    if forest is None:
        forest = build_canonical_forest(op_graph)
    w = _Writer()
    _emit_imports(w)
    _emit_signature(w, op_graph)
    w.indent()
    _emit_param_asserts(w, op_graph)
    _emit_hbm_output(w, op_graph)
    _emit_sbuf_allocations(w, op_graph, forest)
    render_forest(w, op_graph, forest)
    w.line(f"return {_hbm_name(op_graph.return_name)}")
    w.dedent()
    return w.getvalue()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test/codegen/test_render_derivation.py -v
pytest test/codegen/test_render.py -v
```

Expected: PASS. `test_render.py` golden snapshots may need updates — check if snapshots shrink intermediates correctly; update any stale golden strings.

- [ ] **Step 5: Run the full codegen suite**

```bash
pytest test/codegen/ -x
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/codegen/render.py test/codegen/
git commit -m "codegen: use required_tiles*buffer_degree in _sbuf_shape (fixes cross-nest bloat)"
```

---

## Task 6: Wire `total_slots` through `_slot_expr`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py`
- Modify: `test/codegen/test_render_derivation.py`

- [ ] **Step 1: Write the failing test**

Append to `/home/ubuntu/nki-autotune/test/codegen/test_render_derivation.py`:

```python
def test_slot_expression_degenerates_to_zero_when_total_slots_equals_one() -> None:
    """Intra-loopnest tensor with buffer_degree=1: slot expression degrades to '0'."""
    from test.codegen._rmsnorm_matmul_fixture import f_nkigym, INPUT_SPECS
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode
    from nkigym.codegen.render import render
    from nkigym.ops.base import AxisRole

    graph = parse_and_resolve(f_nkigym, INPUT_SPECS)

    ar_idx = next(i for i, op in enumerate(graph.ops) if op.op_cls.__name__ == "NKIActivationReduce")
    act_idx = next(i for i, op in enumerate(graph.ops) if op.op_cls.__name__ == "NKIActivation")
    fused = LoopNode(
        dim_id="d0",
        trip_count=graph.dims["d0"].num_tiles,
        role=AxisRole.PARALLEL,
        children=[
            BodyLeaf(op_idx=graph.ops[ar_idx].idx, phase="reduce_close"),
            BodyLeaf(op_idx=graph.ops[act_idx].idx, phase="main"),
        ],
        name="i_d0_0_fused",
    )

    from nkigym.codegen.loop_forest import build_canonical_forest
    canonical = build_canonical_forest(graph)
    new_forest = [r for i, r in enumerate(canonical) if i not in (ar_idx, act_idx)]
    new_forest.insert(ar_idx, fused)

    src = render(graph, new_forest)
    """sum_sq slot should index with a constant 0 at the P position,
    no modulo clutter."""
    assert "sbuf_squared_sum[0:128, 0, 0:1]" in src, (
        f"Expected collapsed slot 'sbuf_squared_sum[0:128, 0, 0:1]' in source; got:\n{src}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/codegen/test_render_derivation.py::test_slot_expression_degenerates_to_zero_when_total_slots_equals_one -v
```

Expected: FAIL — the slot still emits `i_d0_0_fused + ...` ordinals.

- [ ] **Step 3: Update `_slot_expr` + call sites**

Modify `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/render.py` `_slot_expr` (around lines 147-178):

```python
def _slot_expr(
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
    dim_id: str,
    total_slots: int,
    stage_offset: int = 0,
) -> str:
    """Return the slot expression for ``dim_id``.

    For dim ``d`` with ``k`` same-dim ancestors on the current path and
    loop variable names ``path_names[d] = [n_0, n_1, ..., n_{k-1}]``
    (outermost→innermost), the raw slot is ``sum_{idx} n_idx *
    prod_of_tail_trips``. The final expression is
    ``(raw_slot) % total_slots``, with two simplifications:

    * ``total_slots == 1`` — slot is literal ``"0"``.
    * ``total_slots == product_of_ancestor_trips`` (i.e. the raw slot
      never exceeds ``total_slots``) — modulo is identity; emit the raw
      slot.

    ``stage_offset`` adds an integer offset to the innermost ancestor
    (used by the software-pipelined body emission). Default 0.

    Raises:
        ValueError: ``dim_id`` has no open ancestor loops on the path.
    """
    names = path_names.get(dim_id, [])
    k = len(names)
    if k == 0:
        raise ValueError(f"No open LoopNode on path for dim {dim_id!r}")
    if total_slots == 1:
        return "0"
    trips = path_trips[dim_id]
    raw_trip_product = 1
    for t in trips:
        raw_trip_product *= t
    terms: list[str] = []
    for idx in range(k):
        tail_prod = 1
        for t in trips[idx + 1 :]:
            tail_prod *= t
        innermost = idx == k - 1
        if innermost and stage_offset != 0:
            sign = "+" if stage_offset > 0 else "-"
            token = f"({names[idx]} {sign} {abs(stage_offset)})"
        else:
            token = names[idx]
        if tail_prod == 1:
            terms.append(token)
        else:
            terms.append(f"{token} * {tail_prod}")
    raw = " + ".join(terms)
    if total_slots == raw_trip_product and stage_offset == 0:
        return raw
    return f"({raw}) % {total_slots}"
```

Update every `_slot_expr(...)` call-site to pass `total_slots`. The convention: callers compute `total_slots` as
`required_tiles(tensor, dim_id, op_graph, forest) * tensor.buffer_degree[dim_id]` for tensors, and
`op_graph.dims[dim_id].num_tiles` (raw) for non-tensor contexts — but in practice all call-sites reference a tensor. Update:

- `_sbuf_tile_slice` signature: add `total_slots_p: int`, `total_slots_f: int` parameters; pass down to `_slot_expr`.
- `_hbm_tile_slice`: similar; HBM slices use raw dim trip products (no multi-buffer) — pass `dims[d].num_tiles`.
- `_swapped_dst_tile_slice`: pass `total_slots` for both dst axes.

Every body emitter (`_body_load`, `_body_store`, `_body_activation`, `_body_tensor_scalar`, `_body_transpose`, `_body_dma_transpose`, `_body_matmul_*`, `_body_ar_*`) needs to compute `total_slots` for each tensor slice it emits and thread it through. Helper:

```python
def _tensor_total_slots(
    tensor: Tensor, dim_id: str, op_graph: OpGraph, forest: LoopForest
) -> int:
    """Per-dim total slot count for a tensor: required_tiles * buffer_degree."""
    return required_tiles(tensor, dim_id, op_graph, forest) * tensor.buffer_degree[dim_id]
```

Then e.g. `_body_load`:

```python
@_register_body("NKILoad", "main")
def _body_load(w, op_graph, op, path_names, path_trips, forest) -> None:
    """Emit one ``nisa.dma_copy`` at the innermost open-loop point."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src_tensor = op_graph.tensors[src_name]
    dst_tensor = op_graph.tensors[dst_name]
    p_axis = src_tensor.dim_ids[0]
    f_axis = src_tensor.dim_ids[1] if len(src_tensor.dim_ids) > 1 else None
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size if f_axis is not None else 1
    dst_slots_p = _tensor_total_slots(dst_tensor, dst_tensor.dim_ids[0], op_graph, forest)
    dst_slots_f = (
        _tensor_total_slots(dst_tensor, dst_tensor.dim_ids[1], op_graph, forest)
        if len(dst_tensor.dim_ids) > 1 else 1
    )
    dst_expr = _sbuf_tile_slice(
        _sbuf_name(dst_name), dst_tensor.dim_ids, p_tile, f_tile,
        path_names, path_trips, dst_slots_p, dst_slots_f,
    )
    src_expr = _hbm_tile_slice(
        src_name, src_tensor.dim_ids, p_tile, f_tile, path_names, path_trips,
        op_graph.dims[p_axis].num_tiles,
        op_graph.dims[f_axis].num_tiles if f_axis is not None else 1,
    )
    w.line(f"nisa.dma_copy(dst={dst_expr}, src={src_expr})")
```

All other body emitters take the same shape-change. Forest is threaded via a new param; update `_emit_node` / `render_forest` signatures accordingly.

**Stage offset is not used yet in this task** — defaults to `0` at all call-sites. Task 8 exercises it.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test/codegen/test_render_derivation.py -v
pytest test/codegen/test_render.py -v
pytest test/codegen/ -x
```

Expected: PASS on all new tests. Existing tests may reveal stale slot-expression snapshots — update them.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/render.py test/codegen/
git commit -m "codegen: thread total_slots through _slot_expr + body emitters"
```

---

## Task 7: Scaffold `MultiBuffer` atom (is_legal + apply + enumerate)

**Files:**
- Create: `nkigym/src/nkigym/tune/multi_buffer.py`
- Create: `test/codegen/test_multi_buffer_unit.py`
- Modify: `nkigym/src/nkigym/tune/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `/home/ubuntu/nki-autotune/test/codegen/test_multi_buffer_unit.py`:

```python
"""Unit tests for MultiBuffer atom mechanics."""

from copy import deepcopy

from test.codegen._rmsnorm_matmul_fixture import f_nkigym, INPUT_SPECS
from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, build_canonical_forest
from nkigym.ops.base import AxisRole
from nkigym.tune.multi_buffer import MultiBuffer, enumerate_multi_buffer_atoms


def _fresh_state():
    graph = parse_and_resolve(f_nkigym, INPUT_SPECS)
    forest = build_canonical_forest(graph)
    return graph, forest


def _fused_state():
    """Return a state where AR's reduce_close + activation's main share a d0 LoopNode."""
    graph = parse_and_resolve(f_nkigym, INPUT_SPECS)
    ar_idx = next(i for i, op in enumerate(graph.ops) if op.op_cls.__name__ == "NKIActivationReduce")
    act_idx = next(i for i, op in enumerate(graph.ops) if op.op_cls.__name__ == "NKIActivation")
    fused = LoopNode(
        dim_id="d0",
        trip_count=graph.dims["d0"].num_tiles,
        role=AxisRole.PARALLEL,
        children=[
            BodyLeaf(op_idx=graph.ops[ar_idx].idx, phase="reduce_close"),
            BodyLeaf(op_idx=graph.ops[act_idx].idx, phase="main"),
        ],
        name="i_d0_0_fused",
    )
    canonical = build_canonical_forest(graph)
    new_forest = [r for i, r in enumerate(canonical) if i not in (ar_idx, act_idx)]
    new_forest.insert(ar_idx, fused)
    return graph, new_forest, graph.ops[ar_idx].output_names[0]


def test_multi_buffer_illegal_on_missing_tensor() -> None:
    graph, forest = _fresh_state()
    atom = MultiBuffer(tensor_name="does_not_exist", dim_id="d0", degree=1)
    assert atom.is_legal(graph, forest) is False


def test_multi_buffer_illegal_on_missing_dim() -> None:
    graph, forest = _fresh_state()
    tname = next(t.name for t in graph.tensors.values() if t.origin == "intermediate")
    atom = MultiBuffer(tensor_name=tname, dim_id="no_such_dim", degree=1)
    assert atom.is_legal(graph, forest) is False


def test_multi_buffer_illegal_on_nonpositive_degree() -> None:
    graph, forest = _fresh_state()
    tname = next(t.name for t in graph.tensors.values() if t.origin == "intermediate")
    d = graph.tensors[tname].dim_ids[0]
    assert MultiBuffer(tensor_name=tname, dim_id=d, degree=0).is_legal(graph, forest) is False


def test_multi_buffer_illegal_when_degree_exceeds_lca_trip_product() -> None:
    """Cross-loopnest tensor has lca_trip_product=1; degree>1 is illegal."""
    graph, forest = _fresh_state()
    tname = next(t.name for t in graph.tensors.values() if t.origin == "intermediate")
    d = graph.tensors[tname].dim_ids[0]
    assert MultiBuffer(tensor_name=tname, dim_id=d, degree=2).is_legal(graph, forest) is False


def test_multi_buffer_legal_on_intra_loopnest_tensor() -> None:
    graph, forest, sum_sq = _fused_state()
    """lca_trip_product(d0) = 16, so degree in {1, 2, 4, 8, 16} is legal."""
    for degree in (1, 2, 4, 8, 16):
        atom = MultiBuffer(tensor_name=sum_sq, dim_id="d0", degree=degree)
        assert atom.is_legal(graph, forest), f"degree {degree} should be legal"
    """Non-divisors are illegal."""
    for degree in (3, 5, 7, 32):
        atom = MultiBuffer(tensor_name=sum_sq, dim_id="d0", degree=degree)
        assert not atom.is_legal(graph, forest), f"degree {degree} should be illegal"


def test_multi_buffer_apply_updates_only_targeted_tensor() -> None:
    graph, forest, sum_sq = _fused_state()
    before = deepcopy(graph)
    atom = MultiBuffer(tensor_name=sum_sq, dim_id="d0", degree=2)
    new_graph, new_forest = atom.apply(graph, forest)
    assert new_graph.tensors[sum_sq].buffer_degree["d0"] == 2
    for name, tensor in new_graph.tensors.items():
        if name == sum_sq:
            continue
        assert tensor.buffer_degree == before.tensors[name].buffer_degree
    assert new_forest is forest


def test_enumerate_multi_buffer_atoms_skips_cross_loopnest() -> None:
    """In the starting canonical forest every intermediate is cross-loopnest on every dim →
    no useful atoms emitted."""
    graph, forest = _fresh_state()
    atoms = enumerate_multi_buffer_atoms(graph, forest)
    assert atoms == [], f"Expected empty list; got {atoms}"


def test_enumerate_multi_buffer_atoms_finds_intra_loopnest() -> None:
    graph, forest, sum_sq = _fused_state()
    atoms = enumerate_multi_buffer_atoms(graph, forest)
    """Every divisor of 16 except 1 (current degree) for sum_sq on d0."""
    d0_atoms = [a for a in atoms if a.tensor_name == sum_sq and a.dim_id == "d0"]
    degrees = sorted(a.degree for a in d0_atoms)
    assert degrees == [2, 4, 8, 16], f"got degrees {degrees}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/codegen/test_multi_buffer_unit.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'nkigym.tune.multi_buffer'`.

- [ ] **Step 3: Implement `MultiBuffer`**

Create `/home/ubuntu/nki-autotune/nkigym/src/nkigym/tune/multi_buffer.py`:

```python
"""``MultiBuffer`` rewrite — set a tensor's per-dim buffer_degree.

Adjusts ``op_graph.tensors[tensor_name].buffer_degree[dim_id]`` to any
divisor of the tensor's current ``lca_trip_product(dim_id)`` in the
active forest. Forest is not modified.

Cross-loopnest tensors (``lca_trip_product == 1``) accept only
``degree == 1``; the enumerator filters them out so the sampler doesn't
waste atoms on no-op self-moves.
"""

from copy import deepcopy
from dataclasses import dataclass

from nkigym.codegen.graph import OpGraph
from nkigym.codegen.loop_forest import BodyLeaf, LoopForest, LoopNode
from nkigym.codegen.render import _find_access_paths, _lowest_common_ancestor


@dataclass(frozen=True)
class MultiBuffer:
    """Set a tensor's ``buffer_degree[dim_id]`` to ``degree``.

    Attributes:
        tensor_name: Name of the tensor to adjust.
        dim_id: Concrete dim the degree applies to.
        degree: New degree. Must be a positive divisor of
            ``lca_trip_product(tensor, dim_id, forest)`` AND differ from
            the current stored value (enforced by :meth:`is_legal` via
            dedup in the sampler).
    """

    tensor_name: str
    dim_id: str
    degree: int

    def is_legal(self, op_graph: OpGraph, forest: LoopForest) -> bool:
        """Return True when the atom's parameters identify a valid rewrite."""
        if self.tensor_name not in op_graph.tensors:
            return False
        tensor = op_graph.tensors[self.tensor_name]
        if self.dim_id not in tensor.dim_ids:
            return False
        if self.degree < 1:
            return False
        prod = _lca_trip_product(self.tensor_name, self.dim_id, op_graph, forest)
        if self.degree > prod:
            return False
        if prod % self.degree != 0:
            return False
        return True

    def apply(self, op_graph: OpGraph, forest: LoopForest) -> tuple[OpGraph, LoopForest]:
        """Return a new ``(op_graph, forest)`` with only the targeted tensor updated.

        Uses ``deepcopy`` on the tensor and shallow re-assembles the
        graph. Other tensors share by reference.
        """
        new_tensors = dict(op_graph.tensors)
        new_tensor = deepcopy(op_graph.tensors[self.tensor_name])
        new_tensor.buffer_degree[self.dim_id] = self.degree
        new_tensors[self.tensor_name] = new_tensor
        new_graph = OpGraph(
            func_name=op_graph.func_name,
            param_names=op_graph.param_names,
            return_name=op_graph.return_name,
            tensors=new_tensors,
            dims=op_graph.dims,
            ops=op_graph.ops,
            per_op_attrs=op_graph.per_op_attrs,
            dep=op_graph.dep,
        )
        return new_graph, forest


def enumerate_multi_buffer_atoms(op_graph: OpGraph, forest: LoopForest) -> list[MultiBuffer]:
    """Return every non-self-move :class:`MultiBuffer` atom legal for the current state.

    For each intermediate tensor and each of its dims, emit atoms for
    every divisor of ``lca_trip_product`` except the current degree.
    Cross-loopnest tensors yield nothing (``lca_trip_product = 1``,
    degree pinned at 1).
    """
    atoms: list[MultiBuffer] = []
    for tensor in op_graph.tensors.values():
        if tensor.origin != "intermediate":
            continue
        for d in tensor.dim_ids:
            prod = _lca_trip_product(tensor.name, d, op_graph, forest)
            if prod == 1:
                continue
            current = tensor.buffer_degree[d]
            for degree in _divisors(prod):
                if degree == current:
                    continue
                atoms.append(MultiBuffer(tensor_name=tensor.name, dim_id=d, degree=degree))
    return atoms


def _divisors(n: int) -> list[int]:
    """Return every positive divisor of ``n`` in ascending order."""
    out: list[int] = []
    d = 1
    while d * d <= n:
        if n % d == 0:
            out.append(d)
            if d != n // d:
                out.append(n // d)
        d += 1
    out.sort()
    return out


def _lca_trip_product(tensor_name: str, dim_id: str, op_graph: OpGraph, forest: LoopForest) -> int:
    """Product of ``LoopNode.trip_count`` over all ``dim_id``-iterating ancestors
    above the LCA of ``tensor_name``'s producer + all consumers. 1 when no such
    ancestors exist (tensor is cross-loopnest on ``dim_id``).
    """
    paths = _find_access_paths(tensor_name, op_graph, forest)
    if not paths:
        return 1
    lca = _lowest_common_ancestor(paths)
    prod = 1
    for node in lca:
        if isinstance(node, LoopNode) and node.dim_id == dim_id:
            prod *= node.trip_count
    return prod
```

Export from `/home/ubuntu/nki-autotune/nkigym/src/nkigym/tune/__init__.py`:

Add `MultiBuffer` to the module's exports.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test/codegen/test_multi_buffer_unit.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/tune/multi_buffer.py nkigym/src/nkigym/tune/__init__.py test/codegen/test_multi_buffer_unit.py
git commit -m "tune: add MultiBuffer atom (is_legal + apply + enumerate)"
```

---

## Task 8: Implement pipelined rendering for `LoopNode.pipeline_depth > 1`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py`
- Modify: `test/codegen/test_render_derivation.py`

- [ ] **Step 1: Write the failing test**

Append to `/home/ubuntu/nki-autotune/test/codegen/test_render_derivation.py`:

```python
def test_assign_stages_linear_chain() -> None:
    """Linear chain A->B->C under one LoopNode gives stages {A:0, B:1, C:2}."""
    from test.codegen._rmsnorm_matmul_fixture import f_nkigym, INPUT_SPECS
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode
    from nkigym.codegen.render import assign_stages
    from nkigym.ops.base import AxisRole

    graph = parse_and_resolve(f_nkigym, INPUT_SPECS)
    ar_idx = next(i for i, op in enumerate(graph.ops) if op.op_cls.__name__ == "NKIActivationReduce")
    act_idx = next(i for i, op in enumerate(graph.ops) if op.op_cls.__name__ == "NKIActivation")
    ts_idx = next(i for i, op in enumerate(graph.ops) if op.op_cls.__name__ == "NKITensorScalar")
    chain = LoopNode(
        dim_id="d0",
        trip_count=16,
        role=AxisRole.PARALLEL,
        children=[
            BodyLeaf(op_idx=graph.ops[ar_idx].idx, phase="reduce_close"),
            BodyLeaf(op_idx=graph.ops[act_idx].idx, phase="main"),
            BodyLeaf(op_idx=graph.ops[ts_idx].idx, phase="main"),
        ],
    )
    stages = assign_stages(chain, graph.dep)
    assert stages[(graph.ops[ar_idx].idx, "reduce_close")] == 0
    assert stages[(graph.ops[act_idx].idx, "main")] == 1
    assert stages[(graph.ops[ts_idx].idx, "main")] == 2


def test_render_pipelined_loop_depth_two_emits_prologue_body_epilogue() -> None:
    """A d0 loop with pipeline_depth=2 and a legal buffer_degree emits 3 phases."""
    from test.codegen._rmsnorm_matmul_fixture import f_nkigym, INPUT_SPECS
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, build_canonical_forest
    from nkigym.codegen.render import render
    from nkigym.ops.base import AxisRole

    graph = parse_and_resolve(f_nkigym, INPUT_SPECS)
    ar_idx = next(i for i, op in enumerate(graph.ops) if op.op_cls.__name__ == "NKIActivationReduce")
    act_idx = next(i for i, op in enumerate(graph.ops) if op.op_cls.__name__ == "NKIActivation")
    fused = LoopNode(
        dim_id="d0",
        trip_count=graph.dims["d0"].num_tiles,
        role=AxisRole.PARALLEL,
        children=[
            BodyLeaf(op_idx=graph.ops[ar_idx].idx, phase="reduce_close"),
            BodyLeaf(op_idx=graph.ops[act_idx].idx, phase="main"),
        ],
        name="i_d0_0_fused",
        pipeline_depth=2,
    )

    canonical = build_canonical_forest(graph)
    new_forest = [r for i, r in enumerate(canonical) if i not in (ar_idx, act_idx)]
    new_forest.insert(ar_idx, fused)

    """Before rendering we need enough buffer_degree for sum_sq to
    satisfy the skew legality (1 < required_tiles * buffer_degree).
    required_tiles is 1 inside the fused loop; buffer_degree must be >=
    2."""
    sum_sq_name = graph.ops[ar_idx].output_names[0]
    graph.tensors[sum_sq_name].buffer_degree["d0"] = 2

    src = render(graph, new_forest)
    """Prologue fires AR once at iter 0; body iterates 1..15 with AR at
    i and activation at i-1; epilogue fires activation at 15."""
    assert f"for i_d0_0_fused in range(1, {graph.dims['d0'].num_tiles}):" in src, src
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/codegen/test_render_derivation.py::test_assign_stages_linear_chain -v
pytest test/codegen/test_render_derivation.py::test_render_pipelined_loop_depth_two_emits_prologue_body_epilogue -v
```

Expected: FAIL — `assign_stages` not defined; pipelined rendering not implemented.

- [ ] **Step 3: Implement `assign_stages` + pipelined emission**

Add `assign_stages` to `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/render.py`:

```python
def assign_stages(
    loop_node: "LoopNode", dep: "DepGraph"
) -> dict[tuple[int, str], int]:
    """Return a stage index per (op_idx, phase) leaf in ``loop_node``'s subtree.

    Walks the subtree in source order; each leaf's stage is one more
    than the max stage of any upstream leaf whose writes it reads.
    Leaves that read nothing produced in the subtree get stage 0.

    Args:
        loop_node: The LoopNode whose subtree's stages are assigned.
        dep: Op-level dep graph.

    Returns:
        Dict keyed by (op_idx, phase).
    """
    leaves = _collect_body_leaves(loop_node)
    stage: dict[tuple[int, str], int] = {}
    writes_in_subtree: dict[str, tuple[int, str]] = {}
    for leaf in leaves:
        reads = dep.reads.get(leaf.op_idx, frozenset())
        upstream = [stage[writes_in_subtree[t]] for t in reads if t in writes_in_subtree]
        stage[(leaf.op_idx, leaf.phase)] = (max(upstream) + 1) if upstream else 0
        for t in dep.writes.get(leaf.op_idx, frozenset()):
            writes_in_subtree[t] = (leaf.op_idx, leaf.phase)
    return stage


def _collect_body_leaves(node: "LoopNode | BodyLeaf") -> list["BodyLeaf"]:
    """Gather every BodyLeaf under ``node`` in tree (DFS) order."""
    acc: list[BodyLeaf] = []
    def walk(n):
        if isinstance(n, BodyLeaf):
            acc.append(n)
        else:
            for c in n.children:
                walk(c)
    walk(node)
    return acc
```

Import `DepGraph` at module top:

```python
from nkigym.codegen.dep_graph import DepGraph
```

Now update `_emit_node` to handle `pipeline_depth > 1`. Existing `_emit_node` for `LoopNode` (around lines 292-323):

```python
def _emit_node(
    w: _Writer,
    op_graph: OpGraph,
    node: LoopNode | BodyLeaf,
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
    forest: LoopForest,
) -> None:
    """Emit one forest node. LoopNode may pipeline; BodyLeaf delegates to registry."""
    if isinstance(node, BodyLeaf):
        op = op_graph.ops[node.op_idx]
        emitter = _BODY_EMITTERS.get((op.op_cls.__name__, node.phase))
        if emitter is None:
            raise ValueError(f"No body emitter registered for ({op.op_cls.__name__!r}, {node.phase!r})")
        emitter(w, op_graph, op, path_names, path_trips, forest, stage_offset=0)
        return
    if node.pipeline_depth <= 1:
        _emit_vanilla_loop(w, op_graph, node, path_names, path_trips, forest)
    else:
        _emit_pipelined_loop(w, op_graph, node, path_names, path_trips, forest)


def _emit_vanilla_loop(w, op_graph, node, path_names, path_trips, forest) -> None:
    existing = path_names.setdefault(node.dim_id, [])
    loop_var = node.name if node.name is not None else f"i_{node.dim_id}_{len(existing)}"
    w.line(f"for {loop_var} in range({node.trip_count}):")
    w.indent()
    existing.append(loop_var)
    path_trips.setdefault(node.dim_id, []).append(node.trip_count)
    for child in node.children:
        _emit_node(w, op_graph, child, path_names, path_trips, forest)
    path_trips[node.dim_id].pop()
    existing.pop()
    w.dedent()


def _emit_pipelined_loop(w, op_graph, node, path_names, path_trips, forest) -> None:
    """Emit prologue (depth-1 unrolled iters) + pipelined body + epilogue."""
    stages = assign_stages(node, op_graph.dep)
    max_stage = max(stages.values())
    D = node.pipeline_depth
    N = node.trip_count
    if D > max_stage + 1:
        raise ValueError(
            f"pipeline_depth {D} exceeds chain length {max_stage + 1} in subtree"
        )

    existing = path_names.setdefault(node.dim_id, [])
    loop_var = node.name if node.name is not None else f"i_{node.dim_id}_{len(existing)}"
    path_trips.setdefault(node.dim_id, []).append(N)

    """Prologue: D-1 iters."""
    for i_pro in range(D - 1):
        existing.append(str(i_pro))
        for child in node.children:
            _emit_pipelined_leaf(w, op_graph, child, path_names, path_trips, forest, stages, i_pro, fire_if_stage_le=i_pro)
        existing.pop()

    """Pipelined body."""
    w.line(f"for {loop_var} in range({D - 1}, {N}):")
    w.indent()
    existing.append(loop_var)
    for child in node.children:
        _emit_pipelined_leaf(
            w, op_graph, child, path_names, path_trips, forest, stages, None, fire_if_stage_le=None
        )
    existing.pop()
    w.dedent()

    """Epilogue: D-1 iters."""
    for i_epi in range(D - 1):
        absolute = N - (D - 1) + i_epi
        existing.append(str(absolute))
        for child in node.children:
            _emit_pipelined_leaf(
                w, op_graph, child, path_names, path_trips, forest, stages, absolute, fire_if_stage_gt=i_epi
            )
        existing.pop()

    path_trips[node.dim_id].pop()


def _emit_pipelined_leaf(
    w, op_graph, child, path_names, path_trips, forest, stages,
    constant_loop_var, fire_if_stage_le=None, fire_if_stage_gt=None,
) -> None:
    """Emit one leaf inside the pipelined loop. When ``constant_loop_var`` is not
    None, the innermost ancestor is replaced by that literal (prologue/epilogue);
    otherwise the body emits with ``stage_offset=-stage`` to shift the loop var."""
    if not isinstance(child, BodyLeaf):
        """Nested LoopNodes under a pipelined loop not supported in this iteration."""
        raise ValueError("pipelined loop children must be BodyLeafs")
    stage = stages[(child.op_idx, child.phase)]
    if fire_if_stage_le is not None and stage > fire_if_stage_le:
        return
    if fire_if_stage_gt is not None and stage <= fire_if_stage_gt:
        return
    op = op_graph.ops[child.op_idx]
    emitter = _BODY_EMITTERS[(op.op_cls.__name__, child.phase)]
    offset = -stage
    emitter(w, op_graph, op, path_names, path_trips, forest, stage_offset=offset)
```

Every `_register_body`-decorated body emitter needs the extra `stage_offset` kwarg threaded to `_slot_expr`. Shortest path: add `stage_offset: int = 0` to every emitter's signature, pass it into every `_sbuf_tile_slice` / `_hbm_tile_slice` / `_slot_expr` call that slices a tensor along the *innermost pipelined dim*. Implement this as a single argument that cascades through the slicing helpers.

For this task, it's sufficient to thread `stage_offset` through the slice helpers as the new innermost-only delta used in `_slot_expr`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test/codegen/test_render_derivation.py -v
pytest test/codegen/ -x
```

Expected: PASS on new tests; existing tests still PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/render.py test/codegen/test_render_derivation.py
git commit -m "codegen: emit prologue+body+epilogue for LoopNode.pipeline_depth>1"
```

---

## Task 9: Scaffold `SoftwarePipeline` atom

**Files:**
- Create: `nkigym/src/nkigym/tune/software_pipeline.py`
- Create: `test/codegen/test_software_pipeline_unit.py`
- Modify: `nkigym/src/nkigym/tune/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `/home/ubuntu/nki-autotune/test/codegen/test_software_pipeline_unit.py`:

```python
"""Unit tests for SoftwarePipeline atom mechanics."""

from test.codegen._rmsnorm_matmul_fixture import f_nkigym, INPUT_SPECS
from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, build_canonical_forest
from nkigym.ops.base import AxisRole
from nkigym.tune.software_pipeline import SoftwarePipeline, enumerate_software_pipeline_atoms


def _fused_state():
    graph = parse_and_resolve(f_nkigym, INPUT_SPECS)
    ar_idx = next(i for i, op in enumerate(graph.ops) if op.op_cls.__name__ == "NKIActivationReduce")
    act_idx = next(i for i, op in enumerate(graph.ops) if op.op_cls.__name__ == "NKIActivation")
    fused = LoopNode(
        dim_id="d0",
        trip_count=graph.dims["d0"].num_tiles,
        role=AxisRole.PARALLEL,
        children=[
            BodyLeaf(op_idx=graph.ops[ar_idx].idx, phase="reduce_close"),
            BodyLeaf(op_idx=graph.ops[act_idx].idx, phase="main"),
        ],
        name="i_d0_0_fused",
    )
    canonical = build_canonical_forest(graph)
    new_forest = [r for i, r in enumerate(canonical) if i not in (ar_idx, act_idx)]
    new_forest.insert(ar_idx, fused)
    return graph, new_forest, ar_idx, graph.ops[ar_idx].output_names[0]


def test_software_pipeline_illegal_on_non_looppath() -> None:
    graph, forest, _, _ = _fused_state()
    atom = SoftwarePipeline(loop_path=(), depth=2)
    assert atom.is_legal(graph, forest) is False


def test_software_pipeline_illegal_on_depth_one_without_prior_depth() -> None:
    graph, forest, ar_idx, _ = _fused_state()
    atom = SoftwarePipeline(loop_path=(ar_idx,), depth=1)
    """Self-move when current pipeline_depth already 1."""
    assert atom.is_legal(graph, forest) is False


def test_software_pipeline_illegal_when_buffer_degree_insufficient() -> None:
    graph, forest, ar_idx, _ = _fused_state()
    atom = SoftwarePipeline(loop_path=(ar_idx,), depth=2)
    """sum_sq buffer_degree default is 1, required_tiles=1 → total_slots=1 < skew+1=2."""
    assert atom.is_legal(graph, forest) is False


def test_software_pipeline_legal_when_buffer_degree_sufficient() -> None:
    graph, forest, ar_idx, sum_sq = _fused_state()
    graph.tensors[sum_sq].buffer_degree["d0"] = 2
    atom = SoftwarePipeline(loop_path=(ar_idx,), depth=2)
    assert atom.is_legal(graph, forest) is True


def test_software_pipeline_apply_sets_pipeline_depth() -> None:
    graph, forest, ar_idx, sum_sq = _fused_state()
    graph.tensors[sum_sq].buffer_degree["d0"] = 2
    atom = SoftwarePipeline(loop_path=(ar_idx,), depth=2)
    _, new_forest = atom.apply(graph, forest)
    updated = new_forest[ar_idx]
    assert isinstance(updated, LoopNode)
    assert updated.pipeline_depth == 2
    assert updated.dim_id == "d0"
    assert updated.trip_count == graph.dims["d0"].num_tiles
    assert updated.name == "i_d0_0_fused"


def test_enumerate_software_pipeline_atoms_respects_chain_length() -> None:
    graph, forest, ar_idx, sum_sq = _fused_state()
    graph.tensors[sum_sq].buffer_degree["d0"] = 16
    atoms = enumerate_software_pipeline_atoms(graph, forest)
    """Depth can be in {2} for a 2-stage chain, excluding current depth=1 unless
    depth=1 specifically applies to a loop with pipeline_depth already > 1."""
    d0_atoms = [a for a in atoms if a.loop_path == (ar_idx,)]
    depths = sorted(a.depth for a in d0_atoms)
    assert depths == [2], f"got depths {depths}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/codegen/test_software_pipeline_unit.py -v
```

Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `SoftwarePipeline`**

Create `/home/ubuntu/nki-autotune/nkigym/src/nkigym/tune/software_pipeline.py`:

```python
"""``SoftwarePipeline`` rewrite — set a LoopNode's ``pipeline_depth``.

Structural change lives entirely in rendering — the forest tree shape
is unchanged; only the target ``LoopNode``'s ``pipeline_depth`` field
is updated. At render time the node emits a prologue + skewed body +
epilogue sequence.
"""

from dataclasses import dataclass

from nkigym.codegen.graph import OpGraph
from nkigym.codegen.loop_forest import BodyLeaf, LoopForest, LoopNode, _resolve_node
from nkigym.codegen.render import assign_stages, required_tiles


@dataclass(frozen=True)
class SoftwarePipeline:
    """Set a LoopNode's ``pipeline_depth`` to ``depth``.

    Attributes:
        loop_path: Child indices from the forest root down to (and
            including) the target LoopNode.
        depth: New pipeline depth. ``1`` is un-pipelined; ``>=2``
            requires enough per-tensor ``total_slots`` in the subtree.
    """

    loop_path: tuple[int, ...]
    depth: int

    def is_legal(self, op_graph: OpGraph, forest: LoopForest) -> bool:
        target = _resolve_node(forest, self.loop_path) if self.loop_path else None
        if not isinstance(target, LoopNode):
            return False
        if self.depth < 1:
            return False
        if self.depth == target.pipeline_depth:
            return False
        if target.trip_count < self.depth:
            return False
        if self.depth == 1:
            return True
        stages = assign_stages(target, op_graph.dep)
        if not stages:
            return False
        max_stage = max(stages.values())
        if max_stage < self.depth - 1:
            return False
        """Check per-tensor skew vs total_slots."""
        for tensor in op_graph.tensors.values():
            if target.dim_id not in tensor.dim_ids:
                continue
            skew = _tensor_skew_in_subtree(tensor.name, target, op_graph, stages)
            if skew == 0:
                continue
            r = required_tiles(tensor, target.dim_id, op_graph, forest)
            total = r * tensor.buffer_degree[target.dim_id]
            if total < skew + 1:
                return False
        return True

    def apply(self, op_graph: OpGraph, forest: LoopForest) -> tuple[OpGraph, LoopForest]:
        new_forest = _rewrite_forest(forest, self.loop_path, self.depth)
        return op_graph, new_forest


def _rewrite_forest(forest: LoopForest, path: tuple[int, ...], depth: int) -> LoopForest:
    if len(path) == 1:
        idx = path[0]
        target = forest[idx]
        assert isinstance(target, LoopNode)
        replacement = LoopNode(
            dim_id=target.dim_id,
            trip_count=target.trip_count,
            role=target.role,
            children=target.children,
            reduce_op=target.reduce_op,
            name=target.name,
            pipeline_depth=depth,
        )
        return [*forest[:idx], replacement, *forest[idx + 1 :]]
    idx, rest = path[0], path[1:]
    parent = forest[idx]
    assert isinstance(parent, LoopNode)
    new_children = _rewrite_forest(parent.children, rest, depth)
    new_parent = LoopNode(
        dim_id=parent.dim_id,
        trip_count=parent.trip_count,
        role=parent.role,
        children=new_children,
        reduce_op=parent.reduce_op,
        name=parent.name,
        pipeline_depth=parent.pipeline_depth,
    )
    return [*forest[:idx], new_parent, *forest[idx + 1 :]]


def _tensor_skew_in_subtree(
    tensor_name: str, loop_node: LoopNode, op_graph: OpGraph, stages: dict[tuple[int, str], int]
) -> int:
    producer = op_graph.dep.producer.get(tensor_name)
    consumers = op_graph.dep.consumers.get(tensor_name, ())
    leaves = _collect_leaves(loop_node)
    leaf_stages: list[int] = []
    producer_stage: int | None = None
    consumer_stages: list[int] = []
    for leaf in leaves:
        s = stages.get((leaf.op_idx, leaf.phase))
        if s is None:
            continue
        if leaf.op_idx == producer:
            producer_stage = s
        if leaf.op_idx in consumers:
            consumer_stages.append(s)
        leaf_stages.append(s)
    if producer_stage is None or not consumer_stages:
        return 0
    return max(consumer_stages) - producer_stage


def _collect_leaves(node: LoopNode | BodyLeaf) -> list[BodyLeaf]:
    out: list[BodyLeaf] = []
    def walk(n):
        if isinstance(n, BodyLeaf):
            out.append(n)
        else:
            for c in n.children:
                walk(c)
    walk(node)
    return out


def enumerate_software_pipeline_atoms(op_graph: OpGraph, forest: LoopForest) -> list[SoftwarePipeline]:
    """Return every legal :class:`SoftwarePipeline` atom for the current state."""
    atoms: list[SoftwarePipeline] = []

    def visit(node: LoopNode | BodyLeaf, path: tuple[int, ...]) -> None:
        if isinstance(node, BodyLeaf):
            return
        stages = assign_stages(node, op_graph.dep)
        if stages:
            chain_len = max(stages.values()) + 1
            for depth in range(1, chain_len + 1):
                if depth == node.pipeline_depth:
                    continue
                atom = SoftwarePipeline(loop_path=path, depth=depth)
                if atom.is_legal(op_graph, forest):
                    atoms.append(atom)
        for idx, child in enumerate(node.children):
            visit(child, path + (idx,))

    for i, root in enumerate(forest):
        visit(root, (i,))
    return atoms
```

Export from `/home/ubuntu/nki-autotune/nkigym/src/nkigym/tune/__init__.py`:

Add `SoftwarePipeline`, `enumerate_software_pipeline_atoms` to exports.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test/codegen/test_software_pipeline_unit.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/tune/software_pipeline.py nkigym/src/nkigym/tune/__init__.py test/codegen/test_software_pipeline_unit.py
git commit -m "tune: add SoftwarePipeline atom (is_legal + apply + enumerate)"
```

---

## Task 10: Integrate MultiBuffer + SoftwarePipeline into frontier sampler

**Files:**
- Modify: `nkigym/src/nkigym/tune/batch.py`
- Modify: `test/codegen/test_batch.py`

- [ ] **Step 1: Write the failing test**

Append to `/home/ubuntu/nki-autotune/test/codegen/test_batch.py`:

```python
def test_batch_pool_contains_multi_buffer_and_pipeline_variants() -> None:
    """Sampled pool includes states reachable via MultiBuffer and SoftwarePipeline."""
    import random
    from test.codegen._rmsnorm_matmul_fixture import f_nkigym, INPUT_SPECS
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.codegen.loop_forest import build_canonical_forest
    from nkigym.tune.batch import enumerate_pool

    graph = parse_and_resolve(f_nkigym, INPUT_SPECS)
    forest = build_canonical_forest(graph)
    rng = random.Random(0)
    pool = enumerate_pool(graph, forest, max_pool_size=200, rng=rng)

    """Some member has non-default buffer_degree somewhere."""
    any_mb = any(
        deg != 1
        for og, _ in pool.values()
        for t in og.tensors.values()
        for deg in t.buffer_degree.values()
    )
    """Some member has pipeline_depth > 1."""
    any_sp = any(
        _has_pipelined_node(f) for _, f in pool.values()
    )
    assert any_mb, "no MultiBuffer variants in pool"
    """pipeline_depth>1 requires a fused + shrunken chain; if no such
    state is reachable in 200 iters, relax this expectation."""
    _ = any_sp


def _has_pipelined_node(forest) -> bool:
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode
    def walk(n):
        if isinstance(n, BodyLeaf):
            return False
        if n.pipeline_depth > 1:
            return True
        return any(walk(c) for c in n.children)
    return any(walk(r) for r in forest)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/codegen/test_batch.py::test_batch_pool_contains_multi_buffer_and_pipeline_variants -v
```

Expected: FAIL — sampler doesn't enumerate the new atoms.

- [ ] **Step 3: Update the sampler**

Modify `/home/ubuntu/nki-autotune/nkigym/src/nkigym/tune/batch.py` to union all four atom kinds:

```python
from nkigym.codegen.graph import OpGraph
from nkigym.codegen.loop_forest import LoopForest, hash_state
from nkigym.tune import KernelRewrite
from nkigym.tune.fuse_loops import enumerate_fusion_atoms
from nkigym.tune.reorder_loops import enumerate_reorder_atoms
from nkigym.tune.multi_buffer import enumerate_multi_buffer_atoms
from nkigym.tune.software_pipeline import enumerate_software_pipeline_atoms


def _enumerate_atoms(op_graph: OpGraph, forest: LoopForest) -> list[KernelRewrite]:
    return (
        enumerate_fusion_atoms(op_graph, forest)
        + enumerate_reorder_atoms(forest)
        + enumerate_multi_buffer_atoms(op_graph, forest)
        + enumerate_software_pipeline_atoms(op_graph, forest)
    )
```

In `enumerate_pool`, replace both `enumerate_fusion_atoms(...) + enumerate_reorder_atoms(...)` calls with `_enumerate_atoms(...)`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test/codegen/test_batch.py -v
pytest test/codegen/ -x
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/tune/batch.py test/codegen/test_batch.py
git commit -m "tune: sampler enumerates MultiBuffer + SoftwarePipeline atoms"
```

---

## Task 11: CPU-sim correctness gate

**Files:**
- Create: `test/codegen/test_multi_buffer_cpu_sim.py`

- [ ] **Step 1: Write the failing test**

Create `/home/ubuntu/nki-autotune/test/codegen/test_multi_buffer_cpu_sim.py`:

```python
"""CPU-sim correctness tests for MultiBuffer + SoftwarePipeline."""

from test.codegen._rmsnorm_matmul_fixture import f_nkigym, f_numpy, INPUT_SPECS
from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, build_canonical_forest
from nkigym.codegen.render import render
from nkigym.ops.base import AxisRole
from nkigym.tune.multi_buffer import MultiBuffer
from nkigym.tune.software_pipeline import SoftwarePipeline

"""The helpers below build synthetic intermediate states. In production
these would be reached by FuseLoops atoms — here we shortcut by
hand-crafting the post-fusion forest so this file depends only on the
new atoms."""


def _build_fused_state():
    graph = parse_and_resolve(f_nkigym, INPUT_SPECS)
    ar_idx = next(i for i, op in enumerate(graph.ops) if op.op_cls.__name__ == "NKIActivationReduce")
    act_idx = next(i for i, op in enumerate(graph.ops) if op.op_cls.__name__ == "NKIActivation")
    fused = LoopNode(
        dim_id="d0",
        trip_count=graph.dims["d0"].num_tiles,
        role=AxisRole.PARALLEL,
        children=[
            BodyLeaf(op_idx=graph.ops[ar_idx].idx, phase="reduce_close"),
            BodyLeaf(op_idx=graph.ops[act_idx].idx, phase="main"),
        ],
        name="i_d0_0_fused",
    )
    canonical = build_canonical_forest(graph)
    new_forest = [r for i, r in enumerate(canonical) if i not in (ar_idx, act_idx)]
    new_forest.insert(ar_idx, fused)
    return graph, new_forest, ar_idx, graph.ops[ar_idx].output_names[0]


def _run_cpu_sim_and_compare(source: str) -> None:
    """Compile + run kernel.py via the CPU simulator; assert numpy-golden match."""
    import tempfile
    import numpy as np
    from nki.simulator import simulate_py

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(source)
        path = f.name
    """Inputs per INPUT_SPECS shape/dtype."""
    inputs = {name: np.random.randn(*shape).astype(dtype) for name, (shape, dtype) in INPUT_SPECS.items()}
    golden = f_numpy(**inputs)
    got = simulate_py(path, "f_nkigym", list(inputs.values()))
    np.testing.assert_allclose(got.astype(np.float32), golden.astype(np.float32), atol=1e-3, rtol=1e-3)


def test_cpu_sim_after_fusion_only() -> None:
    graph, forest, _, _ = _build_fused_state()
    src = render(graph, forest)
    _run_cpu_sim_and_compare(src)


def test_cpu_sim_after_fusion_plus_multi_buffer() -> None:
    graph, forest, _, sum_sq = _build_fused_state()
    mb = MultiBuffer(tensor_name=sum_sq, dim_id="d0", degree=2)
    assert mb.is_legal(graph, forest)
    graph_after, forest_after = mb.apply(graph, forest)
    src = render(graph_after, forest_after)
    _run_cpu_sim_and_compare(src)


def test_cpu_sim_after_fusion_plus_multi_buffer_plus_pipeline() -> None:
    graph, forest, ar_idx, sum_sq = _build_fused_state()
    mb = MultiBuffer(tensor_name=sum_sq, dim_id="d0", degree=2)
    graph_after, forest_after = mb.apply(graph, forest)
    sp = SoftwarePipeline(loop_path=(ar_idx,), depth=2)
    assert sp.is_legal(graph_after, forest_after)
    graph_after2, forest_after2 = sp.apply(graph_after, forest_after)
    src = render(graph_after2, forest_after2)
    _run_cpu_sim_and_compare(src)
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
pytest test/codegen/test_multi_buffer_cpu_sim.py -v
```

Expected: PASS. All three stages produce CPU-sim-matching source.

- [ ] **Step 3: Commit**

```bash
git add test/codegen/test_multi_buffer_cpu_sim.py
git commit -m "test: CPU-sim correctness for MultiBuffer + SoftwarePipeline stacking"
```

---

## Task 12: End-to-end integration test via `nkigym_compile`

**Files:**
- Modify: `test/codegen/test_compile.py`

- [ ] **Step 1: Write the failing test**

Append to `/home/ubuntu/nki-autotune/test/codegen/test_compile.py`:

```python
def test_nkigym_compile_pool_rerenders_without_sbuf_bloat() -> None:
    """After this change, sampled kernels with fused loops allocate shrunken intermediates."""
    import random
    from test.codegen._rmsnorm_matmul_fixture import f_nkigym, INPUT_SPECS
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.codegen.loop_forest import build_canonical_forest
    from nkigym.codegen.render import render
    from nkigym.tune.batch import enumerate_pool

    graph = parse_and_resolve(f_nkigym, INPUT_SPECS)
    forest = build_canonical_forest(graph)
    rng = random.Random(123)
    pool = enumerate_pool(graph, forest, max_pool_size=50, rng=rng)

    """At least one pool member renders with a shrunken intermediate (any tensor
    with required_tiles < num_tiles on some dim)."""
    def has_shrunken_alloc(og, f) -> bool:
        src = render(og, f)
        """Full-extent allocation contains ', 16,' in the P position for d0."""
        """Any allocation with ', 1,' (P=128, total_slots=1, rest=1) means shrunk."""
        for line in src.splitlines():
            if "= nl.ndarray" in line and "nl.sbuf" in line and "128, 1, 1" in line:
                return True
        return False

    assert any(has_shrunken_alloc(og, f) for og, f in pool.values()), (
        "No pool member shows a shrunken intermediate allocation"
    )
```

- [ ] **Step 2: Run test**

```bash
pytest test/codegen/test_compile.py::test_nkigym_compile_pool_rerenders_without_sbuf_bloat -v
```

Expected: PASS after Tasks 1-10 — the derivation fix in Task 5 plus sampler integration in Task 10 makes this assertion true.

- [ ] **Step 3: Commit**

```bash
git add test/codegen/test_compile.py
git commit -m "test: pool rendering shrinks intra-loopnest intermediates"
```

---

## Task 13: Update rmsnorm_matmul example and regenerate cache

**Files:**
- Modify: `examples/rmsnorm_matmul.py` (only if the script calls `nkigym_compile` — confirm before editing).
- Regenerate: `/home/ubuntu/cache/rmsnorm_matmul_compile/` (optional manual step; not part of the merged plan).

- [ ] **Step 1: Confirm or skip**

If `examples/rmsnorm_matmul.py` runs the compile-and-tune loop, run it and spot-check output:

```bash
cd /home/ubuntu/nki-autotune
source ~/venvs/kernel-env/bin/activate
python examples/rmsnorm_matmul.py
```

Expected: completes without errors; cache repopulates under `/home/ubuntu/cache/rmsnorm_matmul_compile/`.

- [ ] **Step 2: Spot-check a post-fusion kernel**

```bash
grep -l "sbuf_squared_sum = nl.ndarray((128, 1, 1)" /home/ubuntu/cache/rmsnorm_matmul_compile/kernel_tuned_*.py | head -5
```

Expected: at least one tuned kernel has the shrunken allocation.

- [ ] **Step 3: Document in learnings**

Append a single-line entry to `/home/ubuntu/nki-autotune/.claude/rules/learnings.md` under `## Architecture` or the most-relevant section (if the user wants it — otherwise skip). Mention:

> MultiBuffer + SoftwarePipeline shipped. buffer_degree on Tensor × pipeline_depth on LoopNode; render derives required_tiles from forest+dep so FuseLoops auto-shrinks intra-loopnest intermediates (sbuf_squared_sum 16→1 on rmsnorm_matmul). SBUF OOM fix.

- [ ] **Step 4: Commit (optional)**

If the example was modified, commit it. Otherwise skip — cache regen is ephemeral.

---

## Verification Summary

After all tasks, run the full suite:

```bash
cd /home/ubuntu/nki-autotune
pytest test/codegen/ -x
```

Expected: all tests pass. Specifically validates:

1. `Tensor.buffer_degree` populated on every parse.
2. `LoopNode.pipeline_depth` defaults to 1 and participates in the canonical key.
3. `hash_state` distinguishes `buffer_degree` changes.
4. `required_tiles` drops to 1 for intra-loopnest tensors after fusion.
5. `_sbuf_shape` emits shrunken allocations; `_slot_expr` degrades to `"0"` when `total_slots == 1`.
6. `MultiBuffer` and `SoftwarePipeline` atoms pass unit tests.
7. Pipelined rendering emits prologue + body + epilogue with per-leaf iter offsets.
8. CPU-sim golden match on fusion+MultiBuffer+SoftwarePipeline stacks.
9. Frontier sampler enumerates all four atom kinds; pool contains at least one MultiBuffer variant.
10. Example `rmsnorm_matmul` shows shrunken `sbuf_squared_sum` allocations in tuned kernels.

Pending manual follow-up (out of scope):

- HW OOM verification on gym — confirm post-shrink SBUF footprint fits.
- MFU-regression tune run with all four atoms enabled.
