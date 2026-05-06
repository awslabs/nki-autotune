# Activation-Reduce Pattern 2 Codegen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change `NKIActivationReduce` codegen to lower the F reduction into `N` per-F-tile `nisa.activation_reduce` calls that write distinct slots of a partial-sum vector, followed by one closing `nisa.tensor_reduce` that folds those slots into the op's `(P, 1)` output — matching the production nkilib Pattern 2 used in `attention_cte.py` online softmax. Replaces the current SBUF-merge lowering that emits `memset` + per-tile `activation_reduce` + per-tile `tensor_tensor` fold.

**Architecture:** Introduce op-local buffers (naming `sbuf_local_<id>` / `psum_local_<id>`) that the renderer allocates at function top but sizes at single-iteration scope (no `num_p_tiles` blow-up). `NKIActivationReduce` gains two op-local SBUF buffers — a scratch for the discarded `dst` of `activation_reduce`, and a slot vector for the per-F-tile partial sums. Op-local F-axis dims are sized by tile-count, not tile-element-count; a new derived dim `F_slot` handles this. The `reducer_init` phase is dropped (nothing to do — each slot is overwritten exactly once); a new `reduce_close` phase emits the single `nisa.tensor_reduce` after the F loop exits. Pattern 1 (single-call over full F) is out of scope: a future hoist transform + DCE on the closing `tensor_reduce` will reach it monotonically.

**Tech Stack:** Python 3.12, `nki`, `nki.isa`, `nki.language`, `ast`, `pytest`. Runtime environment: `source ~/venvs/kernel-env/bin/activate`. Codebase: `nkigym` package.

---

## Reference map (read these before starting)

- Current `NKIActivationReduce` op: `nkigym/src/nkigym/ops/activation_reduce.py`
- Current `NKIOp` base: `nkigym/src/nkigym/ops/base.py`
- Current forest builder for `NKIActivationReduce`: `nkigym/src/nkigym/codegen/loop_forest.py:289-310, 313-320, 325-326, 332`
- Current body emitters: `nkigym/src/nkigym/codegen/render.py:475-529`
- Current SBUF allocation pass: `nkigym/src/nkigym/codegen/render.py:109-138`
- Current `OpGraph` / `ParsedOp` schema: `nkigym/src/nkigym/codegen/graph.py`
- Target rendered kernel (pre-change, for diff reference): `/home/ubuntu/cache/rmsnorm_matmul_compile/kernel.py:28-43` — the 34-op `reduce_step` region we are replacing.
- Production Pattern 2 reference: `/home/ubuntu/shared_workplace/KaenaNeuronKernelLibrary/src/nkilib_src/nkilib/core/attention/attention_cte.py:2251-2297` (per-tile `activation_reduce` writing distinct `reduce_res` slot) and `:2453` (closing `tensor_reduce(axis=1)`).
- Production Pattern 1 reference (for future hoist): `/home/ubuntu/shared_workplace/KaenaNeuronKernelLibrary/src/nkilib_src/nkilib/core/rmsnorm/rmsnorm_quant.py:446-462`.
- Tests that currently lock in Pattern 3 and need to change: `test/codegen/test_loop_forest.py:213-249, 316-330`; `test/codegen/test_render.py:292-340, 626-650`.
- Project conventions: `/home/ubuntu/.claude/rules/code_style.md` (docstrings, triple-quoted block comments instead of `#`, 120-char lines, Google/NumPy docstring style, modern type hints).
- Code-style hook: `check-python-style.py` runs on every `.py` edit. If it blocks, fix the underlying issue — don't bypass.

## File structure

**Modify:**
- `nkigym/src/nkigym/ops/base.py` — add `OP_LOCAL_BUFFERS` ClassVar to `NKIOp`.
- `nkigym/src/nkigym/ops/activation_reduce.py` — declare `OP_LOCAL_BUFFERS`; drop `AXIS_ROLES` unchanged (F stays `ACCUMULATION`).
- `nkigym/src/nkigym/codegen/graph.py` — resolve op-local buffers + op-local derived dims during `parse_and_resolve`; attach to `ParsedOp` and `OpGraph.dims`.
- `nkigym/src/nkigym/codegen/loop_forest.py` — drop `reducer_init` phase; add `reduce_close` phase to `_build_leaves_activation_reduce` + `_phase_dims_activation_reduce`.
- `nkigym/src/nkigym/codegen/render.py` — emit op-local buffer allocations at function top with op-local sizing; replace `_body_ar_reducer_init` with `_body_ar_reduce_close`; rewrite `_body_ar_reduce_step` to target slot vector; drop the `tmp_red` + `tensor_tensor` merge; rewrite `_body_ar_post_op` to read from `sbuf_rms_inv` unchanged (but now populated by `reduce_close`, not by F-loop merges).
- `test/codegen/test_loop_forest.py` — update the two locked-in `reducer_init` tests to expect `reduce_close` + remove `reducer_init` assertions.
- `test/codegen/test_render.py` — update the Pattern 3 assertions (`nisa.memset(` + `tensor_tensor`) to Pattern 2 assertions (`nisa.tensor_reduce(`, no `tensor_tensor`, slot-vector buffer allocated).

**No new files.** Everything is a modification to existing code and tests.

---

## Task 0: Scaffolding + baseline verification

**Files:** none (verification only).

- [ ] **Step 1: Activate venv + sanity-check tooling**

Run: `source ~/venvs/kernel-env/bin/activate && python -c "import nki, nki.simulate, nki.isa; print('ok')"`
Expected: `ok`.

- [ ] **Step 2: Confirm baseline test suite passes on current code**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test/codegen/ -x -q`
Expected: all tests pass. If any failures appear, stop and surface them before making changes — this plan assumes a green baseline.

- [ ] **Step 3: Capture baseline rendered kernel for after-diff**

Run:
```bash
source ~/venvs/kernel-env/bin/activate
cp /home/ubuntu/cache/rmsnorm_matmul_compile/kernel.py /tmp/kernel_before_pattern2.py
```
Expected: file copied. The post-implementation kernel must drop `nisa.memset(sbuf_rms_inv` and the `nisa.tensor_tensor(sbuf_rms_inv, sbuf_rms_inv, tmp_red, op=nl.add)` line; must add `nisa.tensor_reduce(sbuf_rms_inv[...], nl.add, sbuf_local_<id>, axis=2)`.

- [ ] **Step 4: Create a working branch marker commit**

```bash
git status
```
Expected: clean working tree on `dev_1`. If dirty, commit or stash separately before starting.

---

## Task 1: Add `OP_LOCAL_BUFFERS` ClassVar to `NKIOp`

**Files:**
- Modify: `nkigym/src/nkigym/ops/base.py:87-124`
- Test: `test/codegen/test_graph.py` (new test)

- [ ] **Step 1: Write the failing test**

Append to `test/codegen/test_graph.py`:

```python
def test_op_local_buffers_defaults_to_empty() -> None:
    """NKIOp.OP_LOCAL_BUFFERS defaults to an empty dict so existing ops are unaffected."""
    from nkigym.ops.base import NKIOp
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.load import NKILoad

    assert NKIOp.OP_LOCAL_BUFFERS == {}
    assert NKIMatmul.OP_LOCAL_BUFFERS == {}
    assert NKILoad.OP_LOCAL_BUFFERS == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/codegen/test_graph.py::test_op_local_buffers_defaults_to_empty -v`
Expected: FAIL with `AttributeError: type object 'NKIOp' has no attribute 'OP_LOCAL_BUFFERS'`.

- [ ] **Step 3: Add the ClassVar to `NKIOp`**

In `nkigym/src/nkigym/ops/base.py`, modify the class attribute block at lines 119-124 to add `OP_LOCAL_BUFFERS`:

```python
    NAME: ClassVar[str] = ""
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    OUTPUT_DTYPES: ClassVar[dict[str, str]] = {}
    AXIS_ROLES: ClassVar[dict[str, "AxisRole"]] = {}
    TILE_LIMITS: ClassVar[dict[str, int]] = {}
    OP_LOCAL_BUFFERS: ClassVar[dict[str, tuple[str, str, tuple[str, ...]]]] = {}
    """Op-local buffers allocated at function top but sized at single-iteration
    scope (no outer-dim blow-up). Each entry maps a logical name to
    ``(location, dtype, axis_ids)`` — ``location`` ∈ ``{"sbuf", "psum"}``,
    ``dtype`` is an ``nl.*`` dtype name (e.g. ``"float32"``), ``axis_ids``
    are abstract axis labels from ``OPERAND_AXES`` keys plus any op-local
    derived axes declared on the op. The emitted buffer identifiers are
    ``{location}_local_<id>`` where ``<id>`` is assigned per op instance
    monotonically across the kernel."""
```

Also update the class docstring (`nkigym/src/nkigym/ops/base.py:104-117`) to mention `OP_LOCAL_BUFFERS`. Insert after the `TILE_LIMITS` docstring bullet:

```python
        OP_LOCAL_BUFFERS: Buffers allocated at function top but sized at
            single-iteration scope (see ClassVar docstring). Empty for
            most ops; used by reducers that need cross-phase scratch
            within one tile's iteration.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/codegen/test_graph.py::test_op_local_buffers_defaults_to_empty -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ops/base.py test/codegen/test_graph.py
git commit -m "$(cat <<'EOF'
nkigym: add OP_LOCAL_BUFFERS ClassVar to NKIOp

Schema-only addition; no op uses it yet. Subsequent commits wire it
through graph/render for activation_reduce's Pattern 2 lowering.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add `F_slot` derived dim support to `OpGraph`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/graph.py`
- Test: `test/codegen/test_graph.py`

Context: when an op declares an op-local buffer shaped `(P, F_slot)`, the `F_slot` axis must resolve to a concrete derived `DimInfo` whose `num_tiles` equals the source F axis's `num_tiles` and whose `tile_size` is 1. The derived dim must be unique per op instance (`d<N>_f_slot` where `d<N>` is the concrete F dim).

- [ ] **Step 1: Write the failing test for derived-dim registration**

Append to `test/codegen/test_graph.py`:

```python
def test_parse_and_resolve_registers_op_local_derived_dims_for_activation_reduce() -> None:
    """After parse_and_resolve, an F_slot derived dim exists for each activation_reduce op.

    The derived dim's tile_size is 1 and num_tiles matches the source F dim's num_tiles.
    """
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.activation_reduce import NKIActivationReduce
    from nkigym.ops.load import NKILoad
    from nkigym.ops.store import NKIStore

    @nkigym_kernel
    def _rms(x):
        xs = NKILoad()(data=x)
        m = NKIActivationReduce(op="square", reduce_op="add", post_op="rsqrt", scale=1 / 2048, bias=1e-6)(data=xs)
        out = NKIStore()(data=m)
        return out

    g = parse_and_resolve(_rms, {"x": ((128, 2048), "bfloat16")})
    ar_op = next(o for o in g.ops if o.op_cls.__name__ == "NKIActivationReduce")
    f_dim_id = ar_op.axis_map["F"]
    f_info = g.dims[f_dim_id]
    assert f_info.tile_size == 512
    assert f_info.num_tiles == 4

    """A derived F_slot dim must be registered for this op instance."""
    f_slot_dim_ids = [d for d in g.dims if d.startswith(f_dim_id) and d.endswith("_f_slot")]
    assert len(f_slot_dim_ids) == 1
    slot_info = g.dims[f_slot_dim_ids[0]]
    assert slot_info.tile_size == 1
    assert slot_info.num_tiles == f_info.num_tiles
    assert slot_info.total_size == f_info.num_tiles
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/codegen/test_graph.py::test_parse_and_resolve_registers_op_local_derived_dims_for_activation_reduce -v`
Expected: FAIL — derived dim does not exist yet.

- [ ] **Step 3: Add `scratch_dims` resolution to `parse_and_resolve`**

In `nkigym/src/nkigym/codegen/graph.py`, modify `parse_and_resolve` (currently around lines 291-341) to register op-local derived dims after `_resolve_dimensions` runs.

Add a new helper function near `_resolve_dimensions`:

```python
def _register_op_local_derived_dims(
    raws: list[_ParsedOpRaw],
    per_op_axis_maps: list[dict[str, str]],
    dims: dict[str, DimInfo],
) -> None:
    """Register derived dims referenced by op-local buffer axis_ids.

    For now, the only derived axis label recognized is ``F_slot``: one
    element per F-tile, inherited from the op's F dim's ``num_tiles``.
    For each op that declares an ``OP_LOCAL_BUFFERS`` entry mentioning
    ``F_slot``, we register a derived ``DimInfo`` keyed ``<f_dim>_f_slot``
    with ``tile_size=1``, ``num_tiles=dims[f_dim].num_tiles``.

    Args:
        raws: Parsed ops in source order.
        per_op_axis_maps: Concrete axis_maps aligned with ``raws``.
        dims: The resolved dim table (mutated in place to add derived dims).
    """
    for raw, axis_map in zip(raws, per_op_axis_maps):
        local_buffers = getattr(raw.op_cls, "OP_LOCAL_BUFFERS", {})
        for _, (_, _, axis_ids) in local_buffers.items():
            for axis_id in axis_ids:
                if axis_id == "F_slot":
                    if "F" not in axis_map:
                        raise ValueError(
                            f"Op {raw.op_cls.NAME}: OP_LOCAL_BUFFERS references F_slot "
                            f"but op has no F axis in axis_map"
                        )
                    f_dim_id = axis_map["F"]
                    derived_id = f"{f_dim_id}_f_slot"
                    if derived_id not in dims:
                        f_info = dims[f_dim_id]
                        dims[derived_id] = DimInfo(
                            dim_id=derived_id,
                            total_size=f_info.num_tiles,
                            tile_size=1,
                            num_tiles=f_info.num_tiles,
                        )
```

In `parse_and_resolve`, call it immediately after `_resolve_dimensions`:

```python
    dims = _resolve_dimensions(raws, per_op_axis_maps, dim_sizes)
    _register_op_local_derived_dims(raws, per_op_axis_maps, dims)
    tensors = _finalize_tensors(tensors_scratch, param_names, return_name, raws)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/codegen/test_graph.py::test_parse_and_resolve_registers_op_local_derived_dims_for_activation_reduce -v`
Expected: FAIL with `Op activation_reduce: OP_LOCAL_BUFFERS references F_slot but op has no F axis in axis_map` — because `NKIActivationReduce.OP_LOCAL_BUFFERS` is still empty. That's correct: Task 3 declares the buffers. To verify the plumbing without relying on Task 3, add a temporary declaration:

In `nkigym/src/nkigym/ops/activation_reduce.py`, under the class body, add (will be refined in Task 3):

```python
    OP_LOCAL_BUFFERS: ClassVar[dict[str, tuple[str, str, tuple[str, ...]]]] = {
        "scratch": ("sbuf", "float32", ("P", "F")),
        "slot_vec": ("sbuf", "float32", ("P", "F_slot")),
    }
```

Run the test again:

Run: `pytest test/codegen/test_graph.py::test_parse_and_resolve_registers_op_local_derived_dims_for_activation_reduce -v`
Expected: PASS.

- [ ] **Step 5: Run the full graph test suite to catch regressions**

Run: `pytest test/codegen/test_graph.py -x -q`
Expected: all tests pass. `OP_LOCAL_BUFFERS` on `NKIActivationReduce` may cause no regressions because nothing else consumes it yet.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/codegen/graph.py nkigym/src/nkigym/ops/activation_reduce.py test/codegen/test_graph.py
git commit -m "$(cat <<'EOF'
nkigym/graph: register F_slot derived dims for op-local buffers

F_slot is a per-op derived axis with tile_size=1 and num_tiles equal
to the source F dim's num_tiles. Used by NKIActivationReduce's
op-local slot vector (one element per F-tile). Declared on the op
via OP_LOCAL_BUFFERS; resolved during parse_and_resolve into a
distinct DimInfo keyed <f_dim>_f_slot.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Resolve op-local buffer names on `ParsedOp`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/graph.py`
- Test: `test/codegen/test_graph.py`

- [ ] **Step 1: Write the failing test**

Append to `test/codegen/test_graph.py`:

```python
def test_parsed_op_resolves_op_local_buffer_names_for_activation_reduce() -> None:
    """ParsedOp.op_local_buffers maps logical → (emitted_name, location, dtype, shape).

    Naming convention: sbuf_local_<id> / psum_local_<id>, id assigned per
    op instance in encounter order across OP_LOCAL_BUFFERS iteration order.
    """
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.activation_reduce import NKIActivationReduce
    from nkigym.ops.load import NKILoad
    from nkigym.ops.store import NKIStore

    @nkigym_kernel
    def _rms(x):
        xs = NKILoad()(data=x)
        m = NKIActivationReduce(op="square", reduce_op="add", post_op="rsqrt", scale=1 / 2048, bias=1e-6)(data=xs)
        out = NKIStore()(data=m)
        return out

    g = parse_and_resolve(_rms, {"x": ((128, 2048), "bfloat16")})
    ar_op = next(o for o in g.ops if o.op_cls.__name__ == "NKIActivationReduce")

    """Buffers dict is keyed by logical name declared on the op."""
    assert set(ar_op.op_local_buffers.keys()) == {"scratch", "slot_vec"}

    scratch = ar_op.op_local_buffers["scratch"]
    assert scratch.emitted_name == "sbuf_local_0"
    assert scratch.location == "sbuf"
    assert scratch.dtype == "float32"
    """Shape: (p_tile, 1, num_f_tiles * f_tile) — P dim contributes p_tile
    plus singleton block dim; F contributes num_f_tiles*f_tile."""
    assert scratch.shape == (128, 1, 2048)

    slot = ar_op.op_local_buffers["slot_vec"]
    assert slot.emitted_name == "sbuf_local_1"
    assert slot.location == "sbuf"
    assert slot.dtype == "float32"
    """Shape: (p_tile, 1, num_f_tiles) — F_slot.num_tiles=4, tile_size=1."""
    assert slot.shape == (128, 1, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/codegen/test_graph.py::test_parsed_op_resolves_op_local_buffer_names_for_activation_reduce -v`
Expected: FAIL — `ParsedOp` has no `op_local_buffers` attribute yet.

- [ ] **Step 3: Add `OpLocalBuffer` dataclass + `ParsedOp.op_local_buffers` field**

In `nkigym/src/nkigym/codegen/graph.py`, after the `DimInfo` dataclass (around line 59), add:

```python
@dataclass
class OpLocalBuffer:
    """Resolved op-local buffer ready for renderer emission.

    Attributes:
        logical_name: Name the op declared (e.g. ``"scratch"``, ``"slot_vec"``).
        emitted_name: Identifier the renderer uses in emitted source
            (e.g. ``"sbuf_local_0"``).
        location: ``"sbuf"`` or ``"psum"``.
        dtype: ``nl.*`` dtype name (e.g. ``"float32"``).
        shape: Emitted 3D shape ``(p_tile, 1, free_extent)``. Partition
            dim contributes ``p_tile``; outer block tier collapses to 1
            (op-local = no cross-iteration persistence); free axis
            contributes ``num_tiles * tile_size``.
    """

    logical_name: str
    emitted_name: str
    location: str
    dtype: str
    shape: tuple[int, int, int]
```

Extend `ParsedOp` (around lines 62-92) to add an `op_local_buffers` field:

```python
@dataclass
class ParsedOp:
    """One ``NKIOp()(...)`` call captured from the ``f_nkigym`` body.

    ... existing docstring bullets ...

        op_local_buffers: Resolved op-local buffers keyed by logical
            name. Renderer emits one allocation per entry at function
            top using ``emitted_name`` / ``location`` / ``shape``. Body
            emitters reference ``emitted_name`` directly. Empty when
            the op class declares no ``OP_LOCAL_BUFFERS``.
    """

    idx: int
    op_cls: type
    operand_names: dict[str, str]
    op_kwargs: dict[str, Any]
    output_names: list[str]
    axis_map: dict[str, str]
    touched_dims: tuple[str, ...]
    dim_role: dict[str, AxisRole]
    op_local_buffers: dict[str, OpLocalBuffer] = field(default_factory=dict)
```

- [ ] **Step 4: Add the resolution pass to `parse_and_resolve`**

Add a helper near `_build_parsed_ops`:

```python
def _resolve_op_local_buffers(
    raw: _ParsedOpRaw,
    axis_map: dict[str, str],
    dims: dict[str, DimInfo],
    counter: list[int],
) -> dict[str, OpLocalBuffer]:
    """Materialize op-local buffer records from ``OP_LOCAL_BUFFERS``.

    Each entry gets a unique ``emitted_name`` via the shared ``counter``.
    Buffer shape is computed from the declared ``axis_ids`` using
    op-local sizing rules: partition dim → ``(tile_size, 1, ...)``; the
    free dim contributes ``num_tiles * tile_size`` (one element per F
    element for ``F``, one element per F-tile for ``F_slot``).

    Args:
        raw: Parsed op record.
        axis_map: Concrete axis_map for this op.
        dims: Resolved dim table (must already include any derived
            ``F_slot`` entries; see :func:`_register_op_local_derived_dims`).
        counter: Mutable single-element list used to assign monotonic
            buffer ids across the whole kernel.

    Returns:
        Dict keyed by logical name.
    """
    out: dict[str, OpLocalBuffer] = {}
    local_buffers = getattr(raw.op_cls, "OP_LOCAL_BUFFERS", {})
    for logical_name, (location, dtype, axis_ids) in local_buffers.items():
        if len(axis_ids) != 2:
            raise ValueError(
                f"Op {raw.op_cls.NAME}: OP_LOCAL_BUFFERS['{logical_name}'] must have 2 axis_ids "
                f"(P, F-like), got {axis_ids}"
            )
        p_axis, f_axis = axis_ids
        p_dim_id = axis_map[p_axis]
        p_info = dims[p_dim_id]
        if f_axis == "F_slot":
            f_dim_id = f"{axis_map['F']}_f_slot"
        else:
            f_dim_id = axis_map[f_axis]
        f_info = dims[f_dim_id]
        shape = (p_info.tile_size, 1, f_info.num_tiles * f_info.tile_size)
        emitted = f"{location}_local_{counter[0]}"
        counter[0] += 1
        out[logical_name] = OpLocalBuffer(
            logical_name=logical_name,
            emitted_name=emitted,
            location=location,
            dtype=dtype,
            shape=shape,
        )
    return out
```

Update `_build_parsed_ops` to accept the `dims` table + a counter, and pass through the resolved buffers:

```python
def _build_parsed_ops(
    raws: list[_ParsedOpRaw],
    per_op_axis_maps: list[dict[str, str]],
    tensors: dict[str, Tensor],
    dims: dict[str, DimInfo],
) -> list[ParsedOp]:
    """Assemble per-op records with canonicalised ``touched_dims``."""
    ops: list[ParsedOp] = []
    counter = [0]
    for idx, (raw, axis_map) in enumerate(zip(raws, per_op_axis_maps)):
        touched = _touched_dims(raw, axis_map, tensors)
        dim_role = _resolve_dim_role(raw.op_cls, axis_map, touched)
        op_local_buffers = _resolve_op_local_buffers(raw, axis_map, dims, counter)
        ops.append(
            ParsedOp(
                idx=idx,
                op_cls=raw.op_cls,
                operand_names=dict(raw.operand_names),
                op_kwargs=dict(raw.op_kwargs),
                output_names=list(raw.output_names),
                axis_map=dict(axis_map),
                touched_dims=touched,
                dim_role=dim_role,
                op_local_buffers=op_local_buffers,
            )
        )
    return ops
```

And update the `parse_and_resolve` call site:

```python
    ops = _build_parsed_ops(raws, per_op_axis_maps, tensors, dims)
```

- [ ] **Step 5: Run the test**

Run: `pytest test/codegen/test_graph.py::test_parsed_op_resolves_op_local_buffer_names_for_activation_reduce -v`
Expected: PASS.

- [ ] **Step 6: Run the full graph test suite**

Run: `pytest test/codegen/test_graph.py -x -q`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add nkigym/src/nkigym/codegen/graph.py test/codegen/test_graph.py
git commit -m "$(cat <<'EOF'
nkigym/graph: resolve op-local buffers on ParsedOp

Adds OpLocalBuffer dataclass carrying (logical_name, emitted_name,
location, dtype, shape) and a ParsedOp.op_local_buffers field.
Names follow {sbuf|psum}_local_<id>; ids assigned monotonically
across the kernel. Shape sizes partition dim at (tile_size, 1, ...)
— single-iteration scope, no cross-nest persistence — and free dim
at num_tiles*tile_size.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Emit op-local buffer allocations in renderer

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py:109-122`
- Test: `test/codegen/test_render.py`

- [ ] **Step 1: Write the failing test**

Append to `test/codegen/test_render.py`:

```python
def test_render_emits_op_local_buffer_allocations_at_top() -> None:
    """Renderer emits one nl.ndarray per op-local buffer, after tensor allocations.

    Buffer shape matches ParsedOp.op_local_buffers[...].shape; identifiers
    use sbuf_local_<id> / psum_local_<id> from emitted_name.
    """
    specs = {"x": ((128, 2048), "bfloat16")}
    g = parse_and_resolve(_rms_kernel, specs)
    src = render(g)

    """scratch and slot_vec allocations both appear after the intermediate
    SBUF allocations (order: tensors first, then op-locals)."""
    assert "sbuf_local_0 = nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.sbuf)" in src
    assert "sbuf_local_1 = nl.ndarray((128, 1, 16), dtype=nl.float32, buffer=nl.sbuf)" in src
```

Note: `_rms_kernel` uses F=128 in current test — let's switch the specs in the new test to a case that produces 16 F-tiles. Actually with `specs = {"x": ((128, 2048), "bfloat16")}` and `NKIActivationReduce.TILE_LIMITS["F"] = VE_FREE_MAX = 512` (unchanged in this plan), F=2048 splits into 4 tiles of 512 each. Update the assertion accordingly:

```python
def test_render_emits_op_local_buffer_allocations_at_top() -> None:
    """Renderer emits one nl.ndarray per op-local buffer, after tensor allocations."""
    specs = {"x": ((128, 2048), "bfloat16")}
    g = parse_and_resolve(_rms_kernel, specs)
    src = render(g)

    """scratch: (p_tile=128, 1, num_f_tiles*f_tile=4*512=2048) — f_tile-wide per call."""
    assert "sbuf_local_0 = nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.sbuf)" in src
    """slot_vec: (p_tile=128, 1, num_f_tiles=4)."""
    assert "sbuf_local_1 = nl.ndarray((128, 1, 4), dtype=nl.float32, buffer=nl.sbuf)" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/codegen/test_render.py::test_render_emits_op_local_buffer_allocations_at_top -v`
Expected: FAIL — no `sbuf_local_*` identifier present yet.

- [ ] **Step 3: Extend `_emit_sbuf_allocations` to emit op-local buffers**

In `nkigym/src/nkigym/codegen/render.py`, modify `_emit_sbuf_allocations` (lines 109-122):

```python
def _emit_sbuf_allocations(w: _Writer, op_graph: OpGraph) -> None:
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
        shape = _sbuf_shape(tensor, op_graph)
        w.line(f"{_sbuf_name(name)} = nl.ndarray({shape}, dtype=nl.{tensor.dtype}, buffer=nl.sbuf)")
    for op in op_graph.ops:
        for buf in op.op_local_buffers.values():
            nl_buffer = "nl.sbuf" if buf.location == "sbuf" else "nl.psum"
            w.line(f"{buf.emitted_name} = nl.ndarray({buf.shape}, dtype=nl.{buf.dtype}, buffer={nl_buffer})")
    w.line()
```

- [ ] **Step 4: Run the test**

Run: `pytest test/codegen/test_render.py::test_render_emits_op_local_buffer_allocations_at_top -v`
Expected: PASS.

- [ ] **Step 5: Run the full render test suite**

Run: `pytest test/codegen/test_render.py -x -q`
Expected: existing tests may fail because the op-local buffers get emitted even though downstream emitters haven't been updated yet. Specifically, `test_render_emits_header_and_allocations` checks exact allocation content — op-local lines will appear too. Inspect the output and accept that:
- Tests asserting **presence** of a specific line keep passing.
- Tests asserting **absence** (`sbuf_out` not present) still pass because op-local naming is distinct.
- Tests asserting **exact line count** (if any) need updating.

If any test fails, update its assertion to account for the new op-local lines or stop and surface before continuing.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/codegen/render.py test/codegen/test_render.py
git commit -m "$(cat <<'EOF'
nkigym/render: emit op-local buffer allocations at function top

Walks op_graph.ops after the cross-nest tensor pass; emits one
nl.ndarray per OpLocalBuffer using emitted_name/location/shape.
Op-local buffers persist at the function-top scope but size at
single-iteration extent — (tile_size, 1, free_extent) — because
no consumer crosses loop nests.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Drop `reducer_init` phase + add `reduce_close` phase to forest builder

**Files:**
- Modify: `nkigym/src/nkigym/codegen/loop_forest.py:289-320, 325-326, 332`
- Test: `test/codegen/test_loop_forest.py` (update existing tests)

Context: `reducer_init` is dropped because with Pattern 2 no prologue memset is needed — each `activation_reduce` call writes a distinct `reduce_res` slot, never accumulates into an existing one. The closing phase `reduce_close` emits a single `nisa.tensor_reduce` that folds the slot vector into the op's `(P, 1)` output. `reduce_close` lives outside the F loop (same level as the dropped `reducer_init` and the existing `post_op`).

- [ ] **Step 1: Update the existing forest-shape tests**

Modify `test/codegen/test_loop_forest.py:213-249`. Replace the two tests with:

```python
def test_canonical_forest_activation_reduce_with_post_op_has_expected_leaves() -> None:
    """ActivationReduce with post_op emits F chain + reduce_close + post_op.

    Pattern 2: reducer_init is dropped (no prologue needed because each
    activation_reduce call writes a distinct slot); reduce_close folds
    the slot vector via nisa.tensor_reduce after the F loop exits.
    """
    from nkigym.codegen.loop_forest import build_canonical_forest

    g = _parse(_rms_kernel_with_post_op, _RMS_SPECS)
    forest = build_canonical_forest(g)
    ar_idx = next(i for i, op in enumerate(g.ops) if op.op_cls.__name__ == "NKIActivationReduce")
    tree = forest[ar_idx]
    p_tile = tree.children[0]
    children = p_tile.children
    """Three children: F chain, reduce_close, post_op (reducer_init dropped)."""
    assert len(children) == 3
    f_block = children[0]
    assert isinstance(f_block, LoopNode)
    f_tile = f_block.children[0]
    reduce_leaf = f_tile.children[0]
    assert isinstance(reduce_leaf, BodyLeaf) and reduce_leaf.phase == "reduce_step"
    assert isinstance(children[1], BodyLeaf) and children[1].phase == "reduce_close"
    assert isinstance(children[2], BodyLeaf) and children[2].phase == "post_op"


def test_canonical_forest_activation_reduce_no_post_op_omits_post_op_leaf() -> None:
    """When the op has no post_op, the post_op leaf is omitted but reduce_close stays."""
    from nkigym.codegen.loop_forest import build_canonical_forest

    g = _parse(_rms_kernel_without_post_op, _RMS_SPECS)
    forest = build_canonical_forest(g)
    ar_idx = next(i for i, op in enumerate(g.ops) if op.op_cls.__name__ == "NKIActivationReduce")
    tree = forest[ar_idx]
    p_tile = tree.children[0]
    children = p_tile.children
    """Two children: F chain, reduce_close. No post_op, no reducer_init."""
    assert len(children) == 2
    assert isinstance(children[0], LoopNode)
    assert isinstance(children[1], BodyLeaf) and children[1].phase == "reduce_close"
    for c in children:
        if isinstance(c, BodyLeaf):
            assert c.phase != "post_op"
            assert c.phase != "reducer_init"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test/codegen/test_loop_forest.py::test_canonical_forest_activation_reduce_with_post_op_has_expected_leaves test/codegen/test_loop_forest.py::test_canonical_forest_activation_reduce_no_post_op_omits_post_op_leaf -v`
Expected: FAIL — current code emits `reducer_init` + F chain + `post_op`.

- [ ] **Step 3: Update `_build_leaves_activation_reduce`**

In `nkigym/src/nkigym/codegen/loop_forest.py:289-310`, replace the function body:

```python
def _build_leaves_activation_reduce(op: ParsedOp, op_graph: OpGraph) -> list[LoopNode | BodyLeaf]:
    """ActivationReduce Pattern 2: ``[<F-chain with reduce_step>, reduce_close, post_op?]``.

    The outer P dim is consumed by ``_wrap_dims``. The F dim is handled
    here: F-block → F-tile → BodyLeaf(reduce_step) writes each tile's
    partial sum into a distinct slot of the op-local ``slot_vec``. After
    the F loop exits, ``reduce_close`` folds the slot vector via a
    single ``nisa.tensor_reduce`` into the op's ``(P, 1)`` output. The
    optional ``post_op`` leaf follows when ``op.op_kwargs['post_op']``
    is set.

    No ``reducer_init`` phase — each ``activation_reduce`` call writes a
    distinct slot, so no prologue memset is needed.
    """
    f_dim = op.axis_map["F"]
    f_role = op.dim_role[f_dim]
    num_f = op_graph.dims[f_dim].num_tiles
    reduce_op = op.op_kwargs["reduce_op"]
    reduce_leaf = BodyLeaf(op_idx=op.idx, phase="reduce_step")
    f_tile = LoopNode(dim_id=f_dim, trip_count=1, role=f_role, children=[reduce_leaf], reduce_op=reduce_op)
    f_block = LoopNode(dim_id=f_dim, trip_count=num_f, role=f_role, children=[f_tile], reduce_op=reduce_op)
    leaves: list[LoopNode | BodyLeaf] = [f_block, BodyLeaf(op_idx=op.idx, phase="reduce_close")]
    if op.op_kwargs.get("post_op") is not None:
        leaves.append(BodyLeaf(op_idx=op.idx, phase="post_op"))
    return leaves
```

- [ ] **Step 4: Update `_phase_dims_activation_reduce`**

In `nkigym/src/nkigym/codegen/loop_forest.py:313-320`, replace:

```python
def _phase_dims_activation_reduce(op: ParsedOp) -> dict[str, tuple[str, ...]]:
    """Return the dims each activation_reduce phase touches.

    reduce_step runs inside F; reduce_close and post_op run outside F.
    """
    p_dim = op.axis_map["P"]
    f_dim = op.axis_map["F"]
    return {"reduce_step": (p_dim, f_dim), "reduce_close": (p_dim,), "post_op": (p_dim,)}
```

(Removed the `"reducer_init"` key.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest test/codegen/test_loop_forest.py::test_canonical_forest_activation_reduce_with_post_op_has_expected_leaves test/codegen/test_loop_forest.py::test_canonical_forest_activation_reduce_no_post_op_omits_post_op_leaf -v`
Expected: PASS.

- [ ] **Step 6: Run the full loop_forest suite**

Run: `pytest test/codegen/test_loop_forest.py -x -q`
Expected: all tests pass. `test_canonical_forest_activation_reduce_f_loops_carry_kwargs_reduce_op` (:316) still works because the new forest has `f_block` at `children[0]` (was `children[1]`); update the test's index:

In `test/codegen/test_loop_forest.py:316-330`, change line 325 from `f_block = p_tile.children[1]` to `f_block = p_tile.children[0]`.

Run: `pytest test/codegen/test_loop_forest.py -x -q`
Expected: all tests pass now.

- [ ] **Step 7: Commit**

```bash
git add nkigym/src/nkigym/codegen/loop_forest.py test/codegen/test_loop_forest.py
git commit -m "$(cat <<'EOF'
nkigym/loop_forest: drop reducer_init, add reduce_close phase

Pattern 2 doesn't need a prologue memset because each activation_reduce
call writes a distinct slot of the slot_vec. The new reduce_close
phase fires one nisa.tensor_reduce after the F loop exits to fold
the slot vector into the op's (P, 1) output.

Tree shape:
  [F-block/F-tile/reduce_step, reduce_close, post_op?]

reduce_close touches only P (sits outside the F loop).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Rewrite `reduce_step` body emitter — write slot, drop merge

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py:488-515`
- Test: `test/codegen/test_render.py`

Context: the new `reduce_step` call targets `sbuf_local_<scratch_id>` as `dst` (discarded output) and `sbuf_local_<slot_id>[0:128, 0, f_slot_idx : f_slot_idx + 1]` as `reduce_res`. No `tmp_red` scratch; no `tensor_tensor` merge.

- [ ] **Step 1: Write the failing test**

Append to `test/codegen/test_render.py`:

```python
def test_render_reduce_step_writes_slot_and_omits_tensor_tensor_merge() -> None:
    """reduce_step emits activation_reduce(reduce_res=slot_vec[:, :, f_idx:f_idx+1]) with no merge.

    Pattern 2 doesn't need tensor_tensor or tmp_red — each call writes
    a distinct slot directly.
    """
    specs = {"x": ((128, 2048), "bfloat16")}
    g = parse_and_resolve(_rms_kernel, specs)
    src = render(g)

    """activation_reduce writes to sbuf_local_1 (slot_vec); dst is sbuf_local_0 (scratch)."""
    assert "nisa.activation_reduce(" in src
    assert "reduce_res=sbuf_local_1" in src
    """No tmp_red allocation, no tensor_tensor merge."""
    assert "tmp_red" not in src
    assert "nisa.tensor_tensor(" not in src
    """The legacy reducer_init memset on sbuf_<reduce_output> must also be gone."""
    assert "nisa.memset(sbuf_m" not in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/codegen/test_render.py::test_render_reduce_step_writes_slot_and_omits_tensor_tensor_merge -v`
Expected: FAIL — current emitter still emits `tmp_red` + `tensor_tensor`.

- [ ] **Step 3: Rewrite `_body_ar_reduce_step`**

In `nkigym/src/nkigym/codegen/render.py:488-515`, replace:

```python
@_register_body("NKIActivationReduce", "reduce_step")
def _body_ar_reduce_step(w, op_graph, op, path_names, path_trips) -> None:
    """Per-F-tile activation_reduce writing into a distinct slot of ``slot_vec``.

    The dst operand goes to the op-local scratch buffer (discarded);
    the reduce_res lands in ``slot_vec[0:p_tile, 0, f_slot:f_slot+1]``
    where ``f_slot`` is the current F-tile ordinal on the path. No
    prologue memset, no cross-tile merge — each call owns its slot.
    """
    src_name = op.operand_names["data"]
    src = op_graph.tensors[src_name]
    p_axis = op.axis_map["P"]
    f_axis = op.axis_map["F"]
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size
    act_op = op.op_kwargs.get("op", "copy")
    reduce_op = op.op_kwargs.get("reduce_op", "add")
    merge = _REDUCE_MERGE_OP[reduce_op]
    p_slot = _slot_expr(path_names, path_trips, p_axis)
    f_slot = _slot_expr(path_names, path_trips, f_axis)
    scratch_name = op.op_local_buffers["scratch"].emitted_name
    slot_name = op.op_local_buffers["slot_vec"].emitted_name
    scratch_slot = f"{scratch_name}[0:{p_tile}, 0, 0:{f_tile}]"
    reduce_res_slot = f"{slot_name}[0:{p_tile}, 0, {f_slot} : {f_slot} + 1]"
    src_expr = _sbuf_tile_slice(_sbuf_name(src_name), src.dim_ids, p_tile, f_tile, path_names, path_trips)
    w.line("nisa.activation_reduce(")
    w.indent()
    w.line(f"dst={scratch_slot},")
    w.line(f"op=nl.{act_op},")
    w.line(f"data={src_expr},")
    w.line(f"reduce_op={merge},")
    w.line(f"reduce_res={reduce_res_slot},")
    w.dedent()
    w.line(")")
```

- [ ] **Step 4: Delete the now-unused `_body_ar_reducer_init`**

In `nkigym/src/nkigym/codegen/render.py:475-486`, delete the whole `_body_ar_reducer_init` function + its `@_register_body("NKIActivationReduce", "reducer_init")` decorator.

- [ ] **Step 5: Run the failing test**

Run: `pytest test/codegen/test_render.py::test_render_reduce_step_writes_slot_and_omits_tensor_tensor_merge -v`
Expected: PASS.

- [ ] **Step 6: Don't run the full suite yet — Task 7 adds `reduce_close` which is still missing**

Running the full render suite now will fail on `test_render_activation_reduce_rmsnorm` and `test_render_forest_activation_reduce_cpu_sim_matches` because `sbuf_m` (the op's output) is never populated — `reduce_step` writes slots, but nothing folds them into `sbuf_m` until `reduce_close` lands in Task 7. Proceed to Task 7 before running the suite.

- [ ] **Step 7: Commit**

```bash
git add nkigym/src/nkigym/codegen/render.py test/codegen/test_render.py
git commit -m "$(cat <<'EOF'
nkigym/render: rewrite reduce_step to write distinct slot

Pattern 2: each activation_reduce call writes a distinct slot of
slot_vec via reduce_res=slot_vec[:, :, f_idx:f_idx+1]. dst goes to
the op-local scratch buffer (discarded). No prologue memset, no
tmp_red, no tensor_tensor merge.

Drops the reducer_init body emitter (phase removed in previous commit).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Add `reduce_close` body emitter — `nisa.tensor_reduce` on slot vector

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py`
- Test: `test/codegen/test_render.py`

- [ ] **Step 1: Write the failing test**

Append to `test/codegen/test_render.py`:

```python
def test_render_reduce_close_emits_tensor_reduce_on_slot_vec() -> None:
    """reduce_close folds slot_vec into the op's (P, 1) output via nisa.tensor_reduce."""
    specs = {"x": ((128, 2048), "bfloat16")}
    g = parse_and_resolve(_rms_kernel, specs)
    src = render(g)

    """tensor_reduce closes the slot vector along its free axis."""
    assert "nisa.tensor_reduce(" in src
    """Must reduce axis=2 of (p_tile, 1, num_f_tiles) shape."""
    assert "axis=2" in src
    """Output is sbuf_m (the op's reduce output), slot is sbuf_local_1."""
    assert "sbuf_m" in src
    assert "sbuf_local_1" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/codegen/test_render.py::test_render_reduce_close_emits_tensor_reduce_on_slot_vec -v`
Expected: FAIL with `ValueError: No body emitter registered for ('NKIActivationReduce', 'reduce_close')`.

- [ ] **Step 3: Add the `reduce_close` body emitter**

In `nkigym/src/nkigym/codegen/render.py`, add a new emitter where `_body_ar_reducer_init` was deleted (right before `_body_ar_reduce_step`):

```python
@_register_body("NKIActivationReduce", "reduce_close")
def _body_ar_reduce_close(w, op_graph, op, path_names, path_trips) -> None:
    """Fold ``slot_vec`` into the op's ``(P, 1)`` output via ``nisa.tensor_reduce``.

    Runs after the F loop exits; the slot vector holds ``num_f_tiles``
    partial sums, one per F-tile. ``axis=2`` reduces the free axis of
    the 3D ``(p_tile, 1, num_f_tiles)`` slot_vec.
    """
    dst_name = op.output_names[0]
    p_axis = op.axis_map["P"]
    p_tile = op_graph.dims[p_axis].tile_size
    f_axis = op.axis_map["F"]
    num_f = op_graph.dims[f_axis].num_tiles
    reduce_op = op.op_kwargs.get("reduce_op", "add")
    merge = _REDUCE_MERGE_OP[reduce_op]
    p_slot = _slot_expr(path_names, path_trips, p_axis)
    dst_slot = f"{_sbuf_name(dst_name)}[0:{p_tile}, {p_slot}, 0:1]"
    slot_name = op.op_local_buffers["slot_vec"].emitted_name
    src_slot = f"{slot_name}[0:{p_tile}, 0, 0:{num_f}]"
    w.line(f"nisa.tensor_reduce({dst_slot}, {merge}, {src_slot}, axis=2)")
```

- [ ] **Step 4: Run the failing test**

Run: `pytest test/codegen/test_render.py::test_render_reduce_close_emits_tensor_reduce_on_slot_vec -v`
Expected: PASS.

- [ ] **Step 5: Run the full render suite (CPU-sim correctness check)**

Run: `pytest test/codegen/test_render.py -x -q`
Expected: all tests pass, including the CPU-sim end-to-end checks (`test_render_activation_reduce_rmsnorm`, `test_render_forest_activation_reduce_cpu_sim_matches`, `test_render_rmsnorm_matmul_end_to_end`, `test_render_forest_rmsnorm_matmul_cpu_sim_matches`). These tests validate numerical correctness against numpy — if they pass, the Pattern 2 lowering is mathematically equivalent to Pattern 3.

If a correctness test fails:
1. Inspect the rendered source: run
   ```python
   from nkigym.codegen.graph import parse_and_resolve
   from nkigym.codegen.render import render
   from test.codegen.test_render import _rms_kernel
   g = parse_and_resolve(_rms_kernel, {"x": ((128, 128), "bfloat16")})
   print(render(g))
   ```
2. Check: does `tensor_reduce` have the right `axis=2`? Does `slot_vec` shape match the `num_f_tiles` seen in `reduce_step`? Is `sbuf_m[:, p_slot, 0:1]` correctly sliced?
3. Common issue: the `post_op` emitter still calls `nisa.activation(data=sbuf_m, ...)` reading from `sbuf_m`. That's correct post-Pattern-2 because `reduce_close` populates `sbuf_m` before `post_op` fires.

Update the existing locked-in assertions:
- `test_render_activation_reduce_rmsnorm` (line 300): `assert "nisa.memset(" in src` — this assertion was for the `reducer_init` memset, which is gone. Change to `assert "nisa.tensor_reduce(" in src`.
- `test_render_forest_activation_reduce_cpu_sim_matches` (line 636): same — `assert "nisa.memset(" in src` → `assert "nisa.tensor_reduce(" in src`.

Run again:
Run: `pytest test/codegen/test_render.py -x -q`
Expected: all tests pass.

- [ ] **Step 6: Run the loop_forest + graph suites for regressions**

Run: `pytest test/codegen/ -x -q`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add nkigym/src/nkigym/codegen/render.py test/codegen/test_render.py
git commit -m "$(cat <<'EOF'
nkigym/render: add reduce_close body emitter

Fires a single nisa.tensor_reduce on slot_vec after the F loop
exits. axis=2 reduces the free axis of (p_tile, 1, num_f_tiles);
output lands in sbuf_<reduce_output>[0:p_tile, p_slot, 0:1]. post_op
consumes sbuf_<reduce_output> unchanged.

Replaces the 2-op-per-F-tile SBUF-merge form with a
(num_f_tiles activation_reduce + 1 tensor_reduce) form — matches
production Pattern 2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: End-to-end verification on the rmsnorm+matmul example

**Files:** none (verification only).

- [ ] **Step 1: Re-render the rmsnorm+matmul example**

Run:
```bash
source ~/venvs/kernel-env/bin/activate
rm -rf /home/ubuntu/cache/rmsnorm_matmul_compile
python examples/rmsnorm_matmul.py 2>&1 | tee /tmp/rmsnorm_matmul_run.log
```
Expected: the run completes (synthesis + initial_codegen + tune batch path). If synthesis is non-deterministic the agent may re-synthesize `f_nkigym.py`; that's fine. The output paths printed at the end should exist.

- [ ] **Step 2: Inspect the new kernel.py**

Run: `diff -u /tmp/kernel_before_pattern2.py /home/ubuntu/cache/rmsnorm_matmul_compile/kernel.py`

Expected changes:
- **Removed**: `nisa.memset(sbuf_rms_inv[...], value=0.0)` inside the P-tile loop.
- **Removed**: `tmp_red = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)` inline allocation.
- **Removed**: `scratch = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)` inline allocation.
- **Removed**: `nisa.tensor_tensor(sbuf_rms_inv[...], sbuf_rms_inv[...], tmp_red[...], op=nl.add)`.
- **Added**: `sbuf_local_0 = nl.ndarray((128, 1, ...), dtype=nl.float32, buffer=nl.sbuf)` at function top (scratch).
- **Added**: `sbuf_local_1 = nl.ndarray((128, 1, 4), dtype=nl.float32, buffer=nl.sbuf)` at function top (slot_vec, 4 = num_f_tiles for F=2048/f_tile=512).
- **Changed**: `nisa.activation_reduce(... reduce_res=tmp_red[...])` → `nisa.activation_reduce(... reduce_res=sbuf_local_1[0:128, 0, f_slot:f_slot+1])`.
- **Added**: `nisa.tensor_reduce(sbuf_rms_inv[...], nl.add, sbuf_local_1[...], axis=2)` after the F loop.
- `nisa.activation(op=nl.rsqrt, scale=..., bias=...)` on `sbuf_rms_inv` unchanged.

- [ ] **Step 3: Check `results.json` for correctness + MFU**

Run: `jq '.[] | {source, cpu_sim_passed, hw_status: .profile_summary.status, mfu: .profile_summary.mfu_estimated_percent, total_time: .profile_summary.total_time}' /home/ubuntu/cache/rmsnorm_matmul_compile/results.json | head -80`

Expected:
- At least half of the tuned kernels pass CPU-sim (some may OOM under sampling — expected and okay).
- The canonical `kernel.py` CPU-sims correctly (non-zero MFU if it reaches HW).
- MFU should be higher than the Pattern-3 baseline (pre-change number captured in `/tmp/kernel_before_pattern2.py`'s corresponding results — if you recorded it; otherwise this step is informational only).

- [ ] **Step 4: Confirm no tests regressed**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test/codegen/ -x -q`
Expected: all tests pass.

- [ ] **Step 5: Verify the whole `nkigym` test set**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test/ -x -q`
Expected: all tests pass. If `test/codegen/test_batch.py` or `test/codegen/test_tune.py` fail on `hash_forest` cycles or forest structural assertions, inspect which — `hash_forest` includes `op_idx` + `phase` in leaf keys, and the phase renaming (`reducer_init` → gone, new `reduce_close`) changes the hash. That's structural churn, not a bug; update affected tests by re-recording the expected hash if needed.

- [ ] **Step 6: Commit verification artifacts (none to commit unless tests were updated)**

If steps 4 or 5 required updating a test, commit those:

```bash
git add test/
git commit -m "$(cat <<'EOF'
test: update structural hash expectations for Pattern 2 forest

reducer_init → reduce_close phase rename changes hash_forest output.
Tests that assert specific hashes get their expected values re-recorded.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Otherwise: no commit needed; verification complete.

---

## Task 9: Update `activation_reduce.py` docstring + learnings hook

**Files:**
- Modify: `nkigym/src/nkigym/ops/activation_reduce.py`

- [ ] **Step 1: Update `NKIActivationReduce` docstring**

In `nkigym/src/nkigym/ops/activation_reduce.py`, update the class docstring (starts at line 64). Replace the existing class docstring with:

```python
class NKIActivationReduce(NKIOp):
    """Fused activation + free-axis reduce, with optional post-op on the closed reduction.

    Math: ``reduce_op(op(data * scale + bias), axis=F)`` with an optional
    ``post_op`` applied once after the F reduction closes — the typical
    rmsnorm shape is ``post_op(reduce_op(op(data)*scale+bias, axis=F))``.

    Lowering (Pattern 2): one ``nisa.activation_reduce`` per F-tile
    writes to a distinct slot of the op-local ``slot_vec`` buffer;
    after the F loop exits, one ``nisa.tensor_reduce(axis=2)`` folds
    ``slot_vec`` into the op's ``(P, 1)`` output; ``post_op`` (if set)
    fires next via ``nisa.activation(op=<post_op>, scale, bias)``.

    The scratch buffer (``dst`` of each ``activation_reduce``) is
    op-local and discarded — the op-local allocation rule keeps its
    footprint at ``(p_tile, 1, f_tile)`` regardless of ``num_p_tiles``
    or ``num_f_tiles``.

    Future work: a hoist transform that pulls ``activation_reduce`` out
    of the F loop, combined with DCE on the closing ``tensor_reduce``
    (trivial when ``num_f_tiles == 1``), will reach Pattern 1 (one
    full-F ``activation_reduce`` call, no slot vector) monotonically.
    """
```

- [ ] **Step 2: Run tests (docstring change is no-op functionally)**

Run: `pytest test/codegen/ -x -q`
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add nkigym/src/nkigym/ops/activation_reduce.py
git commit -m "$(cat <<'EOF'
nkigym/ops: update NKIActivationReduce docstring for Pattern 2

Documents the new lowering (per-F-tile activation_reduce writing
distinct slot + closing tensor_reduce + optional post_op) and the
future-work hoist + DCE path to Pattern 1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Out of scope

The following are explicitly **not** handled by this plan, even though they relate:

- **Pattern 1 (single-call over full F)** — reachable via a future `Hoist` rewrite transform on `activation_reduce` across the F loop + DCE on the closing `tensor_reduce` once `num_f_tiles` collapses to 1. Ships as a separate plan.
- **Buffer shrinkage under fusion** — a future loop-fusion transform will reduce op-local buffer shapes when adjacent nests get merged. Relies on a final DCE sweep. Not handled here.
- **`NKIActivation` single-op register-chained path** (`reduce_cmd=reduce`) — production avoids it for maintainability; we follow suit.
- **Large-F workloads beyond single-call SBUF capacity** — Pattern 2 already handles these; no further work needed for them.
- **Online-softmax-style consumers of per-tile partials** — those will need explicit support for "slot vector is also an output" (not just an op-local scratch). Different plan.

## Verification summary

After all tasks complete:

1. All tests in `test/codegen/` pass.
2. `examples/rmsnorm_matmul.py` runs end-to-end; `results.json` shows CPU-sim passes for the canonical kernel + at least some tuned variants.
3. The rendered `kernel.py` drops per-P-tile `memset` + per-F-tile `tensor_tensor` + per-F-tile `tmp_red` allocation; adds `sbuf_local_0` / `sbuf_local_1` at function top and a closing `nisa.tensor_reduce`.
4. ISA op count per P-tile iteration drops from 34 (1 memset + 16*(activation_reduce + tensor_tensor) + 1 activation) to 19 (16 activation_reduce + 1 tensor_reduce + 1 activation) — Pattern 2 doctrine.
