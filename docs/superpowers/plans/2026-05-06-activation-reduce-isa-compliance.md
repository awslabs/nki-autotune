# NKIActivationReduce ISA Compliance Cleanup Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip invented kwargs from `NKIActivationReduce` so its Python constructor kwargs mirror the real `nisa.activation_reduce` ISA signature 1:1. Drop `post_op` (doesn't exist in ISA) and drop `scale`/`bias` (our semantics don't match ISA's — ours apply to the closed reduction, ISA's apply per-element pre-activation). After this plan, `NKIActivationReduce(op, reduce_op)` is the only valid signature; RMSNorm-style closures become an explicit second `NKIActivation(op="rsqrt", scale=..., bias=...)` call on the reduction output.

**Architecture:** Every `NKIOp` subclass's constructor kwargs must mirror the valid kwargs of its backing `nisa.*` ISA call. `NKIActivationReduce` is the only op in-scope for this plan; other ops (`NKIActivation`, `NKITensorScalar`, `NKIMatmul`, etc.) are **out of scope** — audited on a touch-when-touched basis. The cleanup removes the `post_op` phase from the forest builder, the `_body_ar_post_op` emitter from the renderer, and updates the synthesis prompt to teach the agent to emit two separate DSL calls for the RMSNorm fused pattern.

**Tech Stack:** Python 3.12, `nki`, `nki.isa`, `nki.language`, `ast`, `pytest`. Runtime: `source ~/venvs/kernel-env/bin/activate`. Codebase: `/home/ubuntu/nki-autotune`.

---

## Reference map (read these before starting)

- Current op: `nkigym/src/nkigym/ops/activation_reduce.py` — class kwargs, `_run`, helper functions.
- Reference ISA signature: `nki.isa.activation_reduce(dst, op, data, reduce_op, reduce_res, bias=None, scale=1.0, name=None)` at `/home/ubuntu/venvs/kernel-env/lib/python3.12/site-packages/nki/isa/activation.py:164`. Note: `bias` and `scale` apply per-element *pre-activation*, not to the closed reduction.
- Reference `NKIActivation` (already ISA-compliant): `nkigym/src/nkigym/ops/activation.py` — `NKIActivation(op, scale, bias)` mirrors `nisa.activation(dst, op, data, bias, scale, reduce_op, reduce_res, reduce_cmd, name)` (we expose the subset we use).
- Forest builder: `nkigym/src/nkigym/codegen/loop_forest.py:289-320, 332`.
- Renderer: `nkigym/src/nkigym/codegen/render.py` — `_body_ar_post_op` at lines 538-549.
- Synthesis prompt: `nkigym/src/nkigym/synthesis/numpy_to_nkigym.py:80-91`.
- Example: `examples/rmsnorm_matmul.py` (the numpy reference — DSL form gets re-synthesized; no direct DSL edit in this repo since `f_nkigym.py` lives under `/home/ubuntu/cache/...`).
- Test files with `post_op`/`scale`/`bias` usage on `NKIActivationReduce`:
  - `test/codegen/test_compile.py:40`
  - `test/codegen/test_tune.py:26`
  - `test/codegen/test_render.py:47` (that one's on `NKIActivation` — keep), :64, :76, :267, :293
  - `test/codegen/test_loop_forest.py:97-122, 213-256, 316-327` (multiple kernel fixtures use `post_op`)
  - `test/codegen/test_graph.py:69, 82, 198, 233`
- Code style: `/home/ubuntu/.claude/rules/code_style.md` — triple-quoted block comments (no `#`), Google/NumPy docstrings, modern type hints.
- Pre-commit hook runs `autoflake`/`isort`/`black` — don't bypass; fix style issues at source.

## File structure

**Modify:**
- `nkigym/src/nkigym/ops/activation_reduce.py` — drop `post_op`/`scale`/`bias` from `_run`; update class docstring; remove `_IDENTITY`/`_MERGE_FNS` if they're only used by the removed post_op path; verify `activation_block`/`activation_reduce_block` helpers are still needed or can be deleted.
- `nkigym/src/nkigym/codegen/loop_forest.py` — drop the `post_op` leaf from `_build_leaves_activation_reduce` + the `post_op` entry from `_phase_dims_activation_reduce`.
- `nkigym/src/nkigym/codegen/render.py` — delete `_body_ar_post_op` function + its `@_register_body` decorator.
- `nkigym/src/nkigym/synthesis/numpy_to_nkigym.py` — strip `post_op`/`scale`/`bias` from the `NKIActivationReduce` row in the op table; drop `post_op` from the op-arg vocabulary line; rewrite the fused-reduce pattern guidance to use two DSL calls.
- `test/codegen/test_compile.py` — update the one fixture.
- `test/codegen/test_tune.py` — update the one fixture.
- `test/codegen/test_render.py` — update `_rms_kernel`, `_rmsnorm_matmul` fixtures and all assertions keyed on the old kwargs / phase.
- `test/codegen/test_loop_forest.py` — update fixtures + all `post_op`-phase assertions.
- `test/codegen/test_graph.py` — update fixtures + any kwarg assertions.

**No new files.**

---

## Task 0: Baseline + worktree prep

**Files:** none (verification only).

- [ ] **Step 1: Activate venv + sanity-check**

Run: `source ~/venvs/kernel-env/bin/activate && python -c "import nki, nki.isa; from nki import simulate; print('ok')"`
Expected: `ok`.

- [ ] **Step 2: Confirm baseline test suite passes**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test/codegen/ -x -q`
Expected: 156 passed. If any fail, stop — the cleanup assumes green baseline.

- [ ] **Step 3: Capture current rendered kernel for after-diff**

Run:
```bash
cp /home/ubuntu/cache/rmsnorm_matmul_compile/kernel.py /tmp/kernel_before_isa_cleanup.py
```
Expected: file copied. After the cleanup, the rendered kernel should keep the `nisa.activation_reduce` call (with no `post_op`/`scale`/`bias` — already the case after Pattern 2) plus one standalone `nisa.activation(op=nl.rsqrt, scale=..., bias=...)` call, now originating from an explicit `NKIActivation` op in the DSL rather than a `post_op` kwarg on `NKIActivationReduce`.

- [ ] **Step 4: Confirm clean working tree**

Run: `git status`
Expected: clean on `dev_1` (or whatever branch). If dirty, commit/stash separately.

---

## Task 1: Simplify `NKIActivationReduce._run` and strip class-level support for removed kwargs

**Files:**
- Modify: `nkigym/src/nkigym/ops/activation_reduce.py`
- Test: `test/codegen/test_graph.py` (new test)

Context: `NKIActivationReduce._run` currently handles `scale`/`bias`/`post_op` in CPU simulation. Strip them so passing any of these kwargs raises. The class now mirrors the valid ISA subset: `{op, reduce_op}`.

- [ ] **Step 1: Write the failing test**

Append to `test/codegen/test_graph.py`:

```python
def test_activation_reduce_rejects_removed_kwargs() -> None:
    """NKIActivationReduce mirrors nisa.activation_reduce kwargs; post_op/scale/bias removed."""
    import numpy as np

    from nkigym.ops.activation_reduce import NKIActivationReduce

    xs = np.ones((128, 2048), dtype=np.float32).view(type=np.ndarray)

    """Valid call: only op + reduce_op."""
    NKIActivationReduce(op="square", reduce_op="add")(data=xs)

    """Removed kwargs must raise TypeError with a clear message."""
    import pytest

    with pytest.raises(TypeError):
        NKIActivationReduce(op="square", reduce_op="add", post_op="rsqrt")(data=xs)
    with pytest.raises(TypeError):
        NKIActivationReduce(op="square", reduce_op="add", scale=0.5)(data=xs)
    with pytest.raises(TypeError):
        NKIActivationReduce(op="square", reduce_op="add", bias=1e-6)(data=xs)
```

Note: the `_run` method in `NKIOp` base class currently accepts any kwargs via `**kwargs`. To make removed-kwargs-raise work, we'll validate inside `_run` explicitly (Step 3).

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test/codegen/test_graph.py::test_activation_reduce_rejects_removed_kwargs -v`
Expected: FAIL — currently the removed kwargs are accepted silently.

- [ ] **Step 3: Rewrite `NKIActivationReduce._run` + strip helpers**

In `nkigym/src/nkigym/ops/activation_reduce.py`, replace the `_run` method (currently lines 107-128) with:

```python
    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: ``reduce_op(op(data), axis=F)``.

        Mirrors the valid kwarg subset of ``nisa.activation_reduce``: only
        ``op`` and ``reduce_op`` are accepted. For fused closures like
        rmsnorm (``rsqrt(sum(x²)/K + eps)``), the DSL must spell out the
        post-reduction activation as a separate ``NKIActivation`` call on
        the reduction output.

        Raises:
            TypeError: any kwarg besides ``data`` / ``op`` / ``reduce_op``
                is supplied.
        """
        allowed = {"data", "op", "reduce_op"}
        extra = set(kwargs) - allowed
        if extra:
            raise TypeError(
                f"NKIActivationReduce received unexpected kwargs: {sorted(extra)}. "
                f"Only {sorted(allowed)} are supported; use a separate NKIActivation "
                f"for post-reduction scale/bias/op."
            )
        data: np.ndarray = kwargs["data"]
        op_name: str = kwargs["op"]
        reduce_op: str = kwargs["reduce_op"]
        data_f32 = data.astype(np.float32)
        activated = _ACT_FNS[op_name](data_f32)
        reduced = _RED_FNS[reduce_op](activated, axis=1).astype(np.float32)
        return reduced
```

Also update the class docstring (lines 64-85) to drop mentions of `post_op`/`scale`/`bias`:

```python
class NKIActivationReduce(NKIOp):
    """Fused activation + free-axis reduce — mirrors ``nisa.activation_reduce``.

    Math: ``reduce_op(op(data), axis=F)``. Output is the ``(P,)`` per-row
    reduction vector. The fully activated ``(P, F)`` tile is an internal
    byproduct the gadget discards.

    Kwargs mirror the valid subset of ``nisa.activation_reduce``:

    * ``op``: activation applied per-element before the reduce.
    * ``reduce_op``: reduction operator along the free axis.

    Fused closures (e.g. rmsnorm's ``rsqrt(sum(x²)·scale + bias)``) must
    be spelled out in the DSL as a separate ``NKIActivation(op="rsqrt",
    scale=..., bias=...)`` call on the reduction output — not as
    ``post_op``/``scale``/``bias`` kwargs on this op.

    Lowering (Pattern 2): one ``nisa.activation_reduce`` per F-tile
    writes to a distinct slot of the op-local ``slot_vec`` buffer;
    after the F loop exits, one ``nisa.tensor_reduce(axis=2)`` folds
    ``slot_vec`` into the op's ``(P, 1)`` output.

    Future work: a hoist transform that pulls ``activation_reduce`` out
    of the F loop, combined with DCE on the closing ``tensor_reduce``
    (trivial when ``num_f_tiles == 1``), will reach Pattern 1 (one
    full-F ``activation_reduce`` call, no slot vector) monotonically.
    """
```

The ClassVars stay unchanged (OPERAND_AXES, OUTPUT_AXES, OUTPUT_DTYPES, AXIS_ROLES, TILE_LIMITS, OP_LOCAL_BUFFERS).

Delete unused helpers at the module level now that `post_op` is gone:
- `_IDENTITY` (lines 58-61) — used only by the removed `reducer_init` path and dropped `post_op` reducer-identity logic. Confirm via grep before deletion.
- `_MERGE_FNS` (lines 52-56) — used only by the removed cross-iteration merge. Confirm via grep.
- `activation_block` function (lines 150-165) — was only invoked by the legacy `post_op` code path. Confirm via grep across the whole repo.
- `activation_reduce_block` function (lines 118-147) — same; was only invoked by legacy codegen.
- `nl_op` / `nl_reduce_op` helpers (lines 168-175) — grep whole repo; delete if unused.

Grep command to verify before deletion:

```bash
source ~/venvs/kernel-env/bin/activate
for sym in _IDENTITY _MERGE_FNS activation_block activation_reduce_block nl_op nl_reduce_op; do
  echo "=== $sym ==="
  grep -rn "$sym" nkigym/ test/ examples/ autotune/ 2>/dev/null | grep -v "__pycache__\|\.pyc\|activation_reduce.py:"
done
```

Delete only symbols with **zero** external references. If any still have a use site, leave them and document why.

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test/codegen/test_graph.py::test_activation_reduce_rejects_removed_kwargs -v`
Expected: PASS.

- [ ] **Step 5: Run the full graph suite to catch regressions**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test/codegen/test_graph.py -x -q`
Expected: multiple existing tests fail. The failing tests use `post_op`/`scale`/`bias` on `NKIActivationReduce`. These get updated in Task 5. Note the failures but don't fix them here.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/ops/activation_reduce.py test/codegen/test_graph.py
git commit -m "$(cat <<'EOF'
nkigym/ops: strip invented kwargs from NKIActivationReduce

NKIOp classes must mirror the valid kwarg subset of their backing ISA
call. nisa.activation_reduce has no post_op field, and its scale/bias
apply per-element pre-activation — not to the closed reduction as our
op previously interpreted them. Drop post_op/scale/bias entirely;
_run now only accepts {data, op, reduce_op}. Fused closures like
rmsnorm's rsqrt(sum(x²)·scale + bias) must be split into separate
NKIActivationReduce + NKIActivation DSL calls.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Drop `post_op` phase from the forest builder

**Files:**
- Modify: `nkigym/src/nkigym/codegen/loop_forest.py`
- Test: `test/codegen/test_loop_forest.py`

Context: `_build_leaves_activation_reduce` currently conditionally appends a `post_op` leaf when `op.op_kwargs.get("post_op") is not None`. Since `post_op` is now never in `op_kwargs`, the branch is dead. Remove it. Also remove `"post_op": (p_dim,)` from `_phase_dims_activation_reduce`.

- [ ] **Step 1: Update test assertions to expect the simpler forest shape**

In `test/codegen/test_loop_forest.py`, replace `test_canonical_forest_activation_reduce_with_post_op_has_expected_leaves` (lines 213-236) and `test_canonical_forest_activation_reduce_no_post_op_omits_post_op_leaf` (lines 239-256) with a single test:

```python
def test_canonical_forest_activation_reduce_has_expected_leaves() -> None:
    """NKIActivationReduce emits F chain + reduce_close. No reducer_init, no post_op.

    Pattern 2: each activation_reduce call writes a distinct slot of
    slot_vec; reduce_close folds the slots via nisa.tensor_reduce. The
    post_op phase was removed — fused closures are now two separate DSL
    calls in f_nkigym.
    """
    from nkigym.codegen.loop_forest import build_canonical_forest

    g = _parse(_rms_kernel_with_post_op, _RMS_SPECS)
    forest = build_canonical_forest(g)
    ar_idx = next(i for i, op in enumerate(g.ops) if op.op_cls.__name__ == "NKIActivationReduce")
    tree = forest[ar_idx]
    p_tile = tree.children[0]
    children = p_tile.children
    """Two children: F chain, reduce_close."""
    assert len(children) == 2
    f_block = children[0]
    assert isinstance(f_block, LoopNode)
    f_tile = f_block.children[0]
    reduce_leaf = f_tile.children[0]
    assert isinstance(reduce_leaf, BodyLeaf) and reduce_leaf.phase == "reduce_step"
    assert isinstance(children[1], BodyLeaf) and children[1].phase == "reduce_close"
    for c in children:
        if isinstance(c, BodyLeaf):
            assert c.phase != "post_op"
            assert c.phase != "reducer_init"
```

Note: `_rms_kernel_with_post_op` and `_rms_kernel_without_post_op` fixtures at lines 97-110 both declare `post_op="rsqrt"` / no post_op. These kernels now won't parse because Task 1 rejects the kwarg. Task 5 rewrites them; for Task 2, temporarily rename `_rms_kernel_with_post_op` to `_rms_kernel` at the fixture site and drop the `post_op="rsqrt"` kwarg to make it parse-able:

```python
@nkigym_kernel
def _rms_kernel(x):
    """Test kernel: activation_reduce with no extra kwargs."""
    xs = NKILoad()(data=x)
    m = NKIActivationReduce(op="square", reduce_op="add")(data=xs)
    out = NKIStore()(data=m)
    return out
```

Delete `_rms_kernel_without_post_op` (no longer distinct from the renamed fixture) at lines 106-112.

Update every reference to `_rms_kernel_with_post_op` / `_rms_kernel_without_post_op` in the test module to use `_rms_kernel`. Specifically:
- Line 222: `g = _parse(_rms_kernel_with_post_op, _RMS_SPECS)` → `g = _parse(_rms_kernel, _RMS_SPECS)`
- Line 243: `g = _parse(_rms_kernel_without_post_op, _RMS_SPECS)` (delete that test entirely per above)
- Line 320: `g = _parse(_rms_kernel_with_post_op, _RMS_SPECS)` → `g = _parse(_rms_kernel, _RMS_SPECS)`

- [ ] **Step 2: Run tests to verify they fail**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test/codegen/test_loop_forest.py -q`
Expected: multiple failures — the forest still emits the `post_op` leaf; the test fixtures may fail to parse until Task 1's changes are in place. If Task 1 is already committed (it is), fixtures with `post_op="rsqrt"` in their decorator calls will raise TypeError on `_parse`.

- [ ] **Step 3: Update `_build_leaves_activation_reduce`**

In `nkigym/src/nkigym/codegen/loop_forest.py`, replace the function (lines 289-310) with:

```python
def _build_leaves_activation_reduce(op: ParsedOp, op_graph: OpGraph) -> list[LoopNode | BodyLeaf]:
    """ActivationReduce Pattern 2: ``[<F-chain with reduce_step>, reduce_close]``.

    The outer P dim is consumed by ``_wrap_dims``. The F dim is handled
    here: F-block → F-tile → BodyLeaf(reduce_step) writes each tile's
    partial sum into a distinct slot of the op-local ``slot_vec``. After
    the F loop exits, ``reduce_close`` folds the slot vector via a
    single ``nisa.tensor_reduce`` into the op's ``(P, 1)`` output.

    No ``reducer_init`` phase — each ``activation_reduce`` call writes a
    distinct slot. No ``post_op`` phase — fused closures are spelled out
    as a separate ``NKIActivation`` op in the DSL.
    """
    f_dim = op.axis_map["F"]
    f_role = op.dim_role[f_dim]
    num_f = op_graph.dims[f_dim].num_tiles
    reduce_op = op.op_kwargs["reduce_op"]
    reduce_leaf = BodyLeaf(op_idx=op.idx, phase="reduce_step")
    f_tile = LoopNode(dim_id=f_dim, trip_count=1, role=f_role, children=[reduce_leaf], reduce_op=reduce_op)
    f_block = LoopNode(dim_id=f_dim, trip_count=num_f, role=f_role, children=[f_tile], reduce_op=reduce_op)
    return [f_block, BodyLeaf(op_idx=op.idx, phase="reduce_close")]
```

Update `_phase_dims_activation_reduce` (lines 313-322) to drop the `post_op` entry:

```python
def _phase_dims_activation_reduce(op: ParsedOp) -> dict[str, tuple[str, ...]]:
    """Return the dims each activation_reduce phase touches.

    reduce_step runs inside F; reduce_close runs outside F.
    """
    p_dim = op.axis_map["P"]
    f_dim = op.axis_map["F"]
    return {"reduce_step": (p_dim, f_dim), "reduce_close": (p_dim,)}
```

- [ ] **Step 4: Run the tests**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test/codegen/test_loop_forest.py -x -q`
Expected: all loop_forest tests pass.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/loop_forest.py test/codegen/test_loop_forest.py
git commit -m "$(cat <<'EOF'
nkigym/loop_forest: drop post_op phase from activation_reduce

NKIActivationReduce no longer has a post_op kwarg (see prior commit).
The forest builder's post_op leaf is unreachable; drop it and drop
the post_op entry from _phase_dims_activation_reduce. Tree shape
collapses to [F-block/F-tile/reduce_step, reduce_close].

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Delete `_body_ar_post_op` body emitter

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py`

- [ ] **Step 1: Delete `_body_ar_post_op`**

In `nkigym/src/nkigym/codegen/render.py`, delete the entire `_body_ar_post_op` function + its `@_register_body("NKIActivationReduce", "post_op")` decorator. Currently at lines 538-549.

- [ ] **Step 2: Run the render suite**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test/codegen/test_render.py -x -q`
Expected: some tests fail — their fixtures still use `post_op="rsqrt"` on `NKIActivationReduce`. Those fixtures get updated in Task 5. Tests that don't use `post_op` pass.

- [ ] **Step 3: Commit**

```bash
git add nkigym/src/nkigym/codegen/render.py
git commit -m "$(cat <<'EOF'
nkigym/render: delete _body_ar_post_op emitter

The post_op phase was removed from the forest builder (prior commit);
its body emitter is now dead code. Delete both the function and its
@_register_body decorator.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Update synthesis prompt

**Files:**
- Modify: `nkigym/src/nkigym/synthesis/numpy_to_nkigym.py`

Context: the synthesis prompt teaches the agent to emit `NKIActivationReduce(op, reduce_op, post_op, scale, bias)` for fused reduce-then-activation patterns. That guidance now produces invalid DSL; the agent must emit two calls instead.

- [ ] **Step 1: Update the op table row**

In `nkigym/src/nkigym/synthesis/numpy_to_nkigym.py`, replace the `NKIActivationReduce` row at line 80:

```
| `NKIActivationReduce` | `nkigym.ops.activation_reduce` | `data:(P,F)` | `output:(P,)` | `NKIActivationReduce(op=..., reduce_op=...)(data=X)` — reduces along F; output is the per-row reduction vector |
```

- [ ] **Step 2: Drop `post_op` from the op-arg vocabulary line**

In `nkigym/src/nkigym/synthesis/numpy_to_nkigym.py`, replace line 84:

```
Op-arg vocabulary: `op` ∈ `{square, exp, copy, reciprocal, tanh, rsqrt, sqrt}`; `reduce_op` ∈ `{add, max}`; `NKITensorScalar.op` ∈ `{multiply, add, subtract}`; `NKIActivation.op` ∈ `{square, exp, copy, reciprocal, tanh, rsqrt, sqrt}`; `NKIActivation.scale` / `NKIActivation.bias` apply per-element pre-activation.
```

- [ ] **Step 3: Rewrite the fused-reduce pattern guidance**

In `nkigym/src/nkigym/synthesis/numpy_to_nkigym.py`, replace the pattern at line 91:

```
   - Fused reduce-then-activation (e.g. rmsnorm's `rsqrt(sum(x²)/F + eps)`): split into two DSL calls. Emit `NKIActivationReduce(op=<act>, reduce_op=<red>)(data=X)` to get the raw reduction; then feed that into `NKIActivation(op=<post>, scale=<scalar>, bias=<scalar>)(data=reduced)` to apply the post-reduction activation with its affine scale/bias. `scale` on `NKIActivation` multiplies every element of its input; for closures where you want `f(reduced/F + eps)`, use `scale=1/F` and `bias=eps`.
```

- [ ] **Step 4: Update common-failure-modes hint**

In `nkigym/src/nkigym/synthesis/numpy_to_nkigym.py` at line 101, update the bullet about `scale`/`bias`:

```
- Common failure modes: wrong matmul transpose orientation, post-reduction activation squeezed into `NKIActivationReduce` (it has no `post_op` / `scale` / `bias` — emit a separate `NKIActivation` instead), wrong axis in a reduction, missing step in the DAG, output-shape mismatch (usually a missing or extra transpose).
```

- [ ] **Step 5: Run any synthesis tests that exist**

Run: `source ~/venvs/kernel-env/bin/activate && pytest -k synthesis -q 2>&1 | tail -10`
Expected: either no synthesis tests run (prompt is text-only, no unit test), or they pass unchanged.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/synthesis/numpy_to_nkigym.py
git commit -m "$(cat <<'EOF'
nkigym/synthesis: teach two-call form for fused reduce-then-activation

NKIActivationReduce no longer accepts post_op/scale/bias. Update the
synthesis prompt so the agent emits two separate DSL calls for the
rmsnorm-style closure: NKIActivationReduce for the raw reduction,
then NKIActivation with post_op/scale/bias on the reduction output.
Explicitly flag the single-call form in the common-failure-modes
hint.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Update remaining test fixtures and assertions

**Files:**
- Modify: `test/codegen/test_render.py`
- Modify: `test/codegen/test_graph.py`
- Modify: `test/codegen/test_compile.py`
- Modify: `test/codegen/test_tune.py`

Context: several test kernel fixtures invoke `NKIActivationReduce(op, reduce_op, post_op, scale, bias)` which now raises TypeError. Rewrite each to the two-call DSL form.

- [ ] **Step 1: Update `test/codegen/test_render.py` fixtures**

Replace the `_rms_kernel` fixture at lines 61-66:

```python
@nkigym_kernel
def _rms_kernel(x):
    xs = NKILoad()(data=x)
    sum_sq = NKIActivationReduce(op="square", reduce_op="add")(data=xs)
    m = NKIActivation(op="rsqrt", scale=1 / 128, bias=1e-6)(data=sum_sq)
    out = NKIStore()(data=m)
    return out
```

Replace the `_rmsnorm_matmul` fixture at lines 72-81:

```python
@nkigym_kernel
def _rmsnorm_matmul(lhs, rhs):
    lhs_sbuf = NKILoad()(data=lhs)
    rhs_sbuf = NKILoad()(data=rhs)
    sum_sq = NKIActivationReduce(op="square", reduce_op="add")(data=lhs_sbuf)
    rms_inv = NKIActivation(op="rsqrt", scale=1 / 256, bias=EPS)(data=sum_sq)
    lhs_rms = NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv)
    lhs_T = NKITranspose()(data=lhs_rms)
    prod = NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out
```

Update `test_render_activation_reduce_rmsnorm` (line 292) — its docstring mentions the old phase. Update docstring to reflect the two-op form; the assertions on `nisa.activation_reduce`, `nisa.activation`, `nisa.tensor_reduce`, `op=nl.rsqrt` all remain valid because the DSL now emits two ops that between them produce those same ISA calls.

- [ ] **Step 2: Update `test/codegen/test_graph.py` fixtures**

Replace the fixture at line 69 (the `_rms_kernel` local inside `test_parse_and_resolve_activation_reduce_kwargs_are_captured`). Decide the test's intent: if it's checking kwargs-are-captured, the point now is that only `op`/`reduce_op` land in `op_kwargs`. Replace the assertion at line 82 (`kwargs["post_op"] == "rsqrt"`) with:

```python
    assert kwargs["op"] == "square"
    assert kwargs["reduce_op"] == "add"
    assert "post_op" not in kwargs
    assert "scale" not in kwargs
    assert "bias" not in kwargs
```

Replace the fixtures at lines 198, 233 (`test_parse_and_resolve_registers_op_local_derived_dims_for_activation_reduce`, `test_parsed_op_resolves_op_local_buffer_names_for_activation_reduce`). Both use `post_op="rsqrt", scale=1/2048, bias=1e-6`. Drop those kwargs from the `NKIActivationReduce(...)` call. The tests validate buffer resolution, not kwargs.

- [ ] **Step 3: Update `test/codegen/test_compile.py`**

Replace the fixture at line 40 with the two-call form. The test's intent is that `nkigym_compile` drives the full synthesis→codegen→tune loop; adjust the math to match.

- [ ] **Step 4: Update `test/codegen/test_tune.py`**

Replace the fixture at line 26 with the two-call form.

- [ ] **Step 5: Run the full test suite**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test/codegen/ -x -q`
Expected: all tests pass (156 or thereabouts — one test dropped in Task 2).

- [ ] **Step 6: Commit**

```bash
git add test/codegen/
git commit -m "$(cat <<'EOF'
test: update fixtures for NKIActivationReduce ISA compliance

NKIActivationReduce no longer accepts post_op/scale/bias. Rewrite
every test fixture that used the fused form into the two-call DSL:
NKIActivationReduce for the raw reduction, then NKIActivation for
the post-reduction activation with scale/bias. Test assertions that
validated captured kwargs are updated to reflect the smaller kwarg
surface.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: End-to-end verification

**Files:** none (verification only).

- [ ] **Step 1: Re-run the rmsnorm_matmul example**

Run:
```bash
source ~/venvs/kernel-env/bin/activate
rm -rf /home/ubuntu/cache/rmsnorm_matmul_compile
python examples/rmsnorm_matmul.py 2>&1 | tee /tmp/rmsnorm_matmul_isa_clean.log
```
Expected: the run completes; synthesis re-runs and produces a `f_nkigym.py` that uses two calls (`NKIActivationReduce` + `NKIActivation`) for the RMSNorm closure. If synthesis fails because the agent kept the old one-call form, inspect the agent transcript — the synthesis prompt update (Task 4) should be teaching the new form. If the agent disregards it, flag and stop — that's a prompt-quality issue worth a follow-up.

- [ ] **Step 2: Inspect the new `f_nkigym.py`**

Run: `cat /home/ubuntu/cache/rmsnorm_matmul_compile/f_nkigym.py`
Expected: contains `NKIActivationReduce(op="square", reduce_op="add")(data=lhs_sbuf)` bound to some intermediate, then `NKIActivation(op="rsqrt", scale=..., bias=...)(data=<intermediate>)` bound to `rms_inv`. No `post_op`/`scale`/`bias` on `NKIActivationReduce`.

- [ ] **Step 3: Inspect the new rendered `kernel.py`**

Run: `diff -u /tmp/kernel_before_isa_cleanup.py /home/ubuntu/cache/rmsnorm_matmul_compile/kernel.py`

Expected:
- **Same**: The `nisa.activation_reduce` call at the reduce_step position — still with only `{dst, op, data, reduce_op, reduce_res}` kwargs.
- **Same**: The closing `nisa.tensor_reduce` after the F loop exits.
- **Changed**: An SBUF buffer name change — the reduction output tensor now has a name from the DSL intermediate (e.g. `sbuf_sum_sq` instead of `sbuf_rms_inv`), since the reduction output in the two-call DSL is no longer the same tensor as the rsqrt output.
- **Added**: A separate loop nest that renders `NKIActivation(op="rsqrt", scale=..., bias=...)` — emits `nisa.activation(dst=sbuf_rms_inv[...], op=nl.rsqrt, data=sbuf_sum_sq[...], scale=..., bias=...)`. Note: previously this `nisa.activation(rsqrt)` call lived inline inside the reduce op's P-tile loop as the post_op phase. Now it's its own op, rendered in its own nest.

The exact structural diff depends on whether the renderer fuses adjacent P-tile loops (it doesn't, in the current eager codegen) — expect the rsqrt call to land in a separate `for i_d0_0 ...` loop after the reduce nest. That's acceptable.

- [ ] **Step 4: Check `results.json` for CPU-sim success**

Run: `source ~/venvs/kernel-env/bin/activate && python -c "
import json
with open('/home/ubuntu/cache/rmsnorm_matmul_compile/results.json') as f:
    r = json.load(f)
print('cpu_sim passed:', r['metrics']['passed_cpu_sim'], '/', r['metadata']['num_kernels'])
"`
Expected: CPU-sim pass count equal to num_kernels (16 by default). The math is unchanged, only the DSL spelling; numerical output must match the pre-cleanup run.

- [ ] **Step 5: Confirm no tests regressed**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test/codegen/ -x -q`
Expected: all tests pass.

- [ ] **Step 6: No commit needed** unless an issue surfaced.

---

## Out of scope

- **Audit of other NKIOp classes for ISA mirror compliance** — `NKITensorScalar`, `NKIMatmul`, `NKITranspose`, `NKIDMATranspose`, `NKILoad`, `NKIStore`. These get audited on a touch-when-touched basis. A future sweep can lock them all to ISA.
- **Pattern 1 (single full-F `activation_reduce` call)** — reached via a future hoist + DCE transform on Pattern 2's forest shape.
- **Buffer shrinkage under fusion** — relies on a final DCE sweep; not handled here.

## Verification summary

After all tasks complete:

1. `NKIActivationReduce(op, reduce_op)` is the only valid signature; any other kwarg raises TypeError.
2. `_body_ar_post_op` is gone; the `post_op` phase no longer exists anywhere.
3. Synthesis prompt teaches the two-call form and flags the old one-call form as a common failure.
4. Re-running `examples/rmsnorm_matmul.py` produces a new `f_nkigym.py` using two DSL calls and a rendered `kernel.py` whose `nisa.activation_reduce` line has exactly 5 ISA-valid kwargs and whose `nisa.activation(op=nl.rsqrt, ...)` closure lives in its own loop nest.
5. All codegen tests pass.
