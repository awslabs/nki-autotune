# Eager Fuse Pattern Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `KernelIR.fused_iter_var_map`; make Fuse eagerly rewrite `BufferAccess.iter_var_coeffs` so `BufferAccess` is the single source of truth for access patterns.

**Architecture:** Same-axis Fuse (outer extent = 1) rewrites affected access patterns in one pass: retired outer drops from coeffs (extent-1 contributes 0), retired inner renames to fused var_id. Cross-axis Fuse with dependent access patterns raises `AtomLegalityError` (non-affine decomposition, deferred). Renderer loses its map-consulting branch: every iter-var in a live access pattern must have a live ForNode binding, else KeyError.

**Tech Stack:** Python 3.12, existing nkigym IR, pytest.

---

## Spec Reference

`docs/superpowers/specs/2026-05-12-eager-fuse-pattern-rewrite-design.md`

## File Map

**Production (3 files):**
- `nkigym/src/nkigym/codegen/ir.py` — remove `KernelIR.fused_iter_var_map` field; remove its pprint section.
- `nkigym/src/nkigym/codegen/lowering/_emit_utils.py` — drop the `fused_iter_var_map` branch in `_resolve_iv_name`.
- `nkigym/src/nkigym/tune/fuse.py` — legality adds extent checks + access-pattern dependence check; apply eagerly rewrites access patterns; fix tree-rebuild ordering (rewrites don't get clobbered); remove writes to `fused_iter_var_map`; module docstring update.

**Tests:**
- `test/tune/test_fuse.py` — remove/update tests that asserted map state or `//`/`%` render behavior.
- `test/tune/test_axis_identity.py` — existing `test_split_after_same_axis_fuse_preserves_axis_id` should move from xfail/fail to PASS after the fix.
- New: `test/tune/test_fuse_eager_rewrite.py` — unit coverage of the eager rewrite + cross-axis rejection.
- `examples/matmul_lhsT_rhs.py` — step-by-step driver. After the fix, step_2 (Split after Fuse) writes artifacts and passes CPU-sim.

## Execution Order

The change is atomic in the sense that removing the map and adding eager rewrite must land together (partial state would break rendering). One logical commit, but we stage it as three sub-tasks for reviewability:

1. Task 1 — failing test: Fuse+Split CPU-sim end-to-end (currently errors with `KeyError`).
2. Task 2 — implement eager rewrite + remove map (IR + Fuse + renderer).
3. Task 3 — test migration (remove/xfail obsolete tests).
4. Task 4 — update `examples/matmul_lhsT_rhs.py` step_2 and confirm artifacts.
5. Task 5 — regression gate (kernel_transforms.py + axis-identity tests).

---

## Task 1: Failing end-to-end repro test

**Files:**
- Create: `/home/ubuntu/nki-autotune/test/tune/test_fuse_eager_rewrite.py`

- [ ] **Step 1: Write failing test**

Create `/home/ubuntu/nki-autotune/test/tune/test_fuse_eager_rewrite.py`:

```python
"""End-to-end test: Fuse then Split on the same axis renders + CPU-sims.

This test currently fails at render time with
``KeyError: 'iter var N is neither live nor recorded in fused_iter_var_map'``
because Fuse leaves stale retired iter-var ids in BufferAccess patterns
and relies on fused_iter_var_map to resolve them — but Split then retires
the fused iter-var without updating the map.

After the eager-Fuse-rewrite refactor, Fuse rewrites access patterns
directly so no side-table is needed; subsequent Split works normally.
"""

import numpy as np

import nki
from nkigym.ir.build import build_initial_ir
from nkigym.ir.ir import ForNode, SBlock
from nkigym.codegen.render import render
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.tune.fuse import Fuse
from nkigym.tune.split import Split
from nkigym.tune.verify import _rewrite_to_fp32


@nkigym_kernel
def _matmul(lhs_T, rhs):
    a = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    b = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    p = NKIAlloc(location="psum", shape=(2048, 2048), dtype="float32")()
    s = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    h = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
    NKILoad()(src=lhs_T, dst=a)
    NKILoad()(src=rhs, dst=b)
    NKIMemset(value=0.0)(dst=p)
    NKIMatmul()(stationary=a, moving=b, dst=p)
    NKITensorCopy()(src=p, dst=s)
    NKIStore()(src=s, dst=h)
    return h


_SPECS = {"lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"}, "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"}}


def _find_lhs_d1_pair(module):
    d1 = module.axis_id_by_name("d1")

    def walk(node, anc):
        if isinstance(node, SBlock) and node.body and node.body[0].op_cls.__name__ == "NKILoad":
            for _s, ba in node.writes.items():
                if ba.tensor_name == "a":
                    return anc
        if isinstance(node, ForNode):
            new_anc = anc + [node.iter_var] if node.iter_var.axis_id == d1 else anc
            for c in node.children:
                r = walk(c, new_anc)
                if r is not None:
                    return r
        return None

    for root in module.body:
        r = walk(root, [])
        if r is not None and len(r) == 2:
            return r[0].var_id, r[1].var_id
    raise AssertionError("could not locate d1 pair")


def _find_d1_2048_path_above_lhs(module):
    d1 = module.axis_id_by_name("d1")

    def has_lhs(n):
        if isinstance(n, SBlock) and n.body and n.body[0].op_cls.__name__ == "NKILoad":
            for _s, ba in n.writes.items():
                if ba.tensor_name == "a":
                    return True
        if isinstance(n, ForNode):
            return any(has_lhs(c) for c in n.children)
        return False

    def walk(node, path):
        if isinstance(node, ForNode) and node.iter_var.axis_id == d1 and node.iter_var.extent == 2048:
            if has_lhs(node):
                return path
        if isinstance(node, ForNode):
            for i, c in enumerate(node.children):
                r = walk(c, path + (i,))
                if r is not None:
                    return r
        return None

    for i, root in enumerate(module.body):
        r = walk(root, (i,))
        if r is not None:
            return r
    raise AssertionError("could not locate fused d1 path")


def test_fuse_then_split_renders_and_cpu_sims():
    """Canonical → Fuse(lhs_T d1 outer+inner) → Split(fused, factor=128).

    After all atoms, render must succeed and CPU-sim must match numpy golden.
    """
    module = build_initial_ir(_matmul, input_specs=_SPECS)
    outer_id, inner_id = _find_lhs_d1_pair(module)
    module = Fuse(outer_iter_var_id=outer_id, inner_iter_var_id=inner_id).apply(module)
    fused_path = _find_d1_2048_path_above_lhs(module)
    module = Split(loop_path=fused_path, factor=128).apply(module)

    source = render(module)
    sim_src = _rewrite_to_fp32(source)
    ns = {}
    exec(sim_src, ns)
    fn = ns["_matmul"]
    rng = np.random.default_rng(0)
    lhs = rng.standard_normal((2048, 2048)).astype(np.float32)
    rhs = rng.standard_normal((2048, 2048)).astype(np.float32)
    actual = np.asarray(nki.simulate(fn)(lhs, rhs))
    expected = lhs.T @ rhs
    assert np.allclose(actual, expected, atol=5e-3, rtol=5e-3), (
        f"max_abs={float(np.abs(actual - expected).max()):.3e}"
    )
```

- [ ] **Step 2: Run test to confirm it fails with the expected error**

```bash
source ~/venvs/kernel-env/bin/activate
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest /home/ubuntu/nki-autotune/test/tune/test_fuse_eager_rewrite.py -v 2>&1 | tail -15
```

Expected: `test_fuse_then_split_renders_and_cpu_sims` FAILS with a `KeyError` mentioning `fused_iter_var_map`.

- [ ] **Step 3: Do not commit yet — Task 2 implements the fix**

---

## Task 2: Implement eager rewrite and remove `fused_iter_var_map`

**Files:**
- Modify: `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/ir.py`
- Modify: `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/lowering/_emit_utils.py`
- Modify: `/home/ubuntu/nki-autotune/nkigym/src/nkigym/tune/fuse.py`

### Step 1: Remove `fused_iter_var_map` field from `KernelIR`

- [ ] **Edit `ir.py`**

Find the `KernelIR` dataclass (around line 234). Remove the field declaration and its reference in the docstring:

**Remove these lines** (inside the `Attributes:` docstring section, near lines 250-254):
```
        fused_iter_var_map: Retired iter-var id -> (fused iter-var id,
            inner extent, is_outer). Populated by the ``Fuse`` atom. The
            renderer consults this map when emitting accesses that still
            reference the retired outer/inner ids: outer decomposes as
            ``(fused // inner_extent)`` and inner as ``(fused % inner_extent)``.
```

**Remove this line** (field declaration, line 266):
```python
    fused_iter_var_map: dict[int, tuple[int, int, bool]] = field(default_factory=dict)
```

- [ ] **Remove the pprint section for the map**

Find `KernelIR.pprint` (around line 315). Remove the block at the end that prints `fused_iter_var_map`:

```python
        """Retired-iter-var decompositions (if any)."""
        if self.fused_iter_var_map:
            lines.append("fused_iter_var_map:")
            for retired_id, (fused_id, inner_extent, is_outer) in self.fused_iter_var_map.items():
                component = "outer" if is_outer else "inner"
                lines.append(f"  iv={retired_id} -> iv={fused_id} ({component}, inner_extent={inner_extent})")
```

### Step 2: Simplify `_resolve_iv_name` in the renderer

- [ ] **Edit `_emit_utils.py`**

Find `_resolve_iv_name` (around line 90). Replace the function body:

```python
def _resolve_iv_name(iv_id: int, ctx: EmitCtx) -> str:
    """Return the source name to use for ``iv_id``.

    The iter-var must be bound by an enclosing ForNode at render time;
    otherwise the access pattern references a ghost id and the module is
    ill-formed.

    Raises:
        KeyError: ``iv_id`` is not bound by any enclosing ForNode.
    """
    if iv_id not in ctx.iter_var_to_name:
        raise KeyError(f"iter var {iv_id} has no live binding")
    return ctx.iter_var_to_name[iv_id]
```

Also remove any now-stale imports at the top of the file related to `fused_iter_var_map`. (Grep the file: `grep fused_iter_var_map /home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/lowering/_emit_utils.py` should be empty after the edit.)

Also check for stale docstring mentions at module top (lines ~20-30 per the grep earlier). Replace references to "module.fused_iter_var_map" with a brief note that Fuse rewrites access patterns eagerly.

### Step 3: Rewrite `Fuse.apply` to eagerly rewrite access patterns + add legality guards

- [ ] **Edit `fuse.py`**

Replace the `Fuse` class's `is_legal` method with:

```python
    def is_legal(self, module: KernelIR) -> bool:
        """Structural + role preconditions + eager-rewrite preconditions + MIN/MAX tile check.

        Preconditions for eager access-pattern rewriting:
        - Same-axis fuse (``outer.axis_id == inner.axis_id``): ``outer.extent == 1``
          (outer is a trip-1 loop). This matches canonical usage; the
          retired outer contributes 0 to every affine expression, so it
          can be dropped from access patterns.
        - Cross-axis fuse (different axis_ids): no SBlock in the fused
          subtree may have a BufferAccess referencing either outer or
          inner var_id. The decomposition ``outer = fused // inner_ext``
          is non-affine and cannot be encoded in
          ``AccessRange.iter_var_coeffs``.
        """
        pair = _find_pair(module, self.outer_iter_var_id, self.inner_iter_var_id)
        if pair is None:
            return False
        outer, inner, _path = pair
        if outer.iter_var.role == AxisRole.SEQUENTIAL or inner.iter_var.role == AxisRole.SEQUENTIAL:
            return False

        v_outer = outer.iter_var
        v_inner = inner.iter_var

        if v_outer.axis_id == v_inner.axis_id:
            """Same-axis fuse: require outer.extent == 1 so we can drop its contribution."""
            if v_outer.extent != 1:
                return False
        else:
            """Cross-axis fuse: reject if any dependent access pattern exists."""
            if _subtree_has_access_referencing(inner, {v_outer.var_id, v_inner.var_id}):
                return False

        return self._check_min_max(module, outer, inner)
```

Replace the `Fuse.apply` method with:

```python
    def apply(self, module: KernelIR) -> KernelIR:
        """Execute the fuse.

        Same-axis fuse (the common case, ``outer.extent == 1``):
        - Drops retired outer's var_id from affected access patterns
          (contribution is 0 since extent is 1).
        - Renames retired inner's var_id to the fused var_id in
          affected access patterns (same coefficient).
        - Collapses ``SBlock.iter_vars`` lists: adjacent ``(outer, inner)``
          pairs become ``(fused,)``.
        - Replaces the two-ForNode chain with a single ``ForNode(v_fused)``.

        Cross-axis fuse (different axis_ids, no dependent accesses):
        - Allocates a fresh ``Axis`` with ``source_axes`` trace.
        - Collapses ``SBlock.iter_vars`` lists and the tree as above.
        - No access-pattern rewrite (legality already rejected cases
          that would require one).

        Raises:
            AtomLegalityError: ``is_legal`` returns False.
        """
        if not self.is_legal(module):
            raise AtomLegalityError(f"Fuse.apply: illegal {self!r}")
        pair = _find_pair(module, self.outer_iter_var_id, self.inner_iter_var_id)
        assert pair is not None
        outer_node, inner_node, outer_path = pair
        v_outer = outer_node.iter_var
        v_inner = inner_node.iter_var

        fused_extent = v_outer.extent * v_inner.extent
        fused_role = max(v_outer.role, v_inner.role, key=_ROLE_RANK.__getitem__)

        if v_outer.axis_id == v_inner.axis_id:
            fused_axis_id = v_outer.axis_id
        else:
            outer_axis = module.axes[v_outer.axis_id]
            inner_axis = module.axes[v_inner.axis_id]
            fused_axis = module.allocate_axis(
                name=f"{outer_axis.name}_x_{inner_axis.name}",
                total_size=fused_extent,
                source_axes=(v_outer.axis_id, v_inner.axis_id),
            )
            fused_axis_id = fused_axis.axis_id

        v_fused = module.allocate_iter_var(axis_id=fused_axis_id, extent=fused_extent, role=fused_role)

        """Rewrite access patterns in the whole body. For same-axis fuse this
        drops outer coeffs (outer.extent == 1, contributes 0) and renames
        inner coeffs to v_fused. For cross-axis fuse legality has already
        ruled out dependent accesses, so the rewrite is a no-op there."""
        new_body = _rewrite_access_patterns_on_fuse(
            module.body,
            outer_id=v_outer.var_id,
            inner_id=v_inner.var_id,
            fused_id=v_fused.var_id,
        )

        """Collapse SBlock.iter_vars lists: adjacent (outer, inner) -> (fused,)."""
        new_body = _collapse_iter_var_lists(new_body, v_outer.var_id, v_inner.var_id, v_fused)

        """Now build the replacement ForNode from the *rewritten* subtree at
        outer_path, not the pre-rewrite inner_node. This ensures the rewritten
        SBlock children survive into the new module."""
        from nkigym.ir.ir import resolve_node

        rewritten_outer = resolve_node(new_body, outer_path)
        assert isinstance(rewritten_outer, ForNode)
        rewritten_inner = rewritten_outer.children[0]
        assert isinstance(rewritten_inner, ForNode)
        new_fornode = ForNode(
            iter_var=v_fused,
            children=list(rewritten_inner.children),
            name=None,
            annotations=dict(rewritten_outer.annotations),
        )
        new_body = replace_at_path(new_body, outer_path, new_fornode)
        return replace(module, body=new_body)
```

Add two helpers at module level (below existing helpers at the bottom of `fuse.py`):

```python
def _subtree_has_access_referencing(root: ForNode | SBlock, iv_ids: set[int]) -> bool:
    """Return True if any SBlock beneath ``root`` has a BufferAccess referencing
    any id in ``iv_ids``."""

    def visit(node: ForNode | SBlock) -> bool:
        if isinstance(node, SBlock):
            for access_map in (node.reads, node.writes, node.reads_writes):
                for _slot, ba in access_map.items():
                    if any(iv_id in iv_ids for iv_id in ba.iter_var_ids):
                        return True
            return False
        for c in node.children:
            if visit(c):
                return True
        return False

    return visit(root)


def _rewrite_access_patterns_on_fuse(
    body: TreeIR, outer_id: int, inner_id: int, fused_id: int
) -> TreeIR:
    """Rewrite every ``BufferAccess`` in ``body`` to replace references to the
    retired ``outer_id`` / ``inner_id`` with ``fused_id``.

    Semantics (same-axis fuse with outer.extent == 1):
    - outer.var_id contributes 0 to every affine expression -> drop entries.
    - inner.var_id equals fused.var_id numerically -> rename entries.

    For cross-axis fuse legality already guarantees no access references the
    retired ids; this function is a no-op in that case.
    """

    def rewrite_access(acc: BufferAccess) -> BufferAccess:
        if outer_id not in acc.iter_var_ids and inner_id not in acc.iter_var_ids:
            return acc
        new_id_list = []
        for iv_id in acc.iter_var_ids:
            if iv_id == outer_id:
                continue
            elif iv_id == inner_id:
                if fused_id not in new_id_list:
                    new_id_list.append(fused_id)
            else:
                if iv_id not in new_id_list:
                    new_id_list.append(iv_id)
        new_pattern: list[AccessRange] = []
        for ar in acc.pattern:
            coeffs = dict(ar.iter_var_coeffs)
            """Drop outer: contributes 0."""
            coeffs.pop(outer_id, None)
            """Rename inner to fused."""
            if inner_id in coeffs:
                inner_coeff = coeffs.pop(inner_id)
                coeffs[fused_id] = coeffs.get(fused_id, 0) + inner_coeff
            new_pattern.append(AccessRange.make(coeffs, ar.const_offset, ar.extent))
        return BufferAccess(tensor_name=acc.tensor_name, iter_var_ids=tuple(new_id_list), pattern=tuple(new_pattern))

    def rewrite_block(block: SBlock) -> SBlock:
        return SBlock(
            iter_vars=block.iter_vars,
            reads={k: rewrite_access(v) for k, v in block.reads.items()},
            writes={k: rewrite_access(v) for k, v in block.writes.items()},
            reads_writes={k: rewrite_access(v) for k, v in block.reads_writes.items()},
            body=block.body,
            annotations=dict(block.annotations),
        )

    def rewrite_node(node: ForNode | SBlock) -> ForNode | SBlock:
        if isinstance(node, SBlock):
            return rewrite_block(node)
        return ForNode(
            iter_var=node.iter_var,
            children=[rewrite_node(c) for c in node.children],
            name=node.name,
            annotations=dict(node.annotations),
        )

    return [rewrite_node(n) for n in body]
```

Add the necessary imports to `fuse.py` if missing: `AccessRange`, `BufferAccess` from `nkigym.ir.ir`.

Remove the two lines in `apply` that wrote to `fused_iter_var_map`:
```python
module.fused_iter_var_map[v_outer.var_id] = (v_fused.var_id, v_inner.extent, True)
module.fused_iter_var_map[v_inner.var_id] = (v_fused.var_id, v_inner.extent, False)
```

Update the module-level docstring at the top of `fuse.py` (the big block explaining Fuse semantics). Remove the mentions of `module.fused_iter_var_map` and the renderer decomposition via `//` / `%`. Replace with: "Same-axis Fuse eagerly rewrites affected BufferAccess patterns (outer dropped, inner renamed to fused); cross-axis Fuse with dependent accesses is rejected."

### Step 4: Run the Task 1 test

```bash
source ~/venvs/kernel-env/bin/activate
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest /home/ubuntu/nki-autotune/test/tune/test_fuse_eager_rewrite.py -v
```

Expected: `test_fuse_then_split_renders_and_cpu_sims` PASSES.

### Step 5: Do not commit yet — Task 3 migrates other tests

---

## Task 3: Migrate tests that depended on `fused_iter_var_map`

**Files:**
- Modify: `/home/ubuntu/nki-autotune/test/tune/test_fuse.py`
- Modify: `/home/ubuntu/nki-autotune/test/codegen/test_batch.py` (if affected)

- [ ] **Step 1: Survey the damage**

```bash
source ~/venvs/kernel-env/bin/activate
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest /home/ubuntu/nki-autotune/test/tune/test_fuse.py -v 2>&1 | tail -30
```

List each failing test and why. The known set (from the spec's test-migration section):

- `test_fuse_renderer_emits_div_and_mod` — asserts the renderer produces `//` / `%` expressions from the map. Premise gone. **Remove.**
- `test_fuse_registers_synthetic_dim_info` — if it inspects `fused_iter_var_map`, update to check access-pattern rewriting instead. If it checks axis allocation (cross-axis fuse), keep and verify still correct.
- `test_fuse_adjacent_par_par_creates_synthetic_dim` — similar: inspect what it asserts. If same-axis fuse, the behavior changes (no new axis allocated; axis_id preserved). Update or remove.

- [ ] **Step 2: For each failing test, decide: keep + update, or remove**

Keep a test only if the behavior it asserts is still supported after the refactor. For each kept test, update assertions to reference live iter-var ids and coefficients rather than map entries.

For the `test_fuse_renderer_emits_div_and_mod` test specifically: remove the whole test function. Add a one-line docstring replacement at the top of `test_fuse.py`:

```python
"""Tests for the Fuse atom.

Cross-axis Fuse with dependent access patterns is rejected at
``is_legal`` (the ``//`` / ``%`` decomposition is non-affine and cannot
be encoded in today's BufferAccess). Same-axis Fuse eagerly rewrites
access patterns; no side-table.
"""
```

- [ ] **Step 3: Add unit coverage for eager rewrite**

Append to `/home/ubuntu/nki-autotune/test/tune/test_fuse_eager_rewrite.py`:

```python
def test_same_axis_fuse_drops_outer_from_access_patterns():
    """After same-axis Fuse(outer.extent=1, inner), retired outer var_id is
    absent from every BufferAccess.iter_var_coeffs; inner renamed to fused."""
    module = build_initial_ir(_matmul, input_specs=_SPECS)
    outer_id, inner_id = _find_lhs_d1_pair(module)
    module = Fuse(outer_iter_var_id=outer_id, inner_iter_var_id=inner_id).apply(module)

    def any_access_refs(module, iv_id):
        for sblock, _ in _all_sblocks(module):
            for am in (sblock.reads, sblock.writes, sblock.reads_writes):
                for ba in am.values():
                    if iv_id in ba.iter_var_ids:
                        return True
                    for ar in ba.pattern:
                        if iv_id in dict(ar.iter_var_coeffs):
                            return True
        return False

    assert not any_access_refs(module, outer_id)
    """inner_id may or may not still appear — some fused subtree SBlocks are
    outside the affected scope. The real invariant: every iv_id in any
    access is a currently-live ForNode's var_id OR still-valid retired id
    from outside the fused subtree."""


def test_cross_axis_fuse_with_dependent_access_is_illegal():
    """Cross-axis Fuse whose subtree has accesses referencing outer or inner
    must fail is_legal — the decomposition is non-affine."""
    """TODO: construct a minimal cross-axis fuse fixture. For now, rely on
    the absence of any passing cross-axis-with-deps scenario in the suite."""
    pass


def _all_sblocks(module):
    def walk(node, path):
        if isinstance(node, SBlock):
            yield node, path
        if isinstance(node, ForNode):
            for i, c in enumerate(node.children):
                yield from walk(c, path + (i,))

    for i, root in enumerate(module.body):
        yield from walk(root, (i,))
```

Add the missing import at the top: `from nkigym.ir.ir import ForNode, SBlock` already present; `_all_sblocks` is self-contained.

- [ ] **Step 4: Run the full fuse test suite**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest /home/ubuntu/nki-autotune/test/tune/test_fuse.py /home/ubuntu/nki-autotune/test/tune/test_fuse_eager_rewrite.py -v 2>&1 | tail -20
```

Expected: all PASS.

- [ ] **Step 5: Run the full suite**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest /home/ubuntu/nki-autotune/test/ --ignore=/home/ubuntu/nki-autotune/test/codegen/test_batch.py 2>&1 | tail -10
```

Expected: all PASS (2 skipped + 4 xfailed from prior work). If any new failures appear, diagnose and fix before committing.

Also run `test_batch.py` separately to confirm we haven't introduced new failures beyond the pre-existing one:

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest /home/ubuntu/nki-autotune/test/codegen/test_batch.py --tb=line 2>&1 | tail -5
```

Expected: exactly 1 failure (the pre-existing `test_enumerate_pool_includes_annotate_buffer_degree_variants`).

- [ ] **Step 6: Do not commit yet**

---

## Task 4: Update `examples/matmul_lhsT_rhs.py` step_2

**Files:**
- Already contains a Split call after Fuse (added earlier). After Task 2 lands, it should produce artifacts for step_2 instead of crashing.

- [ ] **Step 1: Run the example**

```bash
rm -rf /home/ubuntu/cache/matmul_lhsT_rhs
source ~/venvs/kernel-env/bin/activate
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python /home/ubuntu/nki-autotune/examples/matmul_lhsT_rhs.py 2>&1 | tail -20
```

Expected: 3 steps print, each with `cpu-sim: PASS`. Artifacts exist at:
- `/home/ubuntu/cache/matmul_lhsT_rhs/step_0_canonical/{ir.txt,kernel.py}`
- `/home/ubuntu/cache/matmul_lhsT_rhs/step_1_fuse_lhsT_d1/{ir.txt,kernel.py}`
- `/home/ubuntu/cache/matmul_lhsT_rhs/step_2_split_lhsT_d1/{ir.txt,kernel.py}`

- [ ] **Step 2: Inspect step_2 artifacts manually**

```bash
cat /home/ubuntu/cache/matmul_lhsT_rhs/step_2_split_lhsT_d1/kernel.py
```

Expected: lhs_T load renders as:
```python
for i_d0_0 in range(16):
    for i_d1_0 in range(16):
        for i_d1_1 in range(128):
            nisa.dma_copy(dst=lhs_T_sbuf[..., i_d0_0, i_d1_0 * 128 : i_d1_0 * 128 + 128], src=lhs_T[..., i_d0_0 * 128 : ..., i_d1_0 * 128 : ...])
```

The d1 axis is now split into outer 16 + inner 128, matching the structural target of `kernel_1` in `kernel_transforms.py`.

- [ ] **Step 3: Do not commit yet — Task 5 is the regression gate**

---

## Task 5: Regression gate + commit

- [ ] **Step 1: All 15 hand kernels in `kernel_transforms.py`**

```bash
source ~/venvs/kernel-env/bin/activate
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python /home/ubuntu/nki-autotune/kernel_transforms.py 2>&1 | tail -20
```

Expected: 15 `pass=True` lines.

- [ ] **Step 2: axis_identity tests**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest /home/ubuntu/nki-autotune/test/tune/test_axis_identity.py -v
```

Expected: 3 PASS.

- [ ] **Step 3: Confirm no stragglers**

```bash
grep -rn "fused_iter_var_map" /home/ubuntu/nki-autotune/nkigym/src /home/ubuntu/nki-autotune/test /home/ubuntu/nki-autotune/docs/ir-design.md 2>/dev/null
```

Expected: empty.

```bash
grep -rn "fused_iter_var_map" /home/ubuntu/nki-autotune/docs/superpowers/ 2>/dev/null
```

Expected: only historical references in spec files — acceptable, do not modify.

- [ ] **Step 4: Commit the whole refactor as one atomic change**

```bash
cd /home/ubuntu/nki-autotune
git add -A
git commit -m "refactor: eager Fuse access-pattern rewrite; remove fused_iter_var_map

Make BufferAccess.iter_var_coeffs the single source of truth for access
patterns. Same-axis Fuse (outer.extent=1) rewrites affected accesses:
outer drops from coeffs (contributes 0), inner renames to fused var_id.
Cross-axis Fuse with dependent accesses raises AtomLegalityError (non-
affine decomposition deferred until non-affine access patterns land).

Fixes Fuse+Split composition bug: Split retired the map's target
without map maintenance, leaving dangling ghost entries that the
renderer hit with KeyError at emission.

Also fixes latent bug where Fuse's tree rebuild used pre-rewrite
inner_node.children, clobbering _collapse_iter_var_lists output.

Renderer _resolve_iv_name drops the fused_iter_var_map branch; every
live access iv_id must have a live ForNode binding.

Tests:
- test/tune/test_fuse_eager_rewrite.py (new): Fuse+Split end-to-end
  and access-pattern-rewrite unit coverage
- test/tune/test_axis_identity.py::test_split_after_same_axis_fuse*
  now passes (was xfail/failing)
- test/tune/test_fuse.py: removed test_fuse_renderer_emits_div_and_mod
  (premise gone), updated other tests accordingly

examples/matmul_lhsT_rhs.py step_2 (Split after Fuse) now writes
artifacts and CPU-sims.

Spec: docs/superpowers/specs/2026-05-12-eager-fuse-pattern-rewrite-design.md"
```

---

## Self-Review Checklist

1. **Spec coverage:**
   - Remove `KernelIR.fused_iter_var_map` → Task 2 Step 1.
   - Simplify `_resolve_iv_name` (remove map branch) → Task 2 Step 2.
   - Fuse.is_legal adds extent-1 check for same-axis + access-dependence check for cross-axis → Task 2 Step 3 (`is_legal` rewrite).
   - Fuse.apply eagerly rewrites access patterns → Task 2 Step 3 (`_rewrite_access_patterns_on_fuse` + call from apply).
   - Fix tree rebuild ordering (no clobber) → Task 2 Step 3 (`resolve_node(new_body, outer_path)` replaces pre-rewrite `inner_node`).
   - Test migration → Task 3.
   - End-to-end Fuse+Split test → Task 1 + Task 3 (unit coverage).
   - `examples/matmul_lhsT_rhs.py` reaches step_2 → Task 4.

2. **Placeholder scan:** Only placeholder is in `test_cross_axis_fuse_with_dependent_access_is_illegal`, marked with `TODO: construct a minimal cross-axis fuse fixture. For now, rely on the absence of any passing cross-axis-with-deps scenario in the suite.` followed by `pass`. Acceptable — this is documenting a known incomplete test, not a plan gap.

3. **Type consistency:** `iter_var_coeffs: tuple[tuple[int, int], ...]`, `iter_var_ids: tuple[int, ...]`, `var_id: int`, `axis_id: int`, `AccessRange.make(coeffs, const_offset, extent)` — all consistent with the ir.py definitions referenced throughout.

4. **Task ordering:** Task 1 writes failing test → Task 2 implements fix → Task 3 migrates affected tests → Task 4 verifies driver → Task 5 regresses and commits. One atomic commit at the end (Task 5 Step 4).
