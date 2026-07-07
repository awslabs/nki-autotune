# BufferLayout Transform + k0→k26 Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the `BufferLayout` transform (re-factorizes a buffer's tile axis into any `list_len × per_tile = T` form) and drive the shipped transforms to reproduce `examples/manual_transforms.py` k0→k26 byte-exact + CPU-sim + HW-MFU on gym-1.

**Architecture:** `BufferLayout` is a pure field-set on `Buffer.list_len` (mirrors how `SoftwarePipeline` sets `Buffer.versions`) — no region surgery, no tree-structure change. Codegen renders every sbuf/psum buffer as a Python list uniformly (`b=1` as a list-of-1); the renderer is generalized to the `a>1` middle via literal `t//a` / `t%a` emission. The driven ladder lives in `examples/kernel_transforms.py`, diffed rung-for-rung against the uniform-list `manual_transforms.py` reference.

**Tech Stack:** Python 3.12, `networkx` (schedule tree), the in-house `nkigym.ir.arith` expr substrate, `pytest`. No local Python env — all tests run on gym-1 via `transport/ssh_host.sh`.

## Global Constraints

- **No local Python env** — run every test on gym-1: `transport/ssh_host.sh --host gym-1 --cmd "python -m pytest <files>" --cache /home/weittang/workplace/cache/<name>`. `--cache` is REQUIRED even for pytest; `--cmd` needs a `.py` token (enumerate files, no bare `test/` dir). Sets `PYTHONPATH=.:nkigym/src:autotune/src` itself.
- **Verify "new" failures vs the parent commit** — the pre-existing failing-test set is stable; a failure that reproduces at `HEAD~` is not yours.
- **Code style** (`.claude/rules/code_style.md`): triple-quoted block comments only (no `#` comments except tooling directives like `# type: ignore`); Google/NumPy docstrings on every function; modern type hints (`list`/`dict`/`X | None`); single return per function where practical; no bare `except`; loud failures (raise, never silently adapt). Format with `black` (line-length 120) + `isort` — pre-commit reformats-and-aborts, so re-stage and retry.
- **Transform legality gates correctness / dep-order / ISA-wellformedness ONLY, never resource capacity** (user-locked). `BufferLayout`'s checks are all structural renderability guards.
- **`BufferLayout` conserves T** (`a·b = T`) — it re-factorizes existing tiles, never creates them. Tile *creation* (double-buffering) is `versions` territory (SoftwarePipeline), enforced by the `versions>1` reject.
- **Byte-exact means AST-canonical equal** via `assert_matches_render` — "same means same", not "roughly the same".
- **Commit after every task.** Branch is `dev_1` (not default); commit there. End commit messages with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer.

## File Structure

- `nkigym/src/nkigym/ir/tree.py` — `Buffer` dataclass. **Modify:** rename field `num_tiles → list_len` (15 refs).
- `nkigym/src/nkigym/codegen/body.py` — `_emit_alloc`, `render_buffer_region`. **Modify:** rename (7 refs); uniform-list alloc (drop bare-`nl.ndarray` branch); general `a>1` render.
- `nkigym/src/nkigym/codegen/compact.py` — `compact_shapes`. **Modify:** rename (1 ref); (Task 5) fix leading-axis compaction on a list only if the composability test fails.
- `nkigym/src/nkigym/transforms/buffer_layout.py` — **Create:** `BufferLayout`, `BufferLayoutOption`.
- `nkigym/src/nkigym/transforms/__init__.py` — **Modify:** export `BufferLayout`, `BufferLayoutOption`.
- `test/ir/test_node_labels.py` — **Modify:** rename (14 refs).
- `test/codegen/test_body.py` — **Modify:** rename (7 refs); update b=1 assertions to list-of-1; add `a>1` render tests.
- `test/codegen/test_render.py`, `test/codegen/test_compact.py`, `test/transforms/test_ladder_compare.py` — **Modify:** update bare-`b=1` assertions to list-of-1 (part of the 13-assertion blast radius).
- `test/transforms/test_buffer_layout.py` — **Create:** analyze/apply/legality/conservation + standalone k5→k6.
- `examples/kernel_transforms.py` — **Modify (Task 6):** replace `_build_ladder` with the 26-step manual-order chain; wire per-rung `assert_matches_render` + CPU-sim + HW-MFU.

---

### Task 1: Rename `Buffer.num_tiles → Buffer.list_len`

Pure mechanical rename — no behavior change. The shipped field is named `num_tiles` but semantically means **b** (the list length, `for _ in range(num_tiles)`); `a = T // num_tiles` is the per-tile middle. Renaming to `list_len` removes the naming trap before any new code references it. 44 occurrences across 5 files.

**Files:**
- Modify: `nkigym/src/nkigym/ir/tree.py` (15 refs incl. the field def at line 137 + docstrings)
- Modify: `nkigym/src/nkigym/codegen/body.py` (7 refs)
- Modify: `nkigym/src/nkigym/codegen/compact.py` (1 ref, a docstring at line 117)
- Modify: `test/ir/test_node_labels.py` (14 refs)
- Modify: `test/codegen/test_body.py` (7 refs)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `Buffer(name, shape, dtype, location, versions=1, list_len=1)` — the field every later task sets/reads. `Buffer.per_tile_physical_shape()` unchanged in behavior: returns `(P, T // list_len, F)`.

- [ ] **Step 1: Rename across all 5 files**

The token `num_tiles` appears only as this field name (verified: no unrelated `num_tiles` elsewhere). A whole-word substitution is safe:

```bash
cd /workplace/weittang/nki-autotune
for f in nkigym/src/nkigym/ir/tree.py nkigym/src/nkigym/codegen/body.py \
         nkigym/src/nkigym/codegen/compact.py test/ir/test_node_labels.py \
         test/codegen/test_body.py; do
  sed -i 's/\bnum_tiles\b/list_len/g' "$f"
done
```

- [ ] **Step 2: Verify no `num_tiles` remains and `list_len` count matches**

```bash
cd /workplace/weittang/nki-autotune
grep -rn "num_tiles" nkigym/ test/ | grep -v __pycache__   # expect: no output
grep -rc "list_len" nkigym/src/nkigym/ir/tree.py nkigym/src/nkigym/codegen/body.py \
  nkigym/src/nkigym/codegen/compact.py test/ir/test_node_labels.py test/codegen/test_body.py
```
Expected: first grep prints nothing; second prints `15`, `7`, `1`, `14`, `7`.

- [ ] **Step 3: Run the affected unit tests on gym-1**

```bash
transport/ssh_host.sh --host gym-1 \
  --cmd "python -m pytest test/ir/test_node_labels.py test/codegen/test_body.py -q" \
  --cache /home/weittang/workplace/cache/buflayout_t1
```
Expected: PASS (same count as before the rename — pure rename, no behavior change).

- [ ] **Step 4: Commit**

```bash
cd /workplace/weittang/nki-autotune
git add nkigym/src/nkigym/ir/tree.py nkigym/src/nkigym/codegen/body.py \
  nkigym/src/nkigym/codegen/compact.py test/ir/test_node_labels.py test/codegen/test_body.py
git commit -m "Rename Buffer.num_tiles -> Buffer.list_len (b = list length)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Uniform-list rendering + general `a>1` middle

Two coupled render changes in `codegen/body.py`, both removing special-cases. The key quantity is `a = buf.per_tile_physical_shape()[1] = T // list_len` (per-tile middle dim). Three render cases, branched on **`list_len`** (NOT on `a` — see below):

| case | condition | list index | middle index |
|---|---|---|---|
| list-of-1 | `list_len == 1` | `0` | the whole tile index `t` |
| full split | `list_len == T` (⇒ `a == 1`) | `t` | `0` |
| general | `1 < list_len < T` (⇒ `a > 1`) | `t // a` | `t % a` |

**Why branch on `list_len`, not `a`:** a canonical `list_len==1` multi-tile buffer (e.g. `sbuf_lhs_T` `(128,16,2048)`) has `a = 16`. If we branched on `a>1` it would render `sbuf_lhs_T[0][… i_d0_0 // 16 …]` — wrong. `list_len==1` must always render `buf[0][0:P, t, F]` (list index literally `0`, middle the whole tile index), which is what makes the uniform-list rewrite of `manual_transforms.py` byte-exact.

**The `a>1` case** emits LITERAL `t // a`, `t % a` via the non-normalising `_format_raw` — NOT `format_expr`, which routes through `to_affine` and RAISES `NonAffineError` on `(outer*a + inner) // a` (the coeff of `inner` is 1, not divisible by `a`). The manual ladder never uses `a>1`, so no byte-exact fold of the aligned index is required; the literal form is correct and sufficient.

**Files:**
- Modify: `nkigym/src/nkigym/codegen/body.py` — `_emit_alloc` (lines 238-253), `render_buffer_region` (lines 313-347), arith import (line 21).
- Modify: `test/codegen/test_body.py`, `test/codegen/test_render.py`, `test/codegen/test_compact.py`, `test/transforms/test_ladder_compare.py` — update the 13 bare-`b=1` assertions to list-of-1.
- Test: `test/codegen/test_body.py` (new `a>1` + list-of-1 cases).

**Interfaces:**
- Consumes: `Buffer.list_len` (Task 1); `Buffer.per_tile_physical_shape() -> (P, T//list_len, F)`; `Const, Expr, FloorDiv, Mod, Mul, Var, _format_raw, format_expr` from `nkigym.ir.arith.expr`; existing module-local `_format_tile_index(lo, rotation)`.
- Produces: `_emit_alloc(buf) -> str` — HBM bare 2D; sbuf/psum always `name = [nl.ndarray((P, a, F), …) for _ in range(list_len)]`. `render_buffer_region(region, buf, rotation=None) -> str` — sbuf/psum always `name[<list_idx>][0:P, <mid_idx>, F]` per the table above.

- [ ] **Step 1: Write the failing tests**

Extend the arith import at the top of `test/codegen/test_body.py` to `from nkigym.ir.arith.expr import Const, Mod, Mul, Var` (add nothing new — `Mod`/`Mul` already imported; the test builds only `Var`/`Const`/`Mul`). Add:

```python
def test_emit_alloc_b1_is_list_of_one():
    """A list_len==1 sbuf buffer emits a list-of-1, not a bare ndarray."""
    buf = Buffer(name="sbuf_lhs_T", shape=(2048, 2048), dtype="bfloat16", location="sbuf")
    out = _emit_alloc(buf)
    assert out == "sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]"


def test_render_b1_multi_tile_is_list0_whole_index():
    """list_len==1 (T=16) renders buf[0][0:128, i_d0_0, F] — list index 0, whole tile index."""
    buf = Buffer(name="sbuf_lhs_T", shape=(2048, 2048), dtype="bfloat16", location="sbuf")
    region = BufferRegion(
        tensor="sbuf_lhs_T",
        ranges=((Var(name="i_d0_0"), Const(value=128)),
                (Mul(left=Var(name="i_d1_0"), right=Const(value=128)), Const(value=128))),
    )
    out = render_buffer_region(region, buf)
    assert out == "sbuf_lhs_T[0][0:128, i_d0_0, i_d1_0 * 128:i_d1_0 * 128 + 128]"


def test_render_full_split_is_list_index_middle_zero():
    """list_len==T (a==1) renders buf[t][0:128, 0, F] — the k6 full-split form."""
    buf = Buffer(name="psum_prod", shape=(2048, 512), dtype="float32", location="psum", list_len=16)
    region = BufferRegion(
        tensor="psum_prod",
        ranges=((Var(name="i_d1_0"), Const(value=128)),
                (Mul(left=Var(name="i_d2_0"), right=Const(value=512)), Const(value=512))),
    )
    out = render_buffer_region(region, buf)
    assert out == "psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512:i_d2_0 * 512 + 512]"


def test_render_general_a_gt_1_literal_divmod():
    """a>1 (list_len=8 on T=16) emits literal t//a leading, t%a middle."""
    buf = Buffer(name="s", shape=(2048, 512), dtype="bfloat16", location="sbuf", list_len=8)
    assert buf.per_tile_physical_shape() == (128, 2, 512)  # a == 2
    region = BufferRegion(
        tensor="s",
        ranges=((Var(name="t"), Const(value=128)), (Const(value=0), Const(value=512))),
    )
    out = render_buffer_region(region, buf)
    assert out == "s[t // 2][0:128, t % 2, 0:0 + 512]"


def test_emit_alloc_hbm_stays_bare():
    """shared_hbm keeps its bare 2D ndarray (no tile axis, never listed)."""
    buf = Buffer(name="hbm_out", shape=(2048, 2048), dtype="bfloat16", location="shared_hbm")
    out = _emit_alloc(buf)
    assert out == "hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)"
```

- [ ] **Step 2: Run to verify they fail**

```bash
transport/ssh_host.sh --host gym-1 \
  --cmd "python -m pytest test/codegen/test_body.py -q -k 'b1 or full_split or general_a or hbm_stays'" \
  --cache /home/weittang/workplace/cache/buflayout_t2
```
Expected: FAIL — `test_emit_alloc_b1_is_list_of_one` gets the bare-ndarray string; `test_render_b1_multi_tile_is_list0_whole_index` gets the old packed form; `test_render_general_a_gt_1_literal_divmod` raises `AssertionError("per-tile degree 1 only")`.

- [ ] **Step 3: Rewrite `_emit_alloc` (uniform list for sbuf/psum, bare for HBM)**

Replace `_emit_alloc` (body.py lines 238-253) with:

```python
def _emit_alloc(buf: Buffer) -> str:
    """Emit the buffer declaration for ``buf``.

    ``shared_hbm`` buffers emit a single bare ``nl.ndarray`` of
    :meth:`Buffer.physical_shape` (no tile axis). Every sbuf/psum buffer emits a
    Python list of :attr:`Buffer.list_len` per-tile ndarrays
    (:meth:`Buffer.per_tile_physical_shape`) — uniformly, including ``list_len == 1``
    (a list-of-one), so the call site always indexes with a leading ``[list_idx]``.
    """
    if buf.location == "shared_hbm":
        shape = "(" + ", ".join(str(s) for s in buf.physical_shape()) + ")"
        result = f"{buf.name} = nl.ndarray({shape}, dtype=nl.{buf.physical_dtype()}, buffer=nl.{buf.location})"
    else:
        shape = "(" + ", ".join(str(s) for s in buf.per_tile_physical_shape()) + ")"
        result = (
            f"{buf.name} = [nl.ndarray({shape}, dtype=nl.{buf.physical_dtype()}, "
            f"buffer=nl.{buf.location}) for _ in range({buf.list_len})]"
        )
    return result
```

- [ ] **Step 4: Rewrite `render_buffer_region` (three-way list branch on `list_len`)**

Add `FloorDiv` to the arith import (body.py line 21 — it currently reads `from nkigym.ir.arith.expr import Const, Expr, Mod, Mul, Var, _format_raw, format_expr`):
```python
from nkigym.ir.arith.expr import Const, Expr, FloorDiv, Mod, Mul, Var, _format_raw, format_expr
```

Replace `render_buffer_region` (body.py lines 313-347) with:

```python
def render_buffer_region(region: BufferRegion, buf: Buffer, rotation: Expr | None = None) -> str:
    """Render a :class:`BufferRegion` as a Python slice expression on its tensor.

    ``shared_hbm`` renders flat ``name[lo:hi, ...]``. Every sbuf/psum buffer renders
    as a list access ``name[list_idx][0:P, mid_idx, F]`` (uniform — there is no bare
    form). The partition axis (axis 0) carries the tile index ``t``; with
    ``a = per_tile middle = T // list_len``, branch on ``list_len``:

    * ``list_len == 1`` — a list-of-one: ``list_idx = 0``, ``mid_idx = t`` (the whole
      tile index). Preserves the pre-uniform packed middle, so a canonical multi-tile
      buffer renders ``buf[0][0:P, t, F]``.
    * ``a == 1`` (``list_len == T``, the full split) — ``list_idx = t``, ``mid_idx = 0``.
    * ``a > 1`` (``1 < list_len < T``) — ``list_idx = t // a``, ``mid_idx = t % a``,
      both via the non-normalising ``_format_raw`` (the aligned index is non-affine
      under ``FloorDiv``, so ``format_expr``/``to_affine`` would raise).

    ``rotation`` (the pipeline version term) applies only on the ``list_len == 1`` and
    ``a == 1`` paths; ``a > 1`` requires ``list_len > 1``, and ``versions > 1`` with
    ``list_len > 1`` is rejected at allocation, so no rotation reaches the ``a > 1`` path.
    """
    list_subscript = ""
    parts: list[str] = []
    for axis_index, (lo, hi) in enumerate(region.ranges):
        if axis_index == 0 and buf.location != "shared_hbm":
            if not isinstance(hi, Const) or hi.value != PARTITION_DIM:
                raise AssertionError(f"{buf.name}: SBUF/PSUM partition axis must use a partition-sized tile; got {hi}")
            a = buf.per_tile_physical_shape()[1]
            if buf.list_len == 1:
                list_subscript = "[0]"
                parts.append(f"0:{PARTITION_DIM}")
                parts.append(_format_tile_index(lo, rotation))
            elif a == 1:
                list_subscript = f"[{_format_tile_index(lo, rotation)}]"
                parts.append(f"0:{PARTITION_DIM}")
                parts.append("0")
            else:
                list_subscript = f"[{_format_raw(FloorDiv(left=lo, right=Const(value=a)))}]"
                parts.append(f"0:{PARTITION_DIM}")
                parts.append(_format_raw(Mod(left=lo, right=Const(value=a))))
        else:
            lo_str = format_expr(lo)
            hi_str = format_expr(hi)
            parts.append(f"{lo_str}:{lo_str} + {hi_str}")
    return f"{region.tensor}{list_subscript}[{', '.join(parts)}]"
```

- [ ] **Step 5: Run the new tests to verify they pass**

```bash
transport/ssh_host.sh --host gym-1 \
  --cmd "python -m pytest test/codegen/test_body.py -q -k 'b1 or full_split or general_a or hbm_stays'" \
  --cache /home/weittang/workplace/cache/buflayout_t2
```
Expected: PASS.

- [ ] **Step 6: Update the 13 pre-existing bare-`b=1` assertions to list-of-1**

The uniform-list change breaks every existing test that asserted a bare `nl.ndarray` decl or a packed `X[0:128, t, F]` access for a `list_len==1` buffer. Find them:

```bash
cd /workplace/weittang/nki-autotune
grep -rn "nl.ndarray((128" test/codegen/test_body.py test/codegen/test_render.py \
  test/codegen/test_compact.py test/transforms/test_ladder_compare.py | grep -v "for _ in range"
grep -rnE "(sbuf_lhs_T|sbuf_rhs|psum_prod|sbuf_prod)\[0:128" test/codegen/test_body.py \
  test/codegen/test_render.py test/codegen/test_compact.py test/transforms/test_ladder_compare.py
```

Update rule (uniform, since all these fixtures are `list_len==1`): a bare decl
`X = nl.ndarray((128, T, F), …)` → `X = [nl.ndarray((128, T, F), …) for _ in range(1)]`;
a bare access `X[0:128, t, F]` → `X[0][0:128, t, F]` (middle index UNCHANGED — the
`list_len==1` branch keeps the whole tile index). Any `list_len>1` fixture already using
the list form is unaffected.

- [ ] **Step 7: Run the full codegen + ladder-compare suites**

```bash
transport/ssh_host.sh --host gym-1 \
  --cmd "python -m pytest test/codegen/test_body.py test/codegen/test_render.py test/codegen/test_compact.py test/transforms/test_ladder_compare.py -q" \
  --cache /home/weittang/workplace/cache/buflayout_t2
```
Expected: PASS (all updated assertions green; no `num_tiles`/bare-`b=1` remnants).

- [ ] **Step 8: Commit**

```bash
cd /workplace/weittang/nki-autotune
git add nkigym/src/nkigym/codegen/body.py test/codegen/test_body.py \
  test/codegen/test_render.py test/codegen/test_compact.py test/transforms/test_ladder_compare.py
git commit -m "Codegen: uniform list-of-tiles render (b=1 as list-of-1) + general a>1

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: The `BufferLayout` transform

A pure field-set transform: sets one buffer's `list_len` and rebuilds the dependency sidecar for contract uniformity. No region surgery, no tree-structure change. Mirrors `SoftwarePipeline` (which sets the sibling `Buffer.versions` via `_set_versions`).

**Files:**
- Create: `nkigym/src/nkigym/transforms/buffer_layout.py`
- Modify: `nkigym/src/nkigym/transforms/__init__.py` — add the two exports.
- Test: `test/transforms/test_buffer_layout.py`

**Interfaces:**
- Consumes: `Transform, TransformLegalityError, TransformOption` from `nkigym.transforms.base`; `KernelIR`; `Dependency` from `nkigym.ir.dependency`; `BlockNode` from `nkigym.ir.tree`; `Buffer.list_len`, `Buffer.physical_shape() -> (P, T, F)`. Pattern-match `SoftwarePipeline._set_versions` (`software_pipeline.py:149-156`) for the alloc-entry replace.
- Produces: `BufferLayout().analyze(ir) -> list[BufferLayoutOption]`; `BufferLayout().apply(ir, BufferLayoutOption(tensor, list_len)) -> KernelIR`. `BufferLayoutOption(tensor: str, list_len: int)` frozen dataclass. Later tasks (5, 6) call these.

- [ ] **Step 1: Write the failing tests**

Create `test/transforms/test_buffer_layout.py`. Use the canonical IR from `build_initial_ir(f_nkigym, INPUT_SPECS)`, whose `psum_prod` is packed `(128,16,2048)` (`T=16, list_len=1`) — a known shape — rather than `tuned_ir()` (whose `psum_prod` tile count depends on how far its trace compacts — do not guess it). The first test (`test_canonical_psum_is_packed_T16`) pins the `T=16` assumption; if it fails, read the actual `physical_shape()[1]` and adjust the divisor-set expectation (the transform logic is T-agnostic; only the test's literal expected set depends on T).

```python
"""Tests for nkigym.transforms.BufferLayout (tile-axis re-factorization)."""

from __future__ import annotations

import pytest

from nkigym.ir import build_initial_ir
from nkigym.transforms import BufferLayout, BufferLayoutOption, TransformLegalityError

from examples.kernel_transforms import f_nkigym, INPUT_SPECS


def _canonical_ir():
    return build_initial_ir(f_nkigym, INPUT_SPECS)


def _tile_count(ir, name):
    return ir.buffer(name).physical_shape()[1]


def _divisors(n):
    return {d for d in range(1, n + 1) if n % d == 0}


def test_canonical_psum_is_packed_T16():
    """Guard: the canonical psum_prod is packed (128,16,2048) — T=16, list_len=1.

    The divisor-set / apply tests below assume T=16; this pins that assumption so a
    fixture change surfaces here, not as a confusing failure downstream."""
    ir = _canonical_ir()
    assert _tile_count(ir, "psum_prod") == 16
    assert ir.buffer("psum_prod").list_len == 1


def test_analyze_enumerates_every_divisor_of_T():
    """A T=16 psum buffer offers list_len in divisors(16) minus its current layout (1)."""
    ir = _canonical_ir()
    opts = [o for o in BufferLayout().analyze(ir) if o.tensor == "psum_prod"]
    assert {o.list_len for o in opts} == _divisors(16) - {1}  # {2, 4, 8, 16}


def test_analyze_skips_shared_hbm():
    """No option targets a shared_hbm buffer (no tile axis)."""
    ir = _canonical_ir()
    assert all(ir.buffer(o.tensor).location != "shared_hbm" for o in BufferLayout().analyze(ir))


def test_apply_sets_list_len_full_split():
    """apply(psum_prod, 16) sets list_len=16; tree node count unchanged; original untouched."""
    ir = _canonical_ir()
    n_before = ir.tree.graph.number_of_nodes()
    new_ir = BufferLayout().apply(ir, BufferLayoutOption(tensor="psum_prod", list_len=16))
    assert new_ir.buffer("psum_prod").list_len == 16
    assert ir.buffer("psum_prod").list_len == 1
    assert new_ir.tree.graph.number_of_nodes() == n_before


def test_apply_conserves_total_tiles():
    """T (=list_len*a) is invariant across apply — re-factorize, never create."""
    ir = _canonical_ir()
    t_before = _tile_count(ir, "psum_prod")
    new_ir = BufferLayout().apply(ir, BufferLayoutOption(tensor="psum_prod", list_len=4))
    b = new_ir.buffer("psum_prod")
    assert b.list_len * b.per_tile_physical_shape()[1] == t_before


def test_apply_round_trip_identity():
    """list->pack->list returns to the same list_len."""
    ir = _canonical_ir()
    listed = BufferLayout().apply(ir, BufferLayoutOption(tensor="psum_prod", list_len=16))
    packed = BufferLayout().apply(listed, BufferLayoutOption(tensor="psum_prod", list_len=1))
    assert packed.buffer("psum_prod").list_len == 1


def test_apply_rejects_missing_tensor():
    ir = _canonical_ir()
    with pytest.raises(TransformLegalityError):
        BufferLayout().apply(ir, BufferLayoutOption(tensor="does_not_exist", list_len=2))


def test_apply_rejects_non_divisor():
    """list_len must divide T; 3 does not divide 16."""
    ir = _canonical_ir()
    with pytest.raises(TransformLegalityError):
        BufferLayout().apply(ir, BufferLayoutOption(tensor="psum_prod", list_len=3))


def test_apply_rejects_noop():
    """Setting list_len to its current value is rejected (no-op)."""
    ir = _canonical_ir()
    with pytest.raises(TransformLegalityError):
        BufferLayout().apply(ir, BufferLayoutOption(tensor="psum_prod", list_len=1))
```

- [ ] **Step 2: Run to verify they fail**

```bash
transport/ssh_host.sh --host gym-1 \
  --cmd "python -m pytest test/transforms/test_buffer_layout.py -q" \
  --cache /home/weittang/workplace/cache/buflayout_t3
```
Expected: FAIL at import (`ImportError: cannot import name 'BufferLayout'`).

- [ ] **Step 3: Write the transform**

Create `nkigym/src/nkigym/transforms/buffer_layout.py`:

```python
"""``BufferLayout`` transform — re-factorize a buffer's tile axis into a
``list_len x per_tile`` form (``list_len * per_tile == T``, the total tile count).

A pure field-set on :attr:`Buffer.list_len`: it changes neither regions nor
tree structure, only allocation granularity. It CONSERVES the tile count (never
creates tiles — that is ``versions`` / SoftwarePipeline). Mirrors
:class:`SoftwarePipeline`, which sets the sibling :attr:`Buffer.versions`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import BlockNode
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class BufferLayoutOption(TransformOption):
    """Relayout ``tensor`` to ``list_len`` tiles (``per_tile = T // list_len`` derived).

    Attributes:
        tensor: buffer name to relayout.
        list_len: target list length b. 1 = list-of-one (packed), T = full split.
    """

    tensor: str
    list_len: int


class BufferLayout(Transform):
    """Re-factorize one buffer's tile axis; sets :attr:`Buffer.list_len`."""

    def analyze(self, ir: KernelIR) -> list[BufferLayoutOption]:
        """Every (tensor, divisor-of-T) relayout for each sbuf/psum, single-version buffer."""
        options: list[BufferLayoutOption] = []
        for name, buf in ir.all_buffers().items():
            if buf.location not in ("sbuf", "psum") or buf.versions != 1:
                continue
            total_tiles = buf.physical_shape()[1]
            for b in range(1, total_tiles + 1):
                if total_tiles % b == 0 and b != buf.list_len:
                    options.append(BufferLayoutOption(tensor=name, list_len=b))
        return options

    def apply(self, ir: KernelIR, option: BufferLayoutOption) -> KernelIR:
        """Re-check legality, deep-copy, set ``list_len``, rebuild the dependency sidecar."""
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        self._set_list_len(new_ir, option.tensor, option.list_len)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def _check_legality(self, ir: KernelIR, option: BufferLayoutOption) -> None:
        """Structural renderability guards only (never resource capacity)."""
        buffers = ir.all_buffers()
        if option.tensor not in buffers:
            raise TransformLegalityError(f"BufferLayout: no buffer named {option.tensor!r}")
        buf = buffers[option.tensor]
        if buf.location == "shared_hbm":
            raise TransformLegalityError(f"BufferLayout: {option.tensor} is shared_hbm (no tile axis)")
        if buf.versions > 1:
            raise TransformLegalityError(
                f"BufferLayout: {option.tensor} has versions={buf.versions} (does not compose with list_len)"
            )
        total_tiles = buf.physical_shape()[1]
        if option.list_len < 1 or total_tiles % option.list_len != 0:
            raise TransformLegalityError(
                f"BufferLayout: list_len {option.list_len} must be a positive divisor of T={total_tiles}"
            )
        if option.list_len == buf.list_len:
            raise TransformLegalityError(f"BufferLayout: {option.tensor} already has list_len={option.list_len}")

    def _set_list_len(self, ir: KernelIR, name: str, list_len: int) -> None:
        """Replace the owning block's alloc entry for ``name`` with a list_len-updated copy."""
        for nid in ir.tree.blocks():
            block = ir.tree.data(nid)
            assert isinstance(block, BlockNode)
            new_allocs = tuple(replace(b, list_len=list_len) if b.name == name else b for b in block.alloc_buffers)
            if new_allocs != block.alloc_buffers:
                ir.tree.graph.nodes[nid]["data"] = replace(block, alloc_buffers=new_allocs)


__all__ = ["BufferLayout", "BufferLayoutOption"]
```

- [ ] **Step 4: Export from the transforms package**

In `nkigym/src/nkigym/transforms/__init__.py`, add the import (alphabetical, after `base`) and the two `__all__` entries:

```python
from nkigym.transforms.buffer_layout import BufferLayout, BufferLayoutOption
```
Add `"BufferLayout",` and `"BufferLayoutOption",` to `__all__`.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
transport/ssh_host.sh --host gym-1 \
  --cmd "python -m pytest test/transforms/test_buffer_layout.py -q" \
  --cache /home/weittang/workplace/cache/buflayout_t3
```
Expected: PASS (all 8 tests).

- [ ] **Step 6: Commit**

```bash
cd /workplace/weittang/nki-autotune
git add nkigym/src/nkigym/transforms/buffer_layout.py nkigym/src/nkigym/transforms/__init__.py \
  test/transforms/test_buffer_layout.py
git commit -m "Add BufferLayout transform (re-factorize tile axis, set list_len)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Standalone k5→k6 byte-exact reproduction

Prove `BufferLayout` produces the manual k6 render: drive the canonical IR to the manual-k5 packed state, apply `BufferLayout(psum_prod → 16)`, assert the render matches `manual_transforms.kernel_6` AST-canonically. This is the transform's first byte-exact gate — isolated from the full ladder.

**Files:**
- Test: `test/transforms/test_buffer_layout.py` (add one reproduction test to the file from Task 3).

**Interfaces:**
- Consumes: `build_initial_ir` from `nkigym.ir`; `render` from `nkigym.codegen`; `Split/SplitOption`, `Reorder/ReorderOption`, `BufferLayout/BufferLayoutOption`; `assert_matches_hand` from `test.transforms._ladder_compare`; the semantic locators `_loop`, `_psum_memset_leaf` from `examples.kernel_transforms`; `f_nkigym`, `INPUT_SPECS` from `examples.kernel_transforms`; `manual_transforms.kernel_6`.
- Produces: nothing downstream — a leaf verification test.

Note on the prefix: manual k0→k5 is `Reorder, Split(K), Split(M), Reorder, Reorder`. The existing `_build_ladder` reaches the SAME pre-RFactor packed nest via `Split(K), Split(M), Reorder×6` (a different but equivalent order — it does NOT match manual rung-for-rung; that 1-to-1 ordering is Task 6). For THIS isolated test we only need a prefix whose render equals manual k5 before the layout, then confirm the layout step alone yields k6. Use the manual order so the pre-layout state is literally manual k5.

- [ ] **Step 1: Write the failing test**

Add to `test/transforms/test_buffer_layout.py`:

```python
from nkigym.codegen import render
from nkigym.ir import build_initial_ir
from nkigym.transforms import Reorder, ReorderOption, Split, SplitOption

from examples.kernel_transforms import _loop, f_nkigym, INPUT_SPECS
from examples import manual_transforms
from test.transforms._ladder_compare import assert_matches_hand


def test_k5_to_k6_reproduces_manual_kernel_6():
    """Drive canonical -> manual-k5 packed nest, apply BufferLayout(psum_prod, 16),
    and assert the render matches manual kernel_6 (the standalone '# Buffer layout' rung)."""
    ir = build_initial_ir(f_nkigym, INPUT_SPECS)
    """manual k0->k5: Reorder(K>M>N -> N-outer), Split(K->2,8), Split(M->4,4), Reorder, Reorder."""
    ir = Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d0_0"), inner_nid=_loop(ir, "i_d2_0")))
    ir = Split().apply(ir, SplitOption(target_nid=_loop(ir, "i_d0_0"), factors=(2, 8), target_axis=None))
    ir = Split().apply(ir, SplitOption(target_nid=_loop(ir, "i_d1_0"), factors=(4, 4), target_axis=None))
    ir = Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d1_0"), inner_nid=_loop(ir, "i_d1_1")))
    ir = Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d1_1"), inner_nid=_loop(ir, "i_d0_1")))
    """The pre-layout render should already equal manual kernel_5 (packed psum)."""
    assert_matches_hand(render(ir), manual_transforms.kernel_5)
    """k5->k6: BufferLayout psum_prod packed (128,16,2048) -> list-of-16."""
    ir = BufferLayout().apply(ir, BufferLayoutOption(tensor="psum_prod", list_len=16))
    assert_matches_hand(render(ir), manual_transforms.kernel_6)
```

- [ ] **Step 2: Run to verify it fails, and diagnose the prefix**

```bash
transport/ssh_host.sh --host gym-1 \
  --cmd "python -m pytest test/transforms/test_buffer_layout.py::test_k5_to_k6_reproduces_manual_kernel_6 -q" \
  --cache /home/weittang/workplace/cache/buflayout_t4
```
Expected: initially may FAIL at the `assert_matches_hand(..., kernel_5)` line if the reorder sequence above does not reach manual k5's exact loop nest. **If it fails there**, the diff printed by `assert_matches_hand` shows got-vs-want loop order; adjust the `Reorder` pair(s) until the pre-layout render matches `kernel_5`. The BufferLayout line is the invariant under test — do not change it. (This is the one place the plan cannot fully pre-compute the reorder nids, because they depend on manual k5's exact nest; the printed diff is the oracle. Manual k5's matmul nest is `N > Mo > Mi > ko > ki` per `manual_transforms.py:273-286`.)

- [ ] **Step 3: Verify it passes once the prefix matches**

Same command as Step 2. Expected: PASS — both `assert_matches_hand` calls green.

- [ ] **Step 4: Commit**

```bash
cd /workplace/weittang/nki-autotune
git add test/transforms/test_buffer_layout.py
git commit -m "Pin BufferLayout k5->k6 byte-exact vs manual kernel_6

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: List-form compaction composability

After a buffer becomes a list (`list_len > 1`), later `CodeMotion` reruns `place_buffers` + `compact_shapes` on it. Neither has ever run on a list buffer (no transform set `list_len` before this plan). Verify the two cases the manual ladder exercises: **k16** (free-axis shrink of a list `psum_prod`, `list_len=16` must survive) and a disjoint-nest case (the tile axis must NOT be collapsed). Fix `compact_shapes` ONLY if a test fails.

**Files:**
- Test: `test/transforms/test_buffer_layout.py` (add composability tests).
- Modify (only if a test fails): `nkigym/src/nkigym/codegen/compact.py`.

**Interfaces:**
- Consumes: `BufferLayout`, `CodeMotion/CodeMotionOption`, and the semantic locators from `examples.kernel_transforms`; `Buffer.per_tile_physical_shape()`.
- Produces: nothing downstream — verification (+ a possible `compact_shapes` hardening).

- [ ] **Step 1: Write the composability test**

Add to `test/transforms/test_buffer_layout.py`. The cleanest driver is to replay `_build_ladder`'s prefix up to the memset-sink CodeMotion (the manual-k16 analogue) with `psum_prod` already listed, and assert its per-tile shape stays `(128, 1, F)` with `list_len == 16` after the CodeMotion reran compaction. Reuse the ladder builder rather than re-deriving nids:

```python
from nkigym.transforms import CodeMotion, CodeMotionOption
from examples.kernel_transforms import (
    _build_ladder_prefix_through_memset_sink,  # added in Task 6; see note
)


def test_list_psum_survives_compaction_through_codemotion():
    """A listed psum_prod (list_len=16) keeps its list length and per-tile (128,1,512)
    after the memset-sink CodeMotion reruns place_buffers + compact_shapes (manual k16)."""
    ir = _build_ladder_prefix_through_memset_sink()
    buf = ir.buffer("psum_prod")
    assert buf.list_len == 16
    assert buf.per_tile_physical_shape() == (128, 1, 512)
```

**Note on the helper:** Task 6 refactors `_build_ladder` into per-rung steps; expose a small helper `_build_ladder_prefix_through_memset_sink()` that returns the IR state right after the manual-k16 memset-sink CodeMotion. If Task 6 is not yet done when executing this task, inline the equivalent step list here instead (drive canonical → BufferLayout(psum,16) at the k6 point → continue through the memset-sink CodeMotion using the locators `_psum_memset_blk`, `_loop`). Prefer the shared helper to avoid a second copy of the step list (DRY).

- [ ] **Step 2: Run it**

```bash
transport/ssh_host.sh --host gym-1 \
  --cmd "python -m pytest test/transforms/test_buffer_layout.py::test_list_psum_survives_compaction_through_codemotion -q" \
  --cache /home/weittang/workplace/cache/buflayout_t5
```
Expected: PASS — per the spec's source trace, `compact_shapes` shrinks only the free axis here (2048→512), leaving `list_len=16` valid. `place_buffers` `replace`s whole Buffers, preserving `list_len`.

- [ ] **Step 3: If (and only if) it fails — harden `compact_shapes`**

If `per_tile_physical_shape()` raises (the leading dim shrank below `list_len × a × 128`, so `list_len` no longer divides `leading//128`), the leading axis was compacted while listed. Fix in `_compact_one` (`compact.py:85-110`): when `buf.list_len > 1`, compact the leading (tile) axis in `list_len`-consistent units — i.e. never shrink the leading extent below `list_len × 128`, or shrink `list_len` alongside it. Raise loudly (no silent clamp) if the requested compaction is inconsistent with `list_len`. Add a focused unit test reproducing the failing shape, then re-run Step 2. Per the spec's trace this branch is NOT expected to fire for the manual ladder; only implement it if the test actually fails.

- [ ] **Step 4: Commit**

```bash
cd /workplace/weittang/nki-autotune
git add test/transforms/test_buffer_layout.py nkigym/src/nkigym/codegen/compact.py
git commit -m "Verify list-form buffers survive compaction through CodeMotion

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Driven k0→k26 ladder in manual order

Replace `_build_ladder` in `examples/kernel_transforms.py` with a 26-step chain in MANUAL rung order (no RFactor), inserting the 4 `BufferLayout` rungs at k6/k13/k21/k26. Wire per-rung verification: byte-exact vs `manual_transforms.kernel_i`, CPU-sim, and HW-MFU match. This is the deliverable proof of "reproduce k0→k26".

**Files:**
- Modify: `examples/kernel_transforms.py` — replace `_build_ladder` (lines 325-414); extend `_main` (lines 451+) to diff each rung vs manual and assert MFU.
- Add: `_build_ladder_prefix_through_memset_sink()` helper (consumed by Task 5).

**Interfaces:**
- Consumes: all shipped transforms + `BufferLayout/BufferLayoutOption`; existing semantic locators (`_loop`, `_psum_memset_leaf`, `_psum_memset_blk`, `_blk_loop`, `_op_leaf`, `_op_blk`, `_blk_m_loop`, `_load_leaf`, `_load_blk`, `_load_for`); `assert_matches_hand`; `manual_transforms`; `autotune.runner.profile`.
- Produces: `_build_ladder() -> list[(name, ir)]` of 27 entries (k0…k26).

- [ ] **Step 1: Map each manual rung to its driving transform**

The 26 rungs (from the spec's rung table) in manual order. Locators are semantic (track nids across structural change), reused from the existing `_build_ladder`:

| rung | transform | locator call (per existing helpers) |
|---|---|---|
| k1 | Reorder | `ReorderOption(outer=_loop("i_d0_0"), inner=_loop("i_d2_0"))` (K>M>N → N-outer) |
| k2 | Split K | `SplitOption(_loop("i_d0_0"), (2,8), None)` |
| k3 | Split M | `SplitOption(_loop("i_d1_0"), (4,4), None)` |
| k4 | Reorder | `ReorderOption(outer=_loop("i_d1_0"), inner=_loop("i_d1_1"))` |
| k5 | Reorder | `ReorderOption(outer=_loop("i_d1_1"), inner=_loop("i_d0_1"))` |
| **k6** | **BufferLayout** | `BufferLayoutOption("psum_prod", 16)` |
| k7 | Split (drain d2) | `SplitOption(_op_leaf("NKITensorCopy"), (4,512), "d2")` |
| k8 | Reorder | reorder the drain copy's `i_d1_0`/`i_d2_0` (match manual k8) |
| k9 | CodeMotion (drain sink) | `CodeMotionOption(_op_blk("NKITensorCopy"), _loop("i_d2_0"), -1)` |
| k10 | Split (store d2) | `SplitOption(_load_leaf... store leaf, (4,512), "d2")` |
| k11 | Reorder | reorder store `i_d1_0`/`i_d2_0` |
| k12 | CodeMotion (store sink) | sink the store block under `i_d2_0` |
| **k13** | **BufferLayout** | `BufferLayoutOption("sbuf_prod", 16)` |
| k14 | Split (memset d2) | `SplitOption(_psum_memset_leaf, (4,512), "d2")` |
| k15 | Reorder | reorder memset `i_d1_0`/`i_d2_0` |
| k16 | CodeMotion (memset sink) | `CodeMotionOption(_psum_memset_blk, _loop("i_d2_0"), 0)` |
| k17 | Split (rhs load d2) | `SplitOption(_load_leaf("rhs"), (4,512), "d2")` |
| k18 | Split/Reorder (rhs load) | split+reorder rhs load per manual k18 |
| k19 | Reorder | `ReorderOption` on rhs-load loops |
| k20 | CodeMotion (rhs load sink) | `CodeMotionOption(_load_blk("rhs"), _loop("i_d2_0"), 0)` |
| **k21** | **BufferLayout** | `BufferLayoutOption("sbuf_rhs", 8)` |
| k22 | Split (lhs_T load d1) | `SplitOption(_load_leaf("lhs_T"), (4,512), "d1")` |
| k23 | Split | `SplitOption(_load_for("lhs_T","i_d0_0"), (2,8), None)` |
| k24 | Reorder | `ReorderOption(_load_for("lhs_T","i_d0_1"), _load_for("lhs_T","i_d1_0"))` |
| k25 | CodeMotion (lhs_T load sink) | `CodeMotionOption(_load_blk("lhs_T"), _loop("i_d1_0"), 0)` |
| **k26** | **BufferLayout** | `BufferLayoutOption("sbuf_lhs_T", 8)` |

The Split/Reorder/CodeMotion locator calls for k7-k25 are ALREADY WRITTEN in the current `_build_ladder` (lines 348-407) — they reach the same operations, just interleaved differently (the current ladder does all matmul reorders first, then RFactor, then drain/store/load tiling). The manual-order rewrite reuses those exact locator expressions; the ONLY new steps are the 4 `BufferLayout` insertions and dropping the RFactor step. **When a locator's exact factors/axis differ from manual (e.g. k8's reorder pair), the `assert_matches_hand` diff in Step 3 is the oracle — adjust until it matches.**

- [ ] **Step 2: Rewrite `_build_ladder` as the manual-order step list**

Replace `_build_ladder` (lines 325-414). Build a `steps` list of 26 lambdas in the table order above, then:

```python
def _build_ladder() -> list[tuple[str, object]]:
    """Drive canonical f_nkigym -> manual_transforms k0..k26, ONE transform per rung,
    in MANUAL rung order (no RFactor; that is k26->k27, out of scope). The 4
    BufferLayout rungs (k6/k13/k21/k26) re-factorize psum_prod/sbuf_prod/sbuf_rhs/
    sbuf_lhs_T to their list forms; every other rung reuses the semantic locators.
    Returns [(name, ir), ...] of 27 entries k0..k26."""
    steps = [
        """k1 Reorder K>M>N -> N-outer."""
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d0_0"), inner_nid=_loop(ir, "i_d2_0"))),
        """k2 Split K -> ko(2), ki(8)."""
        lambda ir: Split().apply(ir, SplitOption(target_nid=_loop(ir, "i_d0_0"), factors=(2, 8), target_axis=None)),
        """k3 Split M -> Mo(4), Mi(4)."""
        lambda ir: Split().apply(ir, SplitOption(target_nid=_loop(ir, "i_d1_0"), factors=(4, 4), target_axis=None)),
        """k4 Reorder."""
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d1_0"), inner_nid=_loop(ir, "i_d1_1"))),
        """k5 Reorder -> matmul nest N > Mo > Mi > ko > ki."""
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d1_1"), inner_nid=_loop(ir, "i_d0_1"))),
        """k6 BufferLayout psum_prod -> list-of-16."""
        lambda ir: BufferLayout().apply(ir, BufferLayoutOption(tensor="psum_prod", list_len=16)),
        """k7..k26: see the FILL-IN note below — derive from the k7-k25 locators already
        in the pre-rewrite _build_ladder (git show HEAD~N:examples/kernel_transforms.py)
        plus BufferLayout at k13/k21/k26. Finalize each against the byte-exact diff."""
    ]
    ir = build_initial_ir(f_nkigym, INPUT_SPECS)
    ladder = [("kernel_0", ir)]
    for step in steps:
        ir = step(ir)
        ladder.append((f"kernel_{len(ladder)}", ir))
    return ladder


def _build_ladder_prefix_through_memset_sink():
    """The IR state right after the manual-k16 memset-sink CodeMotion (consumed by
    test_buffer_layout's compaction-composability test)."""
    ir = build_initial_ir(f_nkigym, INPUT_SPECS)
    for _name, ir in _build_ladder()[:17]:  # k0..k16
        pass
    return ir
```
(Simplify `_build_ladder_prefix_through_memset_sink` to `return _build_ladder()[16][1]` — the k16 IR.)

- [ ] **Step 3: Add per-rung byte-exact assertions in `_main`**

In `_main` (after building `sources`), diff each driven rung against the matching manual kernel. Add:

```python
    for name, ir in _build_ladder():
        manual_fn = getattr(manual_transforms, name)
        assert_matches_hand(render(ir), manual_fn)
        print(f"[byte-exact] {name}: OK")
```
Add `from examples import manual_transforms` and `from test.transforms._ladder_compare import assert_matches_hand` at the top. A mismatch prints the got-vs-want diff and aborts — the locator-tuning oracle for Step 1's uncertain rungs (k8/k11/k15/k18).

- [ ] **Step 4: Run byte-exact + CPU-sim on gym-1 (iterate locators until green)**

```bash
transport/ssh_host.sh --host gym-1 \
  --cmd "python examples/kernel_transforms.py" \
  --cache /home/weittang/workplace/cache/kernel_transforms
```
Expected: `[byte-exact] kernel_0..kernel_26: OK` and `[sim] ... pass=True` for all. Iterate: when a rung's byte-exact diff shows a loop-order or factor mismatch, adjust that rung's locator call and re-run. Work rung-by-rung — a divergence at kᵢ means fix step i before trusting i+1.

- [ ] **Step 5: Confirm HW-MFU matches manual per rung**

The run also profiles every rung (existing `_profile_on_hw` path). Compare the driven-ladder MFU table to `manual_transforms.py`'s (captured this session: k26 ~82%, k25 ~75%, k16-k24 ~82-85%). Byte-exact renders ⇒ identical NEFF ⇒ identical MFU; assert each runnable rung's MFU is within noise (±0.5pp) of its manual twin. The full-extent-buffer rungs (k0-k5 packed psum) are expected non-compiles — same as `manual_transforms.py`.

- [ ] **Step 6: Commit**

```bash
cd /workplace/weittang/nki-autotune
git add examples/kernel_transforms.py
git commit -m "Drive transforms to reproduce manual k0->k26 byte-exact + CPU-sim + MFU

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-review notes (for the executor)

- **Task 6 is the integration risk.** Steps 1's locator table is derived from the existing `_build_ladder` (which is verified to reach a target-equivalent) plus the 4 BufferLayout insertions, but the manual-ORDER interleaving is new: some Split/Reorder rungs (k8/k11/k15/k18) may need locator adjustment to match manual byte-for-byte. The `assert_matches_hand` diff is the oracle; budget iteration there. A rung that cannot be made byte-exact is a real finding (transform gap or a manual kernel needing normalization) — surface it, don't paper over it.
- **Do not rebuild `ir.dependency` on a moved tree** anywhere (learnings: RAW→WAR flip hides violations). `BufferLayout.apply` rebuilds it on the SAME tree (regions unchanged), which is safe.
- **`build_initial_ir` is deterministic** → nids are stable across runs, but the plan uses semantic locators throughout rather than hardcoded nids, so structural changes between rungs don't break later locators.
