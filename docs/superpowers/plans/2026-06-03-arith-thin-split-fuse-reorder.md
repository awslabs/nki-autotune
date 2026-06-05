# Arith-thin Split / Fuse / Reorder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the shipped `arith` module a production consumer by routing Split's one arithmetic legality check (the factor-cover test) through `Analyzer.const_int_bound`, mirroring TVM's `loop_transformation.cc`, while keeping all structural/HW guards and producing byte-identical output.

> **Plan revision (during execution).** The original plan had two implementation tasks. **Task 2 (Part A — route `_normalize`'s binding recompute through `iter_map_simplify`) was dropped** after the implementer discovered (and the controller independently verified) that `_recompute_bindings` rebuilds bindings canonically from scratch, making `iter_map_simplify` a guaranteed identity with an unreachable failure branch — dead ceremony. The surviving work is **Part B only** (Task 3 below). See the spec's "Part A — DROPPED" section. Task numbers are kept stable to match commit history / task tracker.

**Architecture:** One surgical change. `Split._check_legality` replaces the hand `prod(factors) != extent` cover test with TVM's `substitute_value` + `Analyzer.const_int_bound`, preserving our stricter exact-division semantics. Fuse and Reorder are unchanged (no arithmetic legality to route). `_normalize` is untouched.

**Tech Stack:** Python 3.12, `nkigym` IR (`networkx` tree), the ported `arith` substrate (`nkigym/ir/arith/`), pytest. Run everything in the `kernel-env` venv with `PYTHONPATH=nkigym/src:.`.

**Spec:** `docs/superpowers/specs/2026-06-03-arith-thin-split-fuse-reorder-design.md`

---

## Environment & Conventions

All commands run from `/home/ubuntu/nki-autotune` after:

```bash
source ~/venvs/kernel-env/bin/activate
```

Tests are invoked as (the `:.` makes the `test` package importable):

```bash
PYTHONPATH=nkigym/src:. python -m pytest <path> -q
```

**Baseline (untouched tree):** `test/transforms/ test/ir/arith/` → **102 passed, 10 skipped** (the 10 skips are TVM `importorskip` oracle tests). This count must hold after every task (plus any new tests this plan adds).

**Code-style rules that bite here** (from `.claude/rules/code_style.md`):
- Triple-quoted block comments only — **no `#` comments** (tooling directives exempt).
- Single return per function; build a result var, return at bottom.
- Modern type hints (`X | None`, `list`, `dict`).
- A `check-python-style.py` hook runs on `.py` edits outside `/tmp/`.
- Remove now-unused imports (the `math.prod` import in `split.py` becomes dead after Task 3 — delete it).

---

## File Structure

- **Modify** `nkigym/src/nkigym/transforms/split.py` — replace the two `prod(factors) != …` checks with an `Analyzer.const_int_bound`-based `_covers_exactly` helper; drop the now-unused `from math import prod`. (Task 3)
- **Modify** `test/transforms/test_split.py` — extend cover-check coverage (reject under- AND over-cover; accept exact). (Task 3)

No new modules. `_normalize.py` (Task 2/Part A — dropped), Fuse, and Reorder are untouched.

---

## Task 1: Lock the baseline (no code change)

**Files:** none (verification only).

- [ ] **Step 1: Confirm the baseline is green**

Run:
```bash
PYTHONPATH=nkigym/src:. python -m pytest test/transforms/ test/ir/arith/ -q 2>&1 | tail -3
```
Expected: `102 passed, 10 skipped` (ignore the `autoflake/isort/black ... Skipped` hook lines).

- [ ] **Step 2: Confirm the two non-TVM byte-exact gates pass individually**

Run:
```bash
PYTHONPATH=nkigym/src:. python -m pytest \
  test/transforms/test_split.py::test_split_load_d1_matches_hand_k1_byteexact \
  test/transforms/test_fuse.py::test_split_then_fuse_round_trip_byteexact -q 2>&1 | tail -3
```
Expected: `2 passed`.

These are the tripwires for "byte-identical output". Do not edit them.

---

## Task 2: ~~Route block-binding recompute through `iter_map_simplify`~~ — DROPPED (Part A)

**Status: dropped during execution. No code change. Do not implement.**

The implementer began this task and surfaced — controller independently verified
in `kernel-env` — that `_normalize._recompute_bindings` does **not** simplify the
block's existing bindings; it **rebuilds** them from scratch via
`_iter_value` → `_tile_space_affine`. Because `_dim_loops` assigns each loop var
to exactly one dim, every rebuilt binding is a single-coefficient affine over
disjoint loop vars — structurally always a valid, independent iter-map. So:

- `iter_map_simplify` would be a guaranteed **identity** (this is *why* output is
  byte-identical), and
- its `None` (non-iter-map) branch is **unreachable** — the proposed loud-raise
  test cannot pass against the real code path. It would be dead, defensive
  ceremony.

By the same "if it's always true, why need it?" reasoning that keeps Reorder
arith-free, Part A is rejected. `_normalize.py` is left untouched. The
`iter_map` machinery's intended live consumer remains Spec 2's compute_at
region-solve. See the spec's "Part A — DROPPED" section for the full finding.

## Task 3: Route Split's cover check through `Analyzer.const_int_bound` (Part B)

**Files:**
- Modify: `nkigym/src/nkigym/transforms/split.py` (imports; `_check_legality` lines 88-91 and 107-110; add `_covers_exactly` helper; drop unused `from math import prod`)
- Test: `test/transforms/test_split.py`

TVM's Split (`loop_transformation.cc:421-445`) builds `substitute_value = Σ_i var_i·Π(factor_j,j>i)` with `var_i ∈ [0, factor_i)` bound on the analyzer, then `CanProve(substitute < extent)` decides predication. We are exact-division-only, so we require the bound to be *tight and exact*: `const_int_bound(substitute).hi + 1 == extent`. This rejects both under-cover (`Π<extent`) and over-cover (`Π>extent`, which TVM would predicate).

- [ ] **Step 1: Write the failing cover tests**

The existing `test_split_rejects_factor_product_mismatch` (line ~106) only checks the `(3,5)` under-cover case. Add an over-cover case and an exact-accept case. Append to `test/transforms/test_split.py`:

```python
def test_split_rejects_over_cover():
    """Factors whose product EXCEEDS the extent are illegal (we are exact-division
    only — TVM would predicate the ragged tail; we reject). 4*5=20 > 16."""
    ir = build_canonical_ir()
    matmul_block_nid, _ = _matmul_block(ir)
    target = _first_for_under(ir, matmul_block_nid)
    assert ir.tree.data(target).extent == 16
    with pytest.raises(TransformLegalityError):
        Split().apply(ir, SplitOption(target_nid=target, factors=(4, 5)))


def test_split_accepts_exact_cover():
    """Factors whose product EQUALS the extent are legal. 4*4 == 16."""
    ir = build_canonical_ir()
    matmul_block_nid, _ = _matmul_block(ir)
    target = _first_for_under(ir, matmul_block_nid)
    assert ir.tree.data(target).extent == 16
    new_ir = Split().apply(ir, SplitOption(target_nid=target, factors=(4, 4)))
    assert new_ir is not ir
```

(`_matmul_block`, `_first_for_under`, `build_canonical_ir`, `pytest`, `Split`, `SplitOption`, `TransformLegalityError` are all already imported/defined in this test file — verify at the top before adding.)

- [ ] **Step 2: Run the new tests to verify status**

Run:
```bash
PYTHONPATH=nkigym/src:. python -m pytest \
  test/transforms/test_split.py::test_split_rejects_over_cover \
  test/transforms/test_split.py::test_split_accepts_exact_cover -q 2>&1 | tail -5
```
Expected: `test_split_accepts_exact_cover` PASSES already (exact cover is currently legal via `prod==extent`); `test_split_rejects_over_cover` also PASSES already (current `prod(4,5)=20 != 16` rejects). **Both pass on the old code** — that is fine: they pin behavior that Part B must preserve. The point of Task 3 is to keep them green while swapping the mechanism. (If you prefer a strict red-first step, temporarily break the check; not required here since these are behavior-preservation pins.)

- [ ] **Step 3: Add the `_covers_exactly` helper to `split.py`**

Replace the current arith import line in `split.py`:

```python
from nkigym.ir.arith.expr import Const, Expr
```

with one that also brings in `Add`, `Mul`, `Var`, and the `Analyzer`:

```python
from nkigym.ir.arith.analyzer import Analyzer
from nkigym.ir.arith.expr import Add, Const, Expr, Mul, Var
```

Delete the now-unused `from math import prod` line near the top of `split.py`.

Add this module-level helper (place it near the other module-level helpers at the bottom, e.g. above `_factorizations`):

```python
def _covers_exactly(factors: tuple[int, ...], extent: int) -> bool:
    """Whether ``factors`` exactly tile ``extent`` (no under- or over-cover).

    Mirrors TVM Split's mechanism (``loop_transformation.cc`` ~line 421): build
    ``substitute_value = Σ_i var_i · Π(factor_j, j>i)`` with each ``var_i`` bound
    to ``[0, factor_i)`` on an :class:`Analyzer`, then read its constant-integer
    upper bound (TVM's ``ConstIntBoundAnalyzer``). The substitution ranges over
    ``[0, Π factors)``, so its max is ``Π factors - 1``; exact tiling is
    ``hi + 1 == extent``. TVM accepts ``Π >= extent`` and predicates the ragged
    tail — we are exact-division-only (no predicate path in the renderer), so we
    require equality, rejecting both under-cover and over-cover.
    """
    analyzer = Analyzer()
    substitute: Expr = Const(value=0)
    for i, factor in enumerate(factors):
        var = Var(name=f"_split_v{i}")
        analyzer.bind(var.name, 0, factor)
        substitute = Add(left=Mul(left=substitute, right=Const(value=factor)), right=var)
    _lo, hi = analyzer.const_int_bound(substitute)
    return hi is not None and hi + 1 == extent
```

(Verified in venv: `_covers_exactly((4,4),16)=True`, `((3,5),16)=False`, `((4,5),16)=False`, `((16,128),2048)=True`.)

- [ ] **Step 4: Swap the outer-trip cover check**

In `Split._check_legality`, the outer-trip branch currently reads (lines ~88-91):

```python
            if prod(option.factors) != target.extent:
                raise TransformLegalityError(
                    f"Split.factors product {prod(option.factors)} != ForNode.extent {target.extent}"
                )
```

Replace with:

```python
            if not _covers_exactly(option.factors, target.extent):
                raise TransformLegalityError(
                    f"Split.factors {option.factors} do not exactly tile ForNode.extent {target.extent}"
                )
```

- [ ] **Step 5: Swap the tensorize cover check**

In the tensorize branch (lines ~107-110):

```python
            if prod(option.factors) != current:
                raise TransformLegalityError(
                    f"Split.factors product {prod(option.factors)} != current tensorize width {current}"
                )
```

Replace with:

```python
            if not _covers_exactly(option.factors, current):
                raise TransformLegalityError(
                    f"Split.factors {option.factors} do not exactly tile tensorize width {current}"
                )
```

- [ ] **Step 6: Confirm `prod` is fully removed**

Run:
```bash
grep -n "prod" nkigym/src/nkigym/transforms/split.py
```
Expected: **no output** (both the `from math import prod` import and both call sites are gone). If `prod` still appears, remove the remaining reference — a dangling unused import fails the style hook and the dead-code rule.

- [ ] **Step 7: Run the cover tests + full Split suite**

Run:
```bash
PYTHONPATH=nkigym/src:. python -m pytest test/transforms/test_split.py -q 2>&1 | tail -5
```
Expected: all pass, including `test_split_rejects_factor_product_mismatch`, `test_split_rejects_over_cover`, `test_split_accepts_exact_cover`, `test_split_tensorize_below_min_tile_rejected`, and `test_split_load_d1_matches_hand_k1_byteexact`.

- [ ] **Step 8: Commit**

```bash
git add nkigym/src/nkigym/transforms/split.py test/transforms/test_split.py
git commit -m "$(cat <<'EOF'
split: cover check via Analyzer.const_int_bound (Part B)

Replace hand prod(factors)==extent with TVM's substitute_value + const_int_bound
(ConstIntBoundAnalyzer), preserving exact-division-only semantics (reject under-
and over-cover). Structural/axis/MIN_TILE_SIZE guards unchanged. Drop unused
math.prod import.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Full-suite + example end-to-end verification

**Files:** none (verification only).

- [ ] **Step 1: Full transforms + arith suite green**

Run:
```bash
PYTHONPATH=nkigym/src:. python -m pytest test/transforms/ test/ir/arith/ -q 2>&1 | tail -5
```
Expected: `104 passed, 10 skipped` (baseline 102 + 2 cover tests [Task 3]; Task 2/Part A dropped, no test added there). If the number differs, account for every delta before proceeding.

- [ ] **Step 2: Wider regression — codegen + ir suites**

Run:
```bash
PYTHONPATH=nkigym/src:. python -m pytest test/ -q 2>&1 | tail -8
```
Expected: all pass/skip, no new failures vs a `git stash`-clean run. (These exercise `render` and the MDP env, which consume the transforms.)

- [ ] **Step 3: Example end-to-end (CPU-sim primary)**

Run the single example and confirm it still simulates and the dumped kernel is unchanged from `main`:
```bash
PYTHONPATH=nkigym/src:. python examples/matmul_lhsT_rhs.py 2>&1 | tail -15
```
Expected: runs to completion, sim passes (`assert_allclose` does not raise). Then diff the dumped kernel against the committed `main` reference if one exists under `/home/ubuntu/cache/`:
```bash
ls /home/ubuntu/cache/ 2>/dev/null
```
If a prior `kernel.py` artifact exists, confirm byte-identical; if not, this step is "runs clean" only.

- [ ] **Step 4: Confirm shipped `nkigym` still imports zero TVM**

Run:
```bash
PYTHONPATH=nkigym/src python -c "import sys; import nkigym.transforms; import nkigym.ir.arith; assert 'tvm' not in sys.modules, sorted(m for m in sys.modules if 'tvm' in m); print('OK: no tvm in shipped import graph')"
```
Expected: `OK: no tvm in shipped import graph`.

- [ ] **Step 5: Final style-hook sanity on the modified module**

Run:
```bash
PYTHONPATH=nkigym/src python -c "import nkigym.transforms.split; print('import OK')"
```
Expected: `import OK` (no `# comment`, unused-import, or syntax regressions; the edit hook also enforces this on save).

---

## Self-Review notes (reconciled; revised after Part A drop)

- **Spec coverage:** Part A → Task 2 **DROPPED** (rebuild-from-scratch makes `iter_map_simplify` an unreachable-failure identity); Part B → Task 3; "Fuse/Reorder unchanged" → no task (intentional, documented in spec); verification → Tasks 1 & 4.
- **Exact-division semantics:** Part B preserves our stricter `==` (not TVM's `>=`); `_covers_exactly` returns `False` for both under- and over-cover (tested).
- **Dead code:** `math.prod` removed in Task 3 Step 3/6.
- **`_normalize.py` untouched:** Part A drop means the most-depended-on file is not modified; the byte-exact gate is untouched by construction.
