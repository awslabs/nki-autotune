# Arith Substrate + Loop Primitives (Split / Fuse / Reorder) — faithful TVM port

## Status (2026-06-03)

NOT STARTED. This is **Spec 1 of 4** in a program to faithfully port TVM
TensorIR schedule transforms into the custom IR, replacing ad-hoc
per-transform arithmetic with a ported `arith` substrate.

Program decomposition (each its own spec, built on the prior):

1. **(this spec)** `arith/` substrate + `split` / `fuse` / `reorder`.
2. `compute_at` / `reverse_compute_at` (region-cover, IntSet domain solve).
3. `loop_partition`.
4. `multi_buffer` (double-buffer) + `software_pipeline`.

## Motivation

The 2026-06-03 spike (built apache/tvm main + ran it; see learnings) established:

- Online fusion is **no compiler's** transform — our contribution regardless
  of IR. Adopting TVM/TIRx was rejected (TIR→NKI backend dominates cost;
  agent-legibility). Decision: **port TVM's schedule transforms into the
  custom IR, mirroring the tvm source, not inventing hacks.**
- The transform instability we keep paying (compute_at multi-day churn,
  element-vs-tile-space bugs, double-parent) traces to **one root cause**: we
  re-derive TVM's `arith` substrate (region-cover, integer-set domain solve,
  affine binding simplification) thinly and buggily *under each transform*,
  instead of calling one shared, tested substrate.

Measured `arith` footprint of the 8 target transforms: a **focused shared
core** (`Analyzer.Simplify` / `CanProve*`, `IntSet` Union/Intersect/EvalSet,
`IterMap` detect/simplify, `Substitute`) — not the full ~8.2k-line `src/arith/`.
Split/Fuse/Reorder hit the smallest slice, so they validate the substrate first.

TVM's `Split` (`src/s_tir/schedule/primitive/loop_transformation.cc:396`) is the
reference shape: `substitute_value = Σ var_i·factor_i`; `analyzer.Bind` each
new var; substitute old loop_var; **predicate guard** `substitute_value <
extent` unless `analyzer.CanProve(...)` elides it; regenerate nested `For`s;
`IterMapSimplifyBlockBinding::SimplifyBindings` cleans the bindings. Our current
`Split` lacks the predicate path and assumes clean division — exactly the
generality the `arith` substrate supplies.

## Goal

Ship a tested `arith/` subsystem and rebuild `split` / `fuse` / `reorder` as
**thin clients** of it, gated byte-exact against the hand ladder AND against
TVM run as a live oracle. Remove the bespoke arithmetic these three transforms
inline today.

## Architecture

### New module tree — `nkigym/src/nkigym/ir/arith/`

Each file mirrors a TVM `src/arith/` source file; populated **demand-driven**
(only the rules Split/Fuse/Reorder exercise), but each rule a **faithful
transcription** of TVM's, never an invention.

```
ir/arith/
  expr.py               # MOVED + EXTENDED from ir/expr.py: add Sub, Min, Max,
                        #   predicate nodes (LT, LE, EQ); non-affine subterms
                        #   carry OPAQUELY (stop raising NonAffineError).
  analyzer.py           # Analyzer facade — mirror src/arith/analyzer.cc:
                        #   Simplify(expr, steps), CanProve(pred, strength),
                        #   CanProveEqual, CanProveGreaterEqual, CanProveLess,
                        #   Bind(var, range). Dispatches to the simplifiers.
  rewrite_simplify.py   # RewriteSimplifier — mirror src/arith/rewrite_simplify.cc:
                        #   per-node VisitExpr_ rewrite rules + TryCompare +
                        #   EnterConstraint. Only the rule subset the three
                        #   transforms hit (Add/Sub/Mul/FloorDiv/Mod/LT canon +
                        #   const-bound compare).
  canonical_simplify.py # CanonicalSimplifier — mirror src/arith/canonical_simplify.cc:
                        #   sum-of-products canonical form. Minimal subset here
                        #   (heavy use is compute_at / Spec 2).
  iter_map.py           # mirror src/arith/iter_affine_map.cc: DetectIterMap,
                        #   IterMapSimplify, NormalizeIterMapToExpr — the loop-
                        #   binding affine analysis Split/Fuse/Reorder use.
  int_set.py            # mirror src/arith/int_set.cc: IntSet (Interval, Union,
                        #   Intersect, EvalSet). Subset here; region-solve is Spec 2.
```

### Boundary (mirrors TVM's layering)

`ir/arith/` depends on **nothing above it** — operates purely on `Expr` and
`Var`→range bindings. `transforms/` and the rest of `ir/` depend on `arith`,
never the reverse. This is TVM's invariant (`arith` is foundational) and keeps
import direction one-way.

### Expr representation

`ir/expr.py` is **moved into `ir/arith/expr.py`** and extended. Current nodes
(`Const, Var, Add, Mul, FloorDiv, Mod`) are kept; add `Sub, Min, Max` and
predicate nodes (`LT, LE, EQ`) needed for `CanProve` over the split predicate.
The current `to_affine`/`from_affine` closed-form collapse is **retained as a
fast path inside the canonical simplifier**, but the public simplifier no longer
*requires* affinity: a non-affine subterm is carried opaquely (so a predicate
like `i0*512 + i1 < 2048` is representable) rather than raising `NonAffineError`.

All current `ir/expr.py` importers are repointed to `ir/arith/expr.py` via a
re-export shim in `ir/__init__.py` (no call-site churn); the old module file is
deleted.

## Verification — TVM as a live oracle (the new capability)

Two layers; the hand ladder stays the **primary** correctness gate.

### Layer A — arith-level oracle (unit, in-process, fully clean)

`tvm.arith` is pure in-process Python — no NKI codegen, no version skew. For
each ported primitive, run the same query through real TVM and assert equality:

- `Simplify`: build `Expr` → bridge to `tvm.tir.PrimExpr` → `tvm.arith.Analyzer().simplify` → compare canonical forms.
- `CanProve` / `CanProveEqual` / bounds: same expr + bindings → assert same bool.
- `IterMapSimplify` / `DetectIterMap`: same indices + var ranges → assert same simplified result.

A `test/`-only `expr ↔ PrimExpr` bridge is the only glue; the TVM dependency
**never** enters the shipped `nkigym/` package.

**Verified oracle entry points** (built TVM at `/home/ubuntu/tvm`, run
`TVM_LIBRARY_PATH=/home/ubuntu/tvm/build/lib PYTHONPATH=/home/ubuntu/tvm/python`):

```python
import tvm.tirx as T            # Var, Add, Mul, FloorDiv, FloorMod, IntImm — build PrimExprs
from tvm import arith, ir
a = arith.Analyzer()
a.bind(x, ir.Range(0, 128))     # var range
a.simplify(expr)                # -> simplified PrimExpr   (e.g. (x*512+3)%512 -> 3)
a.can_prove(pred)               # -> bool                  (e.g. x<128 -> True)
arith.detect_iter_map(...)      # IterMap analysis
arith.IntSet                    # interval / union / intersect
```

Confirmed working in the built fork (not aspirational): `simplify((x*512+3)%512)
== 3`, `can_prove(x < 128) == True`, `detect_iter_map` callable.

### Layer B — transform-level oracle (integration, structural)

Express the same schedule in both systems and diff the **TIR loop structure**
(stable, in-process) — NOT the NKI source (the fork's `codegen_trn` is version-
skewed against the installed `neuronxcc`, so NKI-source diff is not byte-clean):

- Ours: `Split().apply(ir, opt)` → loop nesting + extents + binding exprs.
- TVM: `sch.split(loop, factors)` → same.
- Assert structural correspondence.

### Gates (both required; strictness unchanged)

1. **Hand-ladder byte-exact** — `kernel_transforms.py` render+sim vs numpy
   golden; the existing AST-canonical oracle. Primary; "same means same."
2. **TVM oracle** — Layer A on every arith primitive; Layer B on each transform.
   A TVM/ours disagreement is a **HARD STOP**: port the missing TVM rule; never
   loosen the gate or paper over the divergence.

## Phasing

### Phase 0 — expr extension + TVM bridge
- Move `ir/expr.py` → `ir/arith/expr.py`; add `Sub/Min/Max/LT/LE/EQ`; opaque
  non-affine carry; re-export shim in `ir/__init__.py`; delete old `ir/expr.py`.
- Build `test/`-only `expr↔PrimExpr` bridge.
- Gate: round-trip `expr→PrimExpr→expr` identity on a generated corpus.

### Phase 1 — arith core
- Port `analyzer.py`, `rewrite_simplify.py` (rule subset), `iter_map.py`,
  `canonical_simplify.py` (subset), `int_set.py`.
- Gate: Layer-A oracle green on a generated corpus for every primitive.

### Phase 2 — split + reorder (thin clients)
- Rewrite `split.py` mirroring TVM `Split` (arbitrary factors + predicate
  guard via `analyzer.CanProve`; nested-loop regen; `IterMapSimplify` binding
  cleanup). Rewrite `reorder.py` as a binding-resimplify client.
- Gate: hand-ladder byte-exact + Layer-B structural oracle.
- Remove: the affine-collapse assumptions and inline arithmetic in
  `split.py` / `reorder.py`.

### Phase 3 — fuse (thin client)
- Rewrite `fuse.py` mirroring TVM `Fuse` (the `IterMapSimplify` inverse-collapse
  proving `fused//f, fused%f` recover the original vars).
- Gate: hand-ladder byte-exact + Layer-B structural oracle.
- Remove: inline arithmetic in `fuse.py`.

## Removals (no compat shims)

- `ir/expr.py` — moved into `ir/arith/expr.py` (re-export shim, then file gone).
- Inline affine/division assumptions inside `split.py` / `fuse.py` / `reorder.py`
  — replaced by `arith` calls.

**NOT removed in Spec 1** (verified by import audit — still load-bearing):
- `transforms/_domain_solve.py` — imported only by `_code_motion.py` (compute_at
  mover); belongs to Spec 2. Untouched here.
- `ir/interval.py` — imported only by `ir/dependency.py`. Its disjointness logic
  is subsumed by `int_set.py` eventually, but `dependency.py` still calls it;
  folding it in is **Spec 2** (when compute_at's IntSet region-solve lands).
  Untouched here.

## Out of scope (explicit)

- `compute_at` / `reverse_compute_at` — keep their current (unstable)
  implementation through Spec 1; this spec does not touch them. **Honest
  consequence:** the MDP example's compute_at instability persists until Spec 2.
  Spec 1's "runs clean" claim covers split/fuse/reorder rollouts only.
- `loop_partition`, `multi_buffer`, `software_pipeline` — later specs.
- `arith` rules those need but Split/Fuse/Reorder don't (canonical_simplify
  depth, IntSet region-solve, presburger / linear-inequality solve,
  transitive-comparison) — ported on demand in their specs.

## Decisions captured

- **Port the shared arith core FIRST, transforms as thin clients** — not each
  transform with its own inline arithmetic (that is the current root-cause
  instability). Rejected: transform-by-transform with demand-pulled arith
  (risks re-fragmenting arith unless disciplined; the shared-core-first order
  removes the risk structurally).
- **Faithfully mirror TVM's rewrite-rule simplifier** (rewrite_simplify +
  canonical_simplify + iter_affine_map), bit-identical to TVM, NOT our
  closed-form affine-dict collapse. Rejected the affine-dict-extension path:
  proven insufficient (it is why Split assumes clean division and can't emit
  predicates). The affine collapse survives only as a fast path inside the
  canonical simplifier.
- **Demand-driven rule coverage, TVM-oracle-arbitrated** — mirror TVM's
  structure exactly; populate only the rules the three transforms exercise;
  the oracle mismatch names the next rule to port. Empirically complete for
  our transforms, never speculative.
- **TVM as a live oracle** — Layer A (arith, in-process, clean) + Layer B
  (transform, TIR-structural not NKI-source, to dodge the codegen version skew).
  Hand ladder stays primary; oracle disagreement is a HARD STOP.
- **First spec = arith core + split/fuse/reorder** — right-sized, end-to-end
  testable, replaces three existing transforms. Rejected arith-core-alone
  (ships nothing exercising the substrate end-to-end).
- **Easy-to-hard port order** — split/fuse/reorder validate the substrate before
  the IntSet-heavy compute_at (Spec 2) leans on it.
