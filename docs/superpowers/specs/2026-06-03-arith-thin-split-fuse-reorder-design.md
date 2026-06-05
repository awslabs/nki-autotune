# Arith-thin Split / Fuse / Reorder — route legality through the shipped `arith` module

*Spec 1.5 of the TVM port (between Spec 1 "arith substrate shipped" and Spec 2 "compute_at region-solve").*

## Motivation

The `arith` substrate (`nkigym/ir/arith/`) shipped in Spec 1 as a faithful TVM
`src/arith/` port — `Analyzer`, `iter_map`, `IntSet` — validated against a built
TVM as a live oracle. But it has **no production consumer**: every caller is a
test. Its intended live consumers are the schedule transforms.

This task makes **Split** a genuine production consumer of `arith`, mirroring
where TVM's `loop_transformation.cc` calls `arith::Analyzer` for the
factor-cover check. It is **refactor-for-validation, not capability**: the
rendered output is byte-identical, so the win is that Split's one piece of
*arithmetic* legality (the factor-cover test) now flows through the
oracle-validated `Analyzer` instead of an inline `math.prod` comparison.

> **Outcome note (post-implementation).** What shipped is **Part B only**. Part A
> — routing `normalize_block`'s binding recompute through `iter_map_simplify` —
> was dropped during implementation: `_recompute_bindings` rebuilds bindings
> canonically from scratch, so `iter_map_simplify` would be a guaranteed identity
> with an unreachable failure branch (dead ceremony). See the Part A section for
> the full finding. `iter_map` therefore still has no production consumer; its
> intended live consumer remains Spec 2's compute_at region-solve.

## Finding: the transforms are already arith-thin

Split / Fuse / Reorder are pure tree-surgery. All affine work already delegates
to `_normalize.normalize_block` (our analog of TVM's
`IterMapSimplifyBlockBinding`), whose single raw-affine builder is
`_tile_space_affine` (`Σ_j loop_j · Π(inner extents)`). It already imports
`from_affine` / `to_affine` from `arith.expr`.

Every current `_check_legality` predicate is **structural**, **hardware**, or
**role** — *none* is affine:

| Transform | Legality predicates | Kind |
|---|---|---|
| Split | `len(factors)>=2`; entries `>=2`; `isinstance ForNode/ISANode`; `prod(factors)==extent`; axis declared by block; innermost `>= MIN_TILE_SIZE` | structural + **arithmetic** (the `prod==extent` cover) + HW |
| Fuse | `len>=2`; nids in graph; `isinstance ForNode`; sole-child chain | structural |
| Reorder | nids in tree; `isinstance ForNode`; sole-child; no descendant `SEQUENTIAL` on either swap axis | structural + role |

So there is **no obsolete affine legality for arith to delete**. The literal ask
("remove the custom legality checks") is reconciled as: route the one piece of
*arithmetic* legality — Split's factor-cover check — through `arith`; keep the
structural / HW / role guards, exactly as TVM keeps its analogous explicit
guards (`HasAnnotationOrThreadBindingError`, `NotOnlyChildError`,
`OuterNotInnerParent`). The only arithmetic predicate in the whole table is
Split's `prod(factors)==extent` cover (Part B). Fuse and Reorder have **no**
arithmetic legality at all, so neither has anything to route.

## Scope decision: faithful TVM mirror, narrowed to Part B

Chosen over "binding-reroute-only" and "strip all legality" (rejected — deletes
real non-affine safety, violates *loud-failures-only*). Mirror TVM at each point
it invokes the analyzer for *legality*; leave everything else. After the Part A
finding (below), the concrete surviving change is **Part B alone** — Split's
cover check.

### Part A — DROPPED (was: binding recompute via `iter_map_simplify`)

**Original intent.** TVM's `IterMapSimplifyBlockBinding::VisitStmt_(SBlockRealizeNode*)`
(`loop_transformation.cc:124`) runs `arith::IterMapSimplify(op->iter_values,
input_iters, …)` on every block realize touched by a Split/Fuse. Part A proposed
to mirror that in `_normalize._recompute_bindings`: build the per-axis bindings,
run them through `iter_map_simplify(bindings, input_iters)`, use the result, and
raise `TransformLegalityError` loudly on a `None` (non-iter-map) result.

**Why it was dropped (discovered during implementation, verified in `kernel-env`).**
Part A rests on a false premise about how `_recompute_bindings` works. It does
**not** simplify the block's *existing* bindings — it **rebuilds them from
scratch** via `_iter_value` → `_tile_space_affine` (`Σ loop · Π inner-extents`).
Because `_dim_loops` partitions every loop var into exactly one dim, each rebuilt
binding is a single-coefficient affine over disjoint loop vars — **structurally
always a valid, independent, surjective iter-map**. Consequences:

- `iter_map_simplify` is a guaranteed **identity** on every rebuilt binding (this
  is *why* the byte-exact output is preserved — confirmed across canonical,
  post-Split `i0*4+i1`, post-Fuse, and compute_at-moved blocks).
- Its `None` branch is therefore **unreachable** from any real
  `_recompute_bindings` input. The proposed "new loud invariant" can never fire;
  the loud-raise test cannot pass against the real code path. It would be dead,
  defensive ceremony.

This is the divergence from TVM: TVM runs `IterMapSimplify` on *substituted*
bindings (e.g. `floordiv(f,4)*4 + f%4` after a split substitutes the loop var),
where canonicalization does load-bearing work and can fail. We never build that
messy intermediate — we regenerate the canonical form directly — so there is
nothing to simplify or reject. Adding the call would make `_normalize` a *visible*
`iter_map` consumer at the cost of a real dependency + byte-gate risk for **zero**
behavior change and an unfireable guard. By the same "if it's always true, why
need it?" principle applied to Reorder, Part A is rejected. (Note: `_normalize`
already consumes `arith.expr` — `from_affine` / `to_affine` — so it is not
arith-free; it simply has no higher-level arith work to route.)

Making `iter_map_simplify` genuinely load-bearing would require restructuring
`_recompute_bindings` to TVM's keep-original-then-substitute-then-simplify shape —
a large rewrite of the most-depended-on file, out of scope here and high-risk to
the byte gate. Deferred (not planned).

### Part B — Split's cover check via `Analyzer.const_int_bound`

TVM's Split (`loop_transformation.cc:421-445`) builds
`substitute_value = Σ_i var_i · Π(factor_j, j>i)` with each `var_i ∈ [0,
factor_i)` bound on the analyzer, then forms `predicate = substitute_value <
loop->extent` and, **only if it cannot `CanProve` the predicate**, appends a
ragged `BlockPredicate`. TVM thus accepts `Π factors >= extent` (its
`WrongFactorError` rejects only `<`) and predicates the tail.

**We are exact-division-only** — `_factorizations` emits only exact divisors and
no predicate-elision path exists in the renderer. So we must reject *both*
`Π < extent` (under-cover) and `Π > extent` (over-cover, which TVM would
predicate). Mirror TVM's mechanism while keeping our stricter semantics:

- Build the same `substitute_value` affine with `var_i ∈ [0, factor_i)` bound on
  an `Analyzer`.
- `lo, hi = analyzer.const_int_bound(substitute_value)` — this is TVM's
  `ConstIntBoundAnalyzer`; `hi == Π factors - 1`.
- Require `hi + 1 == extent`. Equivalent to `Π factors == extent`, i.e. the
  predicate `substitute < extent` is simultaneously tight (no over-cover) and
  exactly covering (no under-cover).

Applies to **both** Split flavours (outer-trip `extent` = `ForNode.extent`;
tensorize `extent` = current tile width). Replaces the inline `prod(...) !=
extent` comparison in both `analyze` gating and `_check_legality`. The
`len(factors)>=2`, `isinstance`, axis-declared, and `MIN_TILE_SIZE` guards
**stay** — none is affine.

### Reorder — no change (nothing to route)

Reorder is pure payload-swap; it does not call `normalize_block` and has no
arithmetic predicate. The two places TVM's Reorder uses the analyzer are both
**vacuous on our IR**, so adding them would be dead ceremony (contradicting the
substrate's own "else it's dead weight" rule):

- `CheckBlockIterTypeAndAffineBinding` (affine half): can fire **in TVM** because
  TVM blocks may carry non-affine bindings. On our IR every binding comes from
  `_tile_space_affine` and is affine by construction, so `detect_iter_map` can
  never return `None` here.
- `ConstructNewLoopChain`'s outer-not-dependent-on-inner check: tests whether an
  inner loop's *extent* uses an outer loop var. Our `ForNode.extent` is a plain
  `int`, never an `Expr`, so it can never depend on another loop var.

Reorder is therefore already maximally arith-thin: its only legality is the
structural perfect-nest and the `SEQUENTIAL`-role guard, both of which stay.

## Out of scope

- `transforms/_domain_solve.py` — its own affine builder belongs to compute_at /
  ReverseComputeAt (Spec 2's region-solve), the substrate's other intended live
  consumer. Not touched here.
- The documented pre-existing "Split-over-a-shared-loop" over-accumulation bug —
  structural, unrelated to arith. Verify this change does not perturb it; do not
  attempt to fix it here.
- `Const`-fold / canonicalization behavior of the simplifier — unchanged.

## Verification

1. **Baseline green first**: `pytest test/transforms/ test/ir/arith/` on the
   untouched tree → `102 passed, 10 skipped`.
2. After the change, the **same** suite stays green, with special attention to
   the byte-exact gates — these must be byte-for-byte unchanged:
   - `test_split.py::test_split_load_d1_matches_hand_k1_byteexact`
   - `test_fuse.py::test_split_then_fuse_round_trip_byteexact`
   - `test_tvm_struct_oracle.py` and `test_split.py::test_*_matches_tvm_structure`
     (skipped without TVM, but run where available)
3. **New Split-cover coverage**: extend the existing `prod`-mismatch rejection
   test to confirm the `const_int_bound`-based check rejects both `Π < extent`
   and `Π > extent`, and accepts `Π == extent`.
4. Run `examples/matmul_lhsT_rhs.py` end-to-end (CPU-sim primary) and confirm the
   dumped `kernel.py` is unchanged from `main`.

## Risk

Part B is **low-risk and self-contained**: it touches only `Split._check_legality`
(the cover comparison), changing the *mechanism* of an arithmetic check whose
*verdict* is identical on every input (`const_int_bound(substitute).hi+1` equals
`prod(factors)` exactly). No IR-output path changes, so the byte-exact gate is
untouched by construction. The dropped Part A was the high-risk piece (it touched
`_normalize.py`, the most-depended-on file); removing it from scope removes that
risk entirely.
