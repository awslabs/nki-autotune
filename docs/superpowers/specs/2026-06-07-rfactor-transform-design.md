# RFactor Transform — TVM-Faithful Reduction Split

*Design date: 2026-06-07 (rewritten 2026-06-09 to be TVM-faithful)*

> **Supersedes the `## 7. RFactor atom` section of
> `2026-05-09-first-class-buffers-and-rfactor-design.md`.** That spec predates the
> BlockNode IR (2026-05-27) and the SSA-ops backend (2026-06-01): its
> `NKIAlloc` / `BodyLeaf` / `module.tensors` / `LoopNode` vocabulary no longer
> exists. This design keeps that spec's surviving contributions — the
> `RFACTOR_RECIPE = "rmw" | "slot"` taxonomy (which shipped on the ops) and the
> legality conditions — and re-expresses the atom on the current IR.

> **Rewrite note (2026-06-09).** An earlier draft of this spec had RFactor emit the
> *fused single-accumulator* form directly. Per the directive to make RFactor a
> faithful port of `tir.Schedule.rfactor`, and confirmed by source audit of TVM
> (`reduction.cc`; see `.claude/rules/tvm_knowledge.md`), **the RFactor atom now
> emits TVM's exact terminal output: a multi-slot rf-buffer + a separate write-back
> block, with the factored loop role-flipped but loops NOT reordered.** The fused
> single-accumulator that matches `kernel_hand_90.92mfu.py` is reached *downstream*
> by composing shipped transforms (ComputeAt + `compact_shapes`); see §7. That fold
> depends on narrowing `_check_no_reduction_axis_covered`, which is **parked** (the
> block-granular / region_cover investigation; see §8 and the learnings).

## Decisions (at-a-glance)

1. **RFactor is a new rewrite transform** under `nkigym/transforms/`, subclassing
   `Transform` with the shipped `analyze`/`apply` contract (deep-copy, mutate,
   re-check legality, loud `TransformLegalityError`).
2. **The atom is a faithful port of `tir.Schedule.rfactor`.** It introduces a
   *second reduction level* over an outer reduction loop `ko` (produced by a prior
   `Split` of a reduction axis) by the exact three edits TVM makes (§3.4). One
   mechanism only.
3. **The output form is TVM's terminal rf-buffer + write-back block** — NOT a fused
   accumulator. RFactor allocates a `[factor, *shape(out)]` rf-buffer `B_rf`, emits
   an **rf-block** that fills it (factored loop flipped reduction→parallel), and a
   separate **wb-block** that reduces `B_rf` over the factored axis into `out`.
   `factor` slots stay live; the loop nest is **not reordered** (verified:
   `CreateLoopOutsideRfactorBlock` reuses original loop order). This is byte-for-byte
   the shape of TVM's `matmul_rfactor` test output.
4. **The fused single-accumulator is downstream, not the atom.** TVM keeps the
   `factor` slots because the factored axis binds to concurrent GPU threads; on
   Trn2's single Tensor Engine the slots are sequential waste, so we *fold* them to
   one reused accumulator — but that fold is ComputeAt + `compact_shapes` applied
   *after* RFactor (§7), not part of the atom. Keeps each transform a faithful,
   single-mechanism port.
5. **The op declares its reducer** (combiner + identity) via a new `REDUCE_COMBINATOR`
   class attribute (mirrors TVM's `CommReducer`). RFactor reads it to synthesize the
   rf-block's init + accumulate and the wb-block's init + combine. New infrastructure
   — `REDUCE_COMBINATOR` does **not** exist in `nkigym/` today.
6. **Two recipes, dispatched on `op_cls.RFACTOR_RECIPE`** (already shipped on the
   ops): `"rmw"` (matmul) and `"slot"` (activation_reduce). Both emit the same
   rf-buffer + wb-block shape; they differ only in how the rf-block's per-slot
   accumulate lowers (HW `+=` into PSUM vs. staged SBUF). `"rmw"`/matmul is the
   implementation target; `"slot"` is specified for symmetry but deferred.
7. **Legality = correctness + dep-order + ISA well-formedness only; never resource
   capacity.** The `[factor, *shape(out)]` rf-buffer is large and may over-subscribe
   SBUF — that is still a *legal* RFactor output; compile/HW profiling prunes it, and
   the downstream fold (§7) is what makes it fit. (User-locked rule.)

## Invariants

- Same reducer at both levels: `Σ_k = Σ_ko (Σ_ki)`. The legality of the split
  rests entirely on the reducer being associative and commutative.
- The factored loop `ko` is **role-flipped to `PARALLEL` in the rf-block** (it now
  indexes a distinct rf-buffer slot, no carried state) and is **`ACCUMULATION` in
  the wb-block** (it carries the reduction of slots into `out`) — **one `ForNode`,
  two blocks, two roles** (the per-block-`IterVar.role` property the IR supports).
- `ki` (inner) keeps role `ACCUMULATION` in the rf-block (unchanged).
- The rf-block's init (memset `B_rf[ko,...]` to identity) dominates its `ki`
  reduction; the wb-block's init (memset `out` to identity) dominates its `ko`
  reduction. Both are sibling memset blocks (our post-`decompose_reduction` form),
  expressed via the shipped `memset → loop` carry edge.
- The loop nest is **NOT reordered** by RFactor — the factored axis stays in its
  original position; only its iter_var *role* changes (verified against TVM
  `CreateLoopOutsideRfactorBlock`, which reuses original loop order).
- Block *count* is determined by RFactor's deterministic emission (it *adds* the
  wb-block + the two memset siblings), not by a per-instruction heuristic — mirroring
  how `canonical_build` mechanically emits one block per op plus a memset sibling.

## Rejected alternatives

- **Emit the fused single-accumulator directly from RFactor** (an earlier draft of
  this spec). Rejected for faithfulness: it bundles RFactor + ComputeAt-fold +
  storage-collapse into one bespoke macro, diverging from `tir.Schedule.rfactor`'s
  single mechanism. TVM emits the terminal rf-buffer + wb-block and reaches any fused
  form by *composing* later primitives; we do the same (§7), keeping each transform a
  faithful single-mechanism port. (This mirrors the SoftwarePipeline precedent:
  faithful mechanism, single-target specialization expressed by composition, not by
  fattening the atom.)
- **Reorder alone to get `ko` outside `M`.** Reorder is storage-preserving — it
  payload-swaps adjacent ForNodes and cannot create the rf-buffer + wb-block that
  rfactor introduces. (See §1.)

> **On the Trn2 vs GPU difference.** TVM keeps the `factor` rf-buffer slots alive
> on purpose: the factored axis binds to concurrent GPU threads, each owning a slot
> (spatial/concurrent-writer parallelism). Trn2 has one Tensor Engine, so the slots
> are produced sequentially and are pure waste — which is *why* we fold them
> downstream (§7). But that is a property of our *target*, expressed by a later
> transform; it does **not** justify changing the rfactor *atom*, which stays
> faithful. (Verified terminal: TVM has no fuse-back primitive — `reduction.cc`,
> `meta_schedule/`, `dlight/`, tests; see `.claude/rules/tvm_knowledge.md`.)

## 1. Motivation

### The 83% → 90% gap

The shipped transforms peak at a **83.4% MFU** matmul (Tier-B `SoftwarePipeline`
double-buffering the accumulator). HW measurement shows this is not a matmul-
efficiency limit: at 83.4% the operands are already SBUF-resident (DMA only ~17.6%
saturated), so further `SoftwarePipeline` / `ComputeAt` prefetch buys nothing. The
gap to the 90.92% hand kernel (`kernel_library/matmul/lhsT_rhs/kernel_hand_90.92mfu.py`)
is **structural**: with a single-level K reduction, the `rhs` operand (which depends
on K and N, *not* M) is trapped *under* the M loop and reloaded per M block instead
of loaded once and reused.

### Why the single accumulator forces M-outside-K

A reduction's accumulator must live in one memory space for the *entire* reduction.
For matmul that space is **PSUM** — the HW accumulator *is* PSUM, ~2 MiB. So a
single-level reduction hard-couples three things:

- **reduction depth** → PSUM residency (depth-3 already OOMs: 3 MiB > 2 MiB, HW prunes)
- a single PSUM accumulator for `(m, n)` must accumulate over all of K contiguously
  → **M must nest outside K** (K-outside-M would need M live PSUM banks at once)
- with M outside K, the `rhs` load sits inside the M loop → **reloaded M times, no reuse**

So the chain is: *single PSUM accumulator ⟹ M-outside-K ⟹ rhs trapped under M ⟹ no
reuse.* No loop-reordering transform breaks it, because the constraint is on
*storage*, not loop order.

### Why this needs RFactor specifically, not Reorder

Getting `ko` outside M is a **dataflow/storage change**: the single full-K
accumulator must become per-`ko` partials that are recombined. `Reorder` is
storage-preserving (payload-swaps adjacent ForNodes); it cannot create the partial
buffer or the recombination block. The measured legal reorders confirm this — M>K>N
gives 83.06 vs 83.00 (the compiler already reuses the stationary weight), and
N-outermost is blocked. RFactor is the missing primitive: it materializes the
per-`ko` partials (the rf-buffer) and the recombination (the wb-block), which is what
ultimately frees `ko` to sit outside M.

Note the full SOTA shape needs *two* things RFactor enables but does not itself do:
the **fold** of the multi-slot rf-buffer to one accumulator (§7) and the **hoist** of
`ko` outside M. RFactor's job is solely to introduce the rf-buffer + wb-block; the
fold and hoist are downstream.

### Scope of this design

This spec covers the **RFactor atom (a faithful `tir.Schedule.rfactor` port) and the
op-side infrastructure it needs** — nothing else. The follow-on transforms toward
90.92% (the §7 fold via ComputeAt + `compact_shapes`; a hoist of `ko` outside M;
`SoftwarePipeline` double-buffering) are *referenced* as the downstream path but
specified and verified separately. RFactor's job is solely to introduce the second
reduction level in TVM's terminal form.

## 2. The reduction-split, abstractly

Independent of any op, RFactor implements this rewrite over a reduction loop:

```text
Before (one accumulator over the whole reduction):
    B = identity                       # init
    for k:                             # ACCUMULATION
        c = compute(k)
        B = reduce_fn(B, c)            # carried RAW on B

After Split(k -> ko, ki) — role-preserving, still one accumulator:
    B = identity
    for ko:                            # ACCUMULATION
        for ki:                        # ACCUMULATION
            c = compute(ko, ki)
            B = reduce_fn(B, c)

After RFactor(ko) — TVM's terminal form (rf-buffer + write-back block):
    # rf-block: ko flipped to PARALLEL; writes its own slot of B_rf
    for ko:                            # PARALLEL (indexes a distinct B_rf slot)
        B_rf[ko] = identity            # rf-block init
        for ki:                        # ACCUMULATION (unchanged)
            c = compute(ko, ki)
            B_rf[ko] = reduce_fn(B_rf[ko], c)
    # wb-block: separate block; reduces the slots into the original output
    out = identity                     # wb-block init
    for ko:                            # ACCUMULATION (the closing reduction)
        out = reduce_fn(out, B_rf[ko])
```

`B_rf` is `[factor, *shape(out)]` — **all `factor` slots stay live**, and the
write-back is a **separate block**, exactly `tir.Schedule.rfactor`. The factored
loop keeps its original nest position (no reorder); only its role changes
(reduction in `out`'s accumulation → parallel over `B_rf` slots in the rf-block,
reduction again in the wb-block).

The same `reduce_fn` (with the same identity) appears at both levels — that
recursion *is* the transform. Correctness rests on `reduce_fn` being associative
and commutative: `reduce_fn` over all `k` equals `reduce_fn` over `ko` of the
per-`ko` `reduce_fn` over `ki`.

> **Why the multi-slot buffer (not a fused single accumulator)?** This is TVM's
> terminal output, kept faithful. On Trn2 the `factor` slots are sequential waste
> (one Tensor Engine), so the **fold to a single reused accumulator is a downstream
> step** (ComputeAt + `compact_shapes`, §7) — *not* baked into this atom. Folding
> here would make RFactor a bespoke macro; keeping it terminal makes it a faithful
> port, and the fold composes cleanly afterward.

## 3. The RFactor atom (matmul / `"rmw"`)

### 3.1 Concrete IR — before and after

Matmul, with K already `Split` into `ko × ki`; M/N spatial wrappers and concrete
node ids omitted for clarity. `factor` = `ko`'s extent.

**Before** (`ki`, `ko` both `ACCUMULATION`; one `psum` accumulator):

```text
memset(psum)                                  # init block
for ko (ACC):
  for ki (ACC):
    nc_matmul(dst=psum, stationary=lhs[ko,ki], moving=rhs[ko,ki])   # HW +=
tensor_copy(out_sbuf <- psum)                 # drain block, write-once
```

**After `RFactor(ko)`** — TVM's terminal rf-buffer + wb-block (no reorder):

```text
# rf-block: ko flipped to PARALLEL; psum_rf carries the factored slot index
for ko (PAR):
  memset(psum_rf[ko])                          # rf-block init (per slot)
  for ki (ACC):
    nc_matmul(dst=psum_rf[ko], stationary=lhs[ko,ki], moving=rhs[ko,ki])  # HW +=
  tensor_copy(B_rf[ko] <- psum_rf[ko])         # drain this slot to the rf-buffer (SBUF)
# wb-block: separate; reduces the factor slots into out_sbuf
memset(out_sbuf)                               # wb-block init
for ko (ACC):
  tensor_tensor(out_sbuf <- out_sbuf, B_rf[ko], op=add)   # closing reduction over slots
```

`B_rf` is the rf-buffer, shape `[factor, *tile(out)]` in SBUF — **all `factor`
slots stay live**. `psum_rf[ko]` is the per-slot PSUM accumulator (the original
`psum`, now indexed by the parallel `ko`). The wb-block is a **separate block** with
its own init, reducing the slots — exactly `tir.Schedule.rfactor`'s output
(cf. the `matmul_rfactor` test: `C_rf = alloc([4, M, N])`, `update_rf` block, then a
separate `update` block reducing over the factored axis).

> This is **not yet** `kernel_hand_90.92mfu.py`. The SOTA kernel has one reused
> accumulator and no `[factor,…]` buffer — that is the *folded* form reached by §7
> (ComputeAt sinks the wb-block's combine into the `ko` loop; `compact_shapes`
> collapses `B_rf[factor,…] → [tile]`). RFactor stops at the faithful terminal form.

### 3.2 How the two recipes lower the rf-block's per-slot accumulate

The rf-block's `B_rf[ko] = reduce_fn over ki` lowers differently by recipe, but the
*block structure above is identical* for both:

- **`"rmw"` (matmul):** the per-`ki` accumulate is HW `+=` into a PSUM slot
  (`psum_rf[ko]`), then one `tensor_copy` drains the closed slot to `B_rf[ko]` in
  SBUF. (PSUM→SBUF copy also avoids the open-material-blocking PSUM-read hazard,
  `NCC_ISCH714`.) The wb-block's combine is `tensor_tensor(out += B_rf[ko])`.
- **`"slot"` (activation_reduce):** no HW accumulator; each `ko` writes its slot
  `B_rf[ko]` directly in SBUF, and the wb-block closes with `tensor_reduce(axis=ko)`.

The `reduce_fn`/identity for both come from `op_cls.REDUCE_COMBINATOR` (§4.1).

### 3.3 The emission delta

| | before | after |
|---|---|---|
| `ko` role | `ACCUMULATION` | **`PARALLEL`** (rf-block) / `ACCUMULATION` (wb-block) |
| `ki` role | `ACCUMULATION` | `ACCUMULATION` (unchanged) |
| loop order | `ko` outer of `ki` | **unchanged** (no reorder; only `ko`'s role flips) |
| accumulator | one `psum`, spans all K | per-slot `psum_rf[ko]` (spans `ki`) → `B_rf` rf-buffer `[factor,…]` |
| blocks added | — | **wb-block** (closing reduction) + its `memset(out_sbuf)` init |
| blocks changed | — | original drain `tensor_copy` now writes `B_rf[ko]` (slot-indexed) |
| output write | `tensor_copy(out_sbuf <- psum)` once | rf-block fills `B_rf`; wb-block reduces `B_rf → out_sbuf` |

### 3.4 Mechanical steps (`apply`) — mirrors `tir.Schedule.rfactor`

1. Resolve `target_loop_nid` → the `ko` ForNode and its owning matmul block; read
   `op_cls.RFACTOR_RECIPE` (`"rmw"`) and `op_cls.REDUCE_COMBINATOR`.
2. Deep-copy the IR (per the `Transform` contract).
3. **Allocate the rf-buffer** `B_rf` of shape `[factor, *tile(out)]` (factored extent
   prepended), location = the op's output location (SBUF). (TVM `CreateRFactorBuffers`.)
4. **rf-block** — in the original block: flip `ko`'s `IterVar.role` to `PARALLEL`;
   redirect the accumulate's dst and the drain to `B_rf[ko]` (slot-indexed by `ko`);
   its init (`memset` of the slot, to `REDUCE_COMBINATOR.identity`) stays a sibling
   inside `ko`. (TVM `RFactorBlockCreator`.)
5. **wb-block** — synthesize a *new* block that reduces `B_rf` over `ko` into the
   original output: a `memset(out_sbuf, identity)` init sibling + a combine leaf
   (`tensor_tensor(out += B_rf[ko])` for `"rmw"`; `tensor_reduce` for `"slot"`),
   with `ko` declared `ACCUMULATION` there. (TVM `WriteBackBlockCreator`.)
6. **No reorder.** Leave the loop nest order unchanged (TVM
   `CreateLoopOutsideRfactorBlock` reuses original order; only iter_var roles changed).
7. Re-run `place_buffers` (LCA) and rebuild `Dependency` — exactly as every shipped
   transform does in `apply`. The `memset → ko-loop` carry edges (shipped model)
   express that each init dominates its reduction.

### 3.5 Recipe `"slot"` (activation_reduce) — specified, deferred

Both recipes emit the same rf-buffer + wb-block *shape* (§3.1); they differ only in
how the rf-block's per-slot accumulate lowers (§3.2). For `activation_reduce` there
is no HW accumulator, so each `ko` writes its `B_rf[ko]` slot directly in SBUF and
the wb-block closes with a single `tensor_reduce(axis=ko)` rather than a
`tensor_tensor` running combine.

**No PSUM anywhere in `"slot"`.** `activation_reduce` accumulates in SBUF
(`OUTPUT_LOCATION = "sbuf"`); the rf-buffer, the per-slot writes, and the closing
`tensor_reduce` are all SBUF. The PSUM per-slot accumulator that appears in `"rmw"`
(`psum_rf[ko]`) has no analog here. This is why the recipe split is keyed on
*HW-accumulator presence*, not memory space — and it only affects the rf-block's
internal lowering, not the rf-buffer + wb-block structure rfactor produces.

It is specified here for taxonomy completeness but **not implemented in this
design** — matmul / `"rmw"` is the target. (The `RFACTOR_RECIPE = "slot"` marker
already ships on `NKIActivationReduce`.)

## 4. Components

### 4.1 `REDUCE_COMBINATOR` on `NKIOp` (new)

RFactor must synthesize the rf-block's per-slot init, the wb-block's init, and the
wb-block's combine for *any* rfactorable op without hardcoding `+`/`0`. The op
exposes its reducer:

```python
class NKIOp:
    REDUCE_COMBINATOR: ClassVar[ReduceCombinator | None] = None
    """The op's commutative-associative reducer, or None if not a reduction.
    RFactor reads this to synthesize the inits (memset to `identity`) and the
    wb-block combine (an ISA op applying `combiner`). Must be set on every op whose
    RFACTOR_RECIPE is not None."""
```

```python
@dataclass(frozen=True)
class ReduceCombinator:
    combiner: str       # nl op name for the wb-block combine: "add" for matmul
    identity: float     # memset value for the rf-block + wb-block inits: 0.0 for matmul
```

`NKIMatmul.REDUCE_COMBINATOR = ReduceCombinator(combiner="add", identity=0.0)`.
`NKIActivationReduce` declares its own (`"add"`/`0.0` or `"max"`/`-inf` per
`reduce_op`) when `"slot"` is implemented.

This mirrors TVM's `CommReducer` (combiner + identity), threaded through
`RFactorBlockCreator` / `WriteBackBlockCreator`. We need only the single-buffer
sum/max cases; TVM's multi-buffer argmax reducers are out of scope.

### 4.2 `NKITensorTensor` op (new dependency)

The `"rmw"` wb-block combine `out_sbuf = out_sbuf + B_rf[ko]` is an element-wise
tensor⊕tensor — `nisa.tensor_tensor`. nkigym has **no such op today** (only
`tensor_scalar`, `tensor_copy`, `tensor_reduce`). It must be added as a 1:1 ISA op:

```python
class NKITensorTensor(NKIOp):
    NAME = "tensor_tensor"
    OPERAND_AXES = {"in0": ("P", "F"), "in1": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS = frozenset({"in1"})       # in0 is the RMW accumulator
    RMW_OPERANDS = frozenset({"in0"})         # out_sbuf is read+written (out += B_rf[ko])
    # _run: dst = _NL_OPS[op](in0, in1)
```

Marking `in0` as RMW gives the wb-block the carried-RAW-on-`out_sbuf` that makes
`ko` `ACCUMULATION` there — the dependency machinery then derives the loop-carry
edge with no special-casing. (The `"slot"` recipe's wb-block uses the existing
`NKITensorReduce` instead, so `NKITensorTensor` is a `"rmw"`-only dependency.)

### 4.3 `RFactor` transform (new)

`nkigym/transforms/rfactor.py`, following the `SoftwarePipeline` idiom:

```python
@dataclass(frozen=True)
class RFactorOption(TransformOption):
    target_loop_nid: int        # the ko ForNode (an outer reduction loop)
    factor_axis: int = 0        # position of the new factored dim in B_rf (TVM factor_axis)

class RFactor(Transform):
    def analyze(self, ir) -> list[RFactorOption]: ...
    def apply(self, ir, option) -> KernelIR: ...   # deep-copy, mutate (§3.4), re-check, return
```

The option carries `target_loop_nid` (the loop to factor) and `factor_axis` (where
the prepended `[factor]` dim lands in `B_rf` — TVM's `rfactor(loop, factor_axis)`
second argument; default `0`). There is **no `outer_factor`**: factor selection is
the prior `Split`'s job (it enumerates factorizations and respects `MIN_TILE_SIZE`).
RFactor consumes whatever two-level loop structure Split produced and factors the
named loop. One mechanism per atom — exactly `tir.Schedule.rfactor`'s signature.

`analyze` enumerates every `ForNode` that (a) binds an axis some block declares
`ACCUMULATION`, (b) whose owning op has a non-`None` `RFACTOR_RECIPE`, and (c) has a
perfect inner reduction sub-nest. Recipe dispatch on `op_cls.RFACTOR_RECIPE`:
`apply` delegates to `_rfactor_rmw` (this design) or `_rfactor_slot` (deferred).

## 5. Legality and error handling

`RFactor.apply` re-checks all conditions and raises `TransformLegalityError`
(loud failure, no recovery) when any fails:

- `target_loop_nid` resolves to a `ForNode` in the tree.
- Its bound axis is `ACCUMULATION` in the owning op's block (it is a reduction loop).
- The owning op declares `RFACTOR_RECIPE is not None` **and** a `REDUCE_COMBINATOR`.
- The inner body is a perfect reduction sub-nest the block owns (init dominates it,
  drain follows it) — the canonical post-Split matmul shape.
- The accumulator (matmul `dst`) has exactly one writer (the reducer leaf) and one
  reader (the current drain). A different composition state would lose information;
  reject rather than guess. (Preserved from the 2026-05-09 spec.)

**Not gated (user-locked rule):** resource capacity. The `[factor, *tile(out)]`
rf-buffer `B_rf` is `factor`× the output tile and may over-subscribe SBUF — that is
still a **legal** RFactor output; compile / HW profiling prunes it, and the §7 fold
is what collapses it to one slot so it fits. RFactor must never reject on "won't
fit." The atom is *always* offered when the structural conditions hold; the search /
profiler decides whether it pays.

**Composition:** RFactor rejects if the target accumulation loop already carries a
`SoftwarePipeline` annotation or the accumulator a `versions > 1` (they compose
poorly *in one direction*). Post-RFactor, `SoftwarePipeline` applies independently
to the new tree — that is the intended downstream order (RFactor → hoist → pipeline).

## 6. Verification

Two oracles, in the order the learnings mandate (dependency-order is the *primary*
oracle; numeric sim is weaker — it can pass on last-write-wins luck while violating
producer/consumer order).

### 6.1 Byte-exact render against a hand kernel (primary gate)

The shipped gate for a transform is
`render(apply(before_IR)) == <hand kernel>`, AST-canonical (the
`test/transforms/_ladder_compare.py` oracle: hoists decls, positional→keyword, sorts
kwargs, distributes affine). For RFactor:

- Build the canonical matmul IR, apply `Split(K → ko, ki)` then `RFactor(ko)`.
- Author the post-RFactor hand kernel (the **rf-buffer + wb-block** form of §3.1 —
  `B_rf[factor,…]`, rf-block fills slots, separate wb-block reduces them) as the
  fixture. Gate: rendered output is AST-identical.
- This is TVM's terminal rfactor form, **not** `kernel_hand_90.92mfu.py` — that is
  the *folded* form reached by the §7 downstream transforms (fold + hoist + pipeline),
  out of scope here. A separate fixture/test covers the folded form when §7 lands.

### 6.2 CPU-sim numeric check (secondary)

`simulate_fp32` the rendered kernel against the numpy golden `lhs_T.T @ rhs` at
`atol=rtol=5e-3` (the matmul contract; unit-op matmul `1e-5`). Confirms the
rf-buffer + wb-block reduction is numerically equivalent to the single-level form.

### 6.3 Dependency-order assertions (the real legality oracle)

Independent of sim, assert on the post-RFactor `ir.dependency`:

- `memset(out_sbuf)` (wb-block init) precedes the wb-block's `ko` loop.
- `memset(psum_rf[ko])` (rf-block init) precedes each `ki` nest (init dominates the
  per-slot inner reduce).
- The wb-block's `tensor_tensor(out_sbuf += B_rf[ko])` carries a RAW on `out_sbuf`
  across the wb `ko` loop (the closing reduction is `ko`-carried).
- The rf-block fills `B_rf` with `ko` `PARALLEL` (each slot independent); the
  wb-block reads `B_rf` with `ko` `ACCUMULATION`. → `ko`'s `IterVar.role` is
  `PARALLEL` in the rf-block, `ACCUMULATION` in the wb-block.
- `B_rf` has exactly one writer per slot (the rf-block drain) and one reader (the
  wb-block) — the producer→consumer edge that orders rf-block before wb-block.

These catch the failure modes sim hides (init-after-reduce, combine sunk into `ki`,
missing carry edge) — the learnings record that the move-engine bugs and the
"drain-sunk-into-K" class were caught by render+dep checks, not sim.

### 6.4 Tests (new)

- `test/transforms/test_rfactor.py` — unit: `analyze` enumerates exactly the
  legal `ko` loops; `apply` byte-exact vs the §3.1 hand fixture; illegal options
  (non-reduction loop, op without recipe, pre-pipelined loop) raise
  `TransformLegalityError`; dependency-order assertions of §6.3.
- `test/ops/test_tensor_tensor.py` — unit: `NKITensorTensor` renders + sims
  (elementwise `1e-6`).
- Extend the `kernel_transforms.py` ladder with an RFactor rung once the
  intermediate form is byte-exact, so the deep-rollout harness exercises it.

## 7. Data flow recap, and the downstream fold

```text
canonical matmul IR
      │  Split(K → ko, ki)              [shipped]
      ▼
two-level loop, one PSUM accumulator    (ko, ki both ACCUMULATION)
      │  RFactor(ko)                     [THIS DESIGN — faithful tir.Schedule.rfactor]
      ▼
rf-buffer + wb-block (TVM terminal)     (ko PARALLEL in rf-block / ACC in wb-block;
  • rf-block fills B_rf[factor,…]         B_rf = [factor, *tile(out)], all slots live;
  • wb-block reduces B_rf → out_sbuf      no loop reorder)
      │  ComputeAt(wb-combine → ko) + compact_shapes   [shipped transforms; see below]
      ▼
fused single-accumulator                (B_rf folds [factor,…] → one reused tile;
  (= kernel_hand_90.92mfu.py shape        wb-combine sunk into ko)
   modulo hoist/pipeline)
      │  hoist ko outside M + SoftwarePipeline double-buffer   [separate spec]
      ▼
→ path to 90.92% MFU
```

**The fold (rf-buffer → fused single accumulator).** TVM stops at the multi-slot
form because the factor axis binds to concurrent threads; on Trn2 we fold. The fold
is *composition of shipped transforms*, not new mechanism:

1. **ComputeAt** sinks the wb-block's combine into the rf-block's `ko` loop (so the
   per-slot drain and the combine co-locate under one `ko`).
2. **`compact_shapes`** then collapses `B_rf[factor, *tile] → [tile]` automatically:
   once all touchers share the `ko` ForNode and index it identically, `ko` becomes a
   compaction anchor and the factored dim is zeroed out of the buffer shape (verified
   by reading `compact.py:_anchor_loop_vars` / `_axis_span`).

> **Open dependency (parked).** Step 1 sinks the wb-block's combine — whose `ko` is
> `ACCUMULATION` — under the rf-block's `ko`. The shipped
> `_check_no_reduction_axis_covered` currently **rejects** covering a reduction axis,
> so the fold needs that check *narrowed* to permit this case (the wb-block's
> `memset(out_sbuf)` init stays outside `ko`, so init still dominates — the move is
> semantically legal). Narrowing that check is **parked** (the block-granular /
> region_cover investigation; see `.claude/rules/tvm_knowledge.md` and the
> learnings). Until it lands, RFactor produces the faithful terminal form and the
> fold is blocked — which is why §6.1 gates on the rf-buffer form, not the fused one.

## 8. Scope and non-goals

**In scope:** the `RFactor` transform (`"rmw"` recipe), `REDUCE_COMBINATOR` on
`NKIOp` + `NKIMatmul`, the `NKITensorTensor` op, and their tests.

**Out of scope (referenced, specified/verified separately):**

- The `"slot"` recipe implementation (activation_reduce). Taxonomy only here.
- **The fold (rf-buffer → fused single accumulator), §7.** It is composition of
  shipped transforms (ComputeAt + `compact_shapes`), but it is **blocked** on
  narrowing `_check_no_reduction_axis_covered`, which is **parked** (block-granular /
  region_cover investigation). RFactor itself (the faithful terminal form) does **not**
  depend on the fold and ships first.
- The hoist of `ko` outside M, and `SoftwarePipeline` double-buffering. Whether the
  shipped move transforms can express the M-blocked / N-tiled drain the SOTA needs is
  an open question flagged in the learnings — **not assumed here.**
- Byte-exact reproduction of `kernel_hand_90.92mfu.py` end-to-end (needs the fold +
  hoist + pipeline composed).
- Capacity-aware pruning of RFactor outputs — profiler/HW's job, never the
  transform's (user-locked). The large `[factor,…]` rf-buffer is the motivating case.
- TVM's multi-buffer reducers (argmax/argmin). Single-buffer sum/max only.

