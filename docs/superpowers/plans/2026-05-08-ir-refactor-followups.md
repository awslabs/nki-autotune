# IR Refactor Followups

**Parent spec:** `docs/superpowers/specs/2026-05-08-ir-and-transforms-refactor-design.md`
**Parent plan:** `docs/superpowers/plans/2026-05-08-ir-and-transforms-refactor.md`
**Created:** 2026-05-08
**Updated:** 2026-05-08 — Bug #3, Bug #2, Task 23, Tasks 22+24 landed on `dev_1`. Bug #1 + Task 25 deferred pending design.

Followups from the IR + transforms refactor. Unit-test suite shipped green (86/86); MFU gate on `matmul_lhsT_rhs` exposed a cluster of atom-composition bugs. Five were fixed in-session; three remain open. Two renderer-split tasks from the parent plan were also deferred.

## Progress

| Item | Status | Commit |
|------|--------|--------|
| Bug #3 — Split canonical-rename | Landed | `7651246` |
| Bug #2 — MultiBuffer apply-time re-validation | Landed | `c88caad` |
| Task 23 — PlaceBuffers pass extraction | Landed | `315a8ee` |
| Tasks 22 + 24 — LowerPhases / InjectMultiBuffer / InjectSoftwarePipeline | Landed | `4e7522f` |
| Bug #1 — DecomposeReduction divergence | Deferred | — |
| Task 25 — LowerDecomposedReduction pass | Deferred | — |

---

## Open composition bugs (MFU gate blockers)

Shared root cause across all three: **per-atom legality checks local structure, not cross-atom semantic invariants**. Each bug needs either a tighter `is_legal`, a post-apply validator, or a repair-pass in the lowering pipeline. 3/100 sampled kernels pass CPU sim today; each fix should lift that materially.

### Design note — Bug #1 blocker (2026-05-08)

`_body_matmul_psum_init` emits `psum_tile = nl.ndarray(...)` as a Python-local binding inside the init leaf's body. After `DecomposeReduction` splits into init/update/drain siblings, the init tree rebinds `psum_tile` per `(m, n)` iteration — ending pointed at the last iter's allocation — so the update tree's `nc_matmul(dst=psum_tile, ...)` calls all land in the final slot. The bug exists independently of any later `Reorder`; decomposition alone is sufficient to break PSUM scope. Option 1 (coupled-rewrite atoms) and Option 2 (widened accumulator) both require re-plumbing how PSUM is named/sized — Option 2 hits the Trn2 2 MiB PSUM ceiling for 2048³ matmul when naively sized per full-tile-count, so the fix requires either per-tree LCA-scoped PSUM shapes or a rewrite-space constraint on DecomposeReduction inputs. Design session needed before implementation.

### 1. DecomposeReduction + Reorder phase-tree divergence

**Symptom:** After `DecomposeReduction` splits a reducer into `init`/`update`/`drain` sibling trees, subsequent `Reorder` / `ComputeAt` on the update tree reshapes its spatial (M, N) iteration differently than the init and drain trees. The three trees reference the same PSUM buffer by name but iterate different tile indices; the buffer ends up holding only the last-written tile of init, so update accumulates into stale slots and drain stores garbage.

**Example:** `kernel_tuned_0001.py` from the last gate run.

**Fix direction:** Two options.
- Enforce per-accumulator shape equivalence: any rewrite that changes one phase tree's spatial loop structure must produce a matching rewrite on the sibling phase trees (coupled-rewrite atoms).
- Accept divergence but widen the accumulator per-tree-shape: `LowerDecomposedReduction` pass (deferred Task 25) rebuilds the PSUM shape post-divergence.

Option 2 aligns with the spec's deferred pass; favor it once the renderer split lands.

### 2. MultiBuffer slot-modulo aliasing on cross-nest tensors

**Symptom:** Even with the tightened `degree ≤ num_tiles / required_tiles` cap, `MultiBuffer` on a cross-nest tensor (LCA at forest root) with degree < num_tiles produces emission like `sbuf_x[:, (i_d0_0) % 8, ...]`. For 16 distinct tiles cycling mod 8, tile 8 overwrites tile 0 before any consumer in a later tree reads tile 0.

**Root cause:** Cross-nest tensors have `required_tiles == num_tiles`, so any `degree < num_tiles` undersizes the buffer. The cap formula happens to collapse to `max_degree == 1` for these tensors (good), but the bug surfaces when `required_tiles` is miscomputed — e.g. after a `ComputeAt` narrows the LCA, the enumerator re-enumerates atoms against the new state where `required_tiles < num_tiles`, and a later-applied `MultiBuffer` atom (enumerated pre-ComputeAt) is now against a different LCA shape.

**Fix direction:** Re-validate `MultiBuffer.is_legal` at apply time against the current state (same pattern as Task 19b's `_find_node_path` for ComputeAt). If post-rewrite `required_tiles` no longer supports the captured degree, reject.

### 3. Split same-dim loop-var collisions

**Symptom:** `Split(loop, factor)` produces `outer = LoopNode(dim_id=d, trip=N/f)` and `inner = LoopNode(dim_id=d, trip=f)`. Canonical naming assigns `i_d_0` to outer and `i_d_1` to inner. But an unrelated ancestor loop on the same dim (e.g. from canonical form's 2N-per-dim nesting) may already occupy `i_d_0`, so the inner rebinds the outer's variable name.

**Example:** `for i_d1_0 in range(2): for i_d1_1 in range(1): ... for i_d1_1 in range(1): ...` — the inner-inner shadows the outer-inner.

**Fix direction:** `_rename_canonical` (in `compute_at.py`) already reassigns ordinals per-tree. Need the same pass to run inside `Split.apply()` — currently Split does not call canonical-rename, so the stale `name=None` on new loops allows the renderer's name-assignment fallback to collide. Minimal fix: call `_rename_canonical(new_body)` at the end of `Split.apply()`.

---

## Deferred renderer-split tasks

From the parent plan, Tasks 22–25 were intentionally deferred after user preference for "HW gate first, defer renderer split." These remain:

### Task 22: Extract `LowerPhases` pass
Pull the `(op_cls, phase) → ISA call-site snippet` dispatch out of `lowering/emit_source.py` into its own pass. Leaf metadata annotation via `_isa_call_source` field.

### Task 23: Extract `PlaceBuffers` pass
Move the LCA walk + `required_tiles` + buffer-placement derivation out of `emit_source.py`. Shared logic with `MultiBuffer.is_legal` (followup #2 above); extracting here lets both call sites consume the same implementation.

### Task 24: Extract `InjectMultiBuffer` + `InjectSoftwarePipeline` passes
Move buffer-degree slot expressions and prologue/body/epilogue emission out of `emit_source.py`. After this, the software-pipeline clamp (shipped in-session for Task 27b) lives in `inject_software_pipeline.py` where it belongs conceptually.

### Task 25: Add `LowerDecomposedReduction` pass
Canonicalize dim_roles on post-fission trees. This is the natural home for followup bug #1's "option 2" fix — per-tree widened accumulator shapes derived from each phase tree's actual LCA.

**Order-of-operations note:** Tasks 23 and 25 together enable followup bug #1's fix. Task 22 is useful for test isolation but doesn't unblock any followup. Task 24 cleanly relocates the software-pipeline clamp hack.

---

## New followups surfaced during 2026-05-08 implementation pass

- **Align `ComputeAt` / `ReverseComputeAt` on `AtomLegalityError`.** Both raise bare `ValueError` on their similar stale-target re-resolve failures (`compute_at.py:69`, `reverse_compute_at.py:74`). `batch.py` now catches `AtomLegalityError` but crashes on `ValueError`. Port the two sites for a unified staleness channel.
- **Cycle-break cleanup in `lowering/`.** `emit_source.py` imports `_emit_pipelined_loop` and `_BODY_EMITTERS` at module bottom with `# noqa: E402`; `inject_software_pipeline.py` aliases `emit_source as _es` for late-bound `_es._emit_vanilla_loop`. Cleaner path: expose `effective_pipeline_depth(node, module)` from `inject_software_pipeline.py` so `_emit_node`'s dispatch decides vanilla vs pipelined without the pipelined emitter needing to call back.
- **Standardize private-vs-public naming in `lowering/`.** `_BODY_EMITTERS`, `_emit_pipelined_loop`, `_Writer`, `_sbuf_name`, `_hbm_name` are underscore-private but imported across module boundaries. Either drop the underscore (matching `slot_expr`/`sbuf_tile_slice` in `inject_multi_buffer.py`) or declare them via `__all__`.
- **Body emitter type annotations.** The 11 body emitters in `lower_phases.py` lack parameter type hints — only return type is annotated. `_BODY_EMITTERS: dict[tuple[str, str], Callable]` uses bare `Callable`; a `Protocol` matching the emitter signature would tighten the registry.

## Suggested order

~~1. Bug #3 (Split canonical-rename) — smallest, no dependency on other work.~~ **Landed `7651246`.**
~~2. Bug #2 (MultiBuffer apply-time re-validation) — follow the `_find_node_path` pattern from Task 19b.~~ **Landed `c88caad`.**
~~3. Task 23 (PlaceBuffers extraction) — sets up shared LCA/required-tiles infrastructure.~~ **Landed `315a8ee`.**
~~6. Tasks 22, 24 — cosmetic cleanup, no bug-fix coupling.~~ **Landed `4e7522f`.**

Remaining:
1. **Bug #1 design session** — resolve the PSUM scope problem (naming/sizing) before attempting Task 25 or coupled-rewrite atoms.
2. **Task 25 (LowerDecomposedReduction)** — home for Bug #1's fix, once design lands.
3. **Bug #1 (DecomposeReduction divergence)** — uses the Task 25 pass.

Expected MFU-gate progression: fixing #3 alone should bring ~10–30/100 kernels to CPU-sim pass; #2 another meaningful chunk; #1 closes the gap to SOTA. Measurement pending on `dev_1` after landed work.
