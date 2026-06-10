# RFactor Spec-Compliance Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make `RFactor._emit_rmw` emit the spec §3.1 nested per-`ko` rf-block (memset + matmul + drain under one `ko` loop, slot-indexed) instead of the shipped flat sibling form — then verify the corrected shape both (a) renders/sims correct and (b) is foldable by `ComputeAt` (which the flat form provably is not).

**Architecture:** The shipped RFactor emits `ko` flipped PARALLEL + a separate wb-block (correct), but emits the rf-init `memset` and rf-drain `tensor_copy` as **flat sibling blocks over all `factor` slots, outside `ko`** — violating spec §3.1/§3.4-step-4 which require them **per-slot, nested inside `ko`**. This flat shape is also exactly why the §7 fold is blocked (the flat drain writes all slots before the wb-combine reads them → backward RAW edge). The fix re-emits the rf-block in the nested per-`ko` form; the fold then becomes a legal `ComputeAt(wb-combine → ko)` because `drain[ko]` and `combine[ko]` co-locate.

**Tech Stack:** Python 3.12, `networkx` IR, `numpy` CPU sim. Tests run remotely via `transport/remote_pytest.sh` on the Kaizen desktop (no local Python env). Driver/example runs use `--cache /home/weittang/workplace/cache`.

**Spec (source of truth):** `docs/superpowers/specs/2026-06-07-rfactor-transform-design.md` §3.1, §3.4, §7
**Current (wrong) emission:** `nkigym/src/nkigym/transforms/rfactor.py` `_emit_rmw` + helpers

---

## Root cause (verified on hardware, 3 scouts)

The shipped `RFactor` (`tuned.py` from a real run) emits:
```text
for i_d1_0 in range(32): memset(psum_prod[:, i_d1_0, :])        # FLAT, all 32 slots, OUTSIDE ko — WRONG
for ko(2): for ki(8): for m(16): for n(4): nc_matmul(psum_prod[:, ko*16+m, :])   # OK
for i_d1_0 in range(32): tensor_copy(psum_prod_rf[:, i_d1_0, :] <- psum_prod[:, i_d1_0, :])  # FLAT — WRONG
memset(sbuf_prod); for ko(2): for m(16): tensor_tensor(sbuf_prod += psum_prod_rf[:, ko*16+m, :])  # OK
```

Spec §3.1 requires (the `memset` and `tensor_copy` **inside `ko`**, slot-indexed):
```text
for ko (PAR):
  memset(psum_rf[ko])                          # per-slot, INSIDE ko
  for ki (ACC): nc_matmul(dst=psum_rf[ko], ...)
  tensor_copy(B_rf[ko] <- psum_rf[ko])         # per-slot, INSIDE ko
memset(out_sbuf)                               # wb-block (separate) — correct already
for ko (ACC): tensor_tensor(out_sbuf <- out_sbuf, B_rf[ko], op=add)
```

**Why it matters (not cosmetic):** the flat drain (`tensor_copy` over all 32 slots) writes every slot before the wb-combine reads any → a real backward RAW edge `drain → combine`, so `ComputeAt(wb-combine → ko)` (the §7 fold) is correctly rejected. With the drain nested per-`ko`, `drain[ko]` and a sunk `combine[ko]` co-locate in the same iteration → forward edge → the fold becomes legal. So spec-compliance is the prerequisite for the fold to the fused single-accumulator (the path to a *fitting*, near-SOTA kernel).

**How this slipped through:** Task 5's byte-exact test gated against the implementer's own captured render, not against spec §3.1. The fix must re-gate against the spec shape.

## Scope

In scope: `_emit_rmw` (and its helpers) re-emission to the nested form; re-capture the byte-exact fixture; update tests; re-verify sim + fold-legality on the corrected shape; then **re-do the matmul hand-tuning from scratch with the full transform set (incl. the fixed RFactor), keep the single best-measured TRACE in one driver** (delete both current drivers). Out of scope (separate follow-on): the `"slot"` recipe.

## Constraints

- NO local Python env — subagents edit+commit; the CONTROLLER runs every `pytest`/render/sim via `transport/remote_pytest.sh` and feeds results back. Subagents use `python3 -m py_compile` for syntax only.
- Code style (`.claude/rules/code_style.md`): triple-quoted comments only (no `#`), Google/NumPy docstrings, modern type hints, single-return preference. black line-length 120 (pre-commit enforced).
- The shipped emission is byte-exact-tested; changing it WILL change the rendered output, so the byte-exact fixture `kernel_library/matmul/lhsT_rhs/kernel_rfactor_ko.py` must be re-captured from the corrected render (controller-captured, since only the controller can render).
- 2 pre-existing desktop failures (`test_dump_tree_runs_on_canonical_ir`, `test_fuse_tensorize_matmul_n_renders_and_sims`, both `mmdc`-missing) are NOT regressions.

---

## File Structure

- **Modify:** `nkigym/src/nkigym/transforms/rfactor.py` — rewrite `_emit_rmw` + replace `_slot_memset`/`_retarget_drain`/`_grow_partition_loop` (which build the flat form) with helpers that nest the per-`ko` memset + drain under the matmul block's `ko` ForNode.
- **Re-capture:** `kernel_library/matmul/lhsT_rhs/kernel_rfactor_ko.py` — the byte-exact fixture, regenerated from the corrected render (controller-captured).
- **Modify:** `test/transforms/test_rfactor.py` — byte-exact + sim + dep-order tests stay; the dep-order asserts get tightened to the nested shape; ADD a fold-legality assertion.
- **Modify:** `docs/superpowers/specs/2026-06-07-rfactor-transform-design.md` — only if the verified nested render differs in any detail from §3.1 (it should match; reconcile if not).

## Current vs target emission (precise)

The current helpers keep memset/drain as flat siblings and grow their partition loop:
- `_slot_memset(tree, psum_name, factor)` → `_grow_partition_loop` (loop 16→32, flat, OUTSIDE ko).
- `_retarget_drain(...)` → retarget dst to `B_rf` + `_grow_partition_loop` (flat, OUTSIDE ko).

Target: the memset and drain become children of the matmul block's `ko` ForNode (#21 in the fixture), each gaining a `ko` iter-binding and a slot-indexed region `[ko*m_tiles + m, ...]` on axis 0 (the same `_slot_region` already applied to the matmul dst). The matmul block thus owns three statements under `ko`: `memset(psum_rf[ko]) → for ki: matmul → tensor_copy(B_rf[ko] <- psum_rf[ko])`.

---

### Task 1: De-risk — prove the nested form sims AND folds (controller scout)

**This is the controller's job (needs render/sim/legality on the desktop); no subagent.** Before rewriting shipped code, hand-construct (or transform-construct) the target nested IR and confirm: (a) it renders + sims `== lhs_T.T @ rhs`; (b) `ComputeAt(wb-combine → ko)` is now LEGAL on it (the fold the flat form blocks); (c) `compact_shapes` collapses the `[factor*M, N]` buffers to `[M, N]` after the fold. If any fails, STOP and reassess — do not rewrite `_emit_rmw` against an unverified target.

- [ ] **Step 1:** Controller writes a throwaway `test/transforms/_nested_scout.py` that builds the nested rf-block IR (simplest path: construct it by directly editing the post-RFactor tree — move the memset/drain leaves under `ko`, add the `ko` iter-binding + slot region — mirroring what `_emit_rmw` will do), renders it, prints the kernel, sim-checks it, then attempts `ComputeAt(wb_combine_block → ko)` and reports ACCEPT/REJECT, then (if accepted) applies it + `compact_shapes` and prints the final buffer shapes.
- [ ] **Step 2:** Controller runs it remotely; reads the printed render + verdicts.
- [ ] **Step 3:** GATE: nested form sims clean AND fold is ACCEPT AND buffers collapse to `[M,N]`. If yes → proceed to Task 2 with the verified target render in hand. If no → STOP, report the specific failure, reassess the plan. Delete the scout.

---

### Task 2: Rewrite `_emit_rmw` to emit the nested per-`ko` rf-block

**Files:** Modify `nkigym/src/nkigym/transforms/rfactor.py`.

- [ ] **Step 1 (controller-run): capture the BEFORE byte-exact baseline.** Controller renders the current (flat) RFactor output and saves it, so we can diff old→new render and confirm ONLY the memset/drain nesting changed (matmul + wb-block unchanged).

- [ ] **Step 2 (subagent): replace the flat memset/drain helpers with nesting helpers.**
  Replace `_slot_memset` and `_retarget_drain` (and retire `_grow_partition_loop`) with helpers that, for the memset leaf and the drain leaf:
  1. **Detach** the leaf's enclosing block from the root and **re-parent the leaf** (with a fresh per-`ko` block) under the matmul block's `ko` ForNode, as a sibling of the `ki` sub-nest — memset BEFORE the `ki` loop, drain AFTER it (sibling order = dataflow: `memset, ki-matmul-nest, drain`).
  2. **Add the `ko` iter-binding** to the new block's `iter_vars`/`iter_values` (axis = the matmul's K dim, role neutral on the ForNode; the block declares it) so the leaf's region can index `ko`.
  3. **Slot-index the region** on axis 0 via the existing `_slot_region(region, ko_var, m_tiles)` helper (the same `ko*m_tiles + m` prefix used for the matmul dst): memset writes `psum_rf[ko-slot]`, drain reads `psum_rf[ko-slot]` and writes `B_rf[ko-slot]`.
  4. Do NOT grow any partition loop — the per-`ko` slot indexing replaces the flat sweep; the memset/drain partition loops keep extent `m_tiles` (16), now nested under `ko`(2).
  Use the existing tree-mutation API: `tree.add_node(BlockNode(...), parent=ko_loop)`, `_replace_in_parent_children` to splice at the right sibling index under `ko`, `tree.graph.remove_node` for the old flat blocks, `dataclasses.replace` for region/iter_var edits. Mirror `canonical_build`'s block-construction idiom. Keep the matmul `_flip_and_slot_matmul` and `_insert_writeback` (wb-block) UNCHANGED — they are already spec-correct.
  Commit: `feat(rfactor): emit nested per-ko rf-block (spec §3.1), not flat siblings`.

- [ ] **Step 3 (controller-run): render + sim.** Controller renders the corrected RFactor output, confirms it matches the Task-1-verified nested shape, sim-checks `== lhs_T.T @ rhs`. If the render diverges from the Task-1 target, hand the diff back to the subagent to iterate (expect 1-3 rounds — IR surgery).

- [ ] **Step 4 (controller): re-capture the byte-exact fixture.** Once sim-clean, controller overwrites `kernel_library/matmul/lhsT_rhs/kernel_rfactor_ko.py` with the corrected render (verbatim) and commits it.

- [ ] **Step 5 (controller-run): the existing RFactor test suite.** Run `test/transforms/test_rfactor.py` — byte-exact now gates against the corrected fixture; sim + ko-role-split must still pass.

- [ ] **Step 6 (controller-run): full regression.** `test/transforms/ test/ir/ test/codegen/ test/ops/` — expect only the 2 pre-existing `mmdc` failures. The dependency-model change is NONE here (we did not touch dependency.py); but `_emit_rmw`'s new tree shape exercises `place_buffers`/`Dependency` build, so confirm no new breakage.

- [ ] **Step 7 (subagent): tighten the dep-order tests + add a fold-legality test.**
  In `test/transforms/test_rfactor.py`:
  - The existing byte-exact / sim / ko-role-split tests stay (byte-exact now gates the nested fixture).
  - Tighten the dep-order asserts to the nested shape: `memset(psum_rf[ko])` and `tensor_copy(B_rf[ko])` are now INSIDE the `ko` loop (assert their enclosing-ForNode chain contains the matmul's `ko` loop).
  - ADD `test_fold_is_legal`: after RFactor, `ComputeAt(wb_combine_block, ko_loop)` must NOT raise (the flat form raised; the nested form must accept) — the regression-guard that locks in spec-compliance-enables-fold.
  Commit: `test(rfactor): assert nested rf-block shape + fold legality`.
  Controller runs it remotely; iterate if any assert is mis-specified.

---

### Task 3: Verify the fold composes to the fused single-accumulator

**Files:** none yet (controller scout) — confirm the fold + compact produce the SOTA shape and fit PSUM.

- [ ] **Step 1 (controller-run):** Build `Split → RFactor → ComputeAt(wb-combine → ko) → [compact_shapes runs in ComputeAt.apply]`, render, and confirm the result is the fused single-accumulator shape: one reused accumulator tile (no `[factor*M,N]` buffer), `memset(out) outside ko`, `for ko: memset(psum_i); ki-matmul; drain; out += acc`. Sim-check `== lhs_T.T @ rhs`.
- [ ] **Step 2 (controller-run):** Profile the folded kernel on Trn2 (`--cache /home/weittang/workplace/cache`). GATE: does it now COMPILE (no PSUM-OOM — the accumulator is one tile, not `factor*M`)? Record the MFU. (This is the real payoff check: spec-compliant RFactor + fold → a *fitting* kernel.)
- [ ] **Step 3:** If it compiles and runs: record the MFU vs the 83.4% trace and the 90.92% hand kernel. If it still OOMs or underperforms, that points to the next lever (ko-outside-M hoist / SoftwarePipeline / M-N tiling) — note which, do NOT silently claim SOTA.

---

### Task 4: Re-do the hand tuning from scratch with the full transform set

**Goal:** With RFactor now spec-correct and foldable, **hand-tune the matmul end-to-end again from canonical**, exploring the FULL transform set (`Split`, `Fuse`, `Reorder`, `ComputeAt`, `ReverseComputeAt`, `SoftwarePipeline`, `RFactor`) on hardware, and keep the single best-measured sequence in ONE driver. The best TRACE is **discovered by profiling on Trn2, not predetermined.** Delete both current drivers; the surviving driver is freshly written around whatever wins.

This is exploratory + controller-led (every candidate must compile+profile on Trn2). The earlier `tune_matmul_lhsT_rhs.py` reached ~83.4% WITHOUT RFactor; the question this task answers empirically is **"does adding RFactor+fold (and re-tuning around it) beat 83.4%, and how close to the 90.92% hand kernel does it get?"**

**Files:** Delete `examples/tune_matmul_lhsT_rhs.py` AND `examples/tune_matmul_lhsT_rhs_rfactor.py`; create one fresh `examples/tune_matmul_lhsT_rhs.py`.

- [ ] **Step 1 (controller): exploration loop on Trn2.** Starting from canonical, hand-build candidate TRACEs and profile each (`--cache /home/weittang/workplace/cache`), reading the profiler MFU + the OOM/fits verdict to steer the next candidate (the documented dev loop: read profiler → adjust → re-run). Candidates to explore, building on the now-foldable RFactor:
  - the spec-compliant `Split(K) → RFactor(ko) → ComputeAt(wb-combine → ko)` fused base (Task 3's result);
  - then layer the levers that took the non-rfactor kernel to 83.4% and the hand kernel to 90.92%: `Reorder` to get `ko`/K outside M (rhs reuse), `ComputeAt`/`ReverseComputeAt` to sink loads + memset/drain, `SoftwarePipeline` to double-buffer the accumulator, `Split` of N/M for tiling;
  - vary factor (the `Split(K)` factor handed to RFactor), buffer multi-versioning, and loop order.
  Each candidate's every rung is CPU-sim-checked (halt-on-mismatch) before its Trn2 profile, so only correct kernels are measured. Record each candidate's `(TRACE, MFU, fits?)` — keep a short log of what was tried and what it measured (no silent truncation of the search).

- [ ] **Step 2 (controller): pick the winner.** The single best-MEASURED sequence that compiles+runs. Honest comparison points: the prior 83.4% (non-rfactor) and the 90.92% hand kernel (`kernel_library/matmul/lhsT_rhs/kernel_hand_90.92mfu.py`). State plainly where the best traced kernel lands between them; if it does NOT beat 83.4%, say so and keep 83.4% as the winner.

- [ ] **Step 3 (subagent): write the single driver around the winning TRACE.** Fresh `examples/tune_matmul_lhsT_rhs.py`: `f_nkigym` (canonical), `INPUT_SPECS`, the winning `TRACE` (literal nids, deterministic), `_validate_trace` (sim every rung), profile the final kernel vs the neuronx-cc baseline, print MFU. Model the structure on the deleted drivers (sim-check + KernelJob + profile_numpy_baseline). Document the winning sequence + its measured MFU in the module docstring honestly.
  **CRITICAL fixture contract:** `test/transforms/_pipeline_fixtures.py:5` imports `INPUT_SPECS, TRACE, f_nkigym` from this module and `tuned_ir()` replays `TRACE` breaking at the first `SoftwarePipeline` atom (to get the pre-pipeline state). So the surviving module MUST export those three names, and the winning `TRACE` MUST contain a `SoftwarePipeline` atom for `tuned_ir()` to break on. If the winning sequence happens to have no SoftwarePipeline atom, update `_pipeline_fixtures.py` to build its pre-pipeline state another way (read it first; adjust the fixture, do not break `test_software_pipeline.py`). Commit: `feat(examples): single matmul tuner — best hand trace (<MFU>%) with full transform set`.

- [ ] **Step 4 (controller-run): the driver on Trn2** (`--cache /home/weittang/workplace/cache`) — confirm every rung sim-PASSes and the final MFU prints. Read the kernel at `/home/weittang/workplace/cache/tune_matmul_lhsT_rhs/tuned/tuned.py`.

- [ ] **Step 5 (controller-run): fixture contract held.** Run `test/transforms/test_software_pipeline.py` (uses `_pipeline_fixtures`) — must pass. Fix the fixture if the new `TRACE` shape requires it; re-run.

- [ ] **Step 6 (controller): record the result in learnings.** Update the Matmul-MFU-ladder bullet with the measured best-traced MFU using RFactor, and whether it closed/narrowed the 83.4%→90.92% gap. State the remaining lever plainly if not at 90%.

---

## Final self-review checklist (controller, before finishing)

- [ ] `render(Split→RFactor)` is byte-exact to the spec §3.1 nested shape (memset + drain INSIDE `ko`, slot-indexed) — not the old flat form.
- [ ] `test_fold_is_legal` passes — the nested form is foldable (locks in the spec-compliance↔fold link).
- [ ] Spec §3.1 and the rendered nested kernel agree; if not, the spec was reconciled.
- [ ] Exactly ONE matmul tuning driver survives (`examples/tune_matmul_lhsT_rhs.py`); the standalone rfactor driver is deleted.
- [ ] The surviving driver exports `INPUT_SPECS, TRACE, f_nkigym`; `_pipeline_fixtures` + `test_software_pipeline` still pass (fixture updated if the winning TRACE has no SoftwarePipeline atom).
- [ ] Full regression: only the 2 known `mmdc` failures.
- [ ] No transform LOOSENED legality (this fix is pure emission shape; confirm dependency.py + _code_motion.py untouched).
- [ ] Honest MFU reporting: the winning traced kernel's MEASURED MFU stated vs 83.4% / 90.92%; no SOTA overclaim; the exploration log notes what was tried.
