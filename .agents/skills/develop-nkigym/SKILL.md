---
name: develop-nkigym
description: Start or continue a persistent goal to debug and improve the NKIGym backend until all tests pass, modifying nkigym/src/nkigym and recording or updating best NKIGym ladders in kernel_library.
---

# Develop NKIGym

When explicitly invoked:

1. Inspect the current goal with `get_goal`.
2. If no goal is active, call `create_goal` without a token budget using:

   ```text
   Debug and improve the NKIGym backend until all repository tests pass.
   Modify only files under nkigym/src/nkigym and kernel_library. Limit
   kernel_library edits to recording or updating best NKIGym ladders.
   Write generated runtime measurements under artifacts/nakb_latency_comparison/.
   Save comparison figures and their JSON metadata directly under kernel_library/.
   Promptly delete stale generated NKIGym caches locally and on gym-trn2-1
   when no active job or future run needs them, preserving validation evidence.
   Keep benchmark/ and its checksum in test/test_repository_structure.py unchanged.
   Do not modify any other repository files except for this cache cleanup.
   ```

3. Continue an existing matching goal. The user has explicitly authorized
   recording or updating best NKIGym ladders under `kernel_library/`, including
   `kernel_library/_best_nkigym.py`. Apply the permissions below when an older
   matching goal records narrower scope. Never replace an
   unrelated active goal.
4. Work on the goal immediately. Reading files and running tests are allowed,
   but create, modify, move, delete, or format files only under
   `nkigym/src/nkigym/` and `kernel_library/`. Limit `kernel_library/` edits to
   recording or updating best NKIGym ladders.
   Generated runtime measurements may also be written under
   `artifacts/nakb_latency_comparison/`. Save comparison figures and their JSON
   metadata directly under `kernel_library/`.
   Deleting stale generated NKIGym caches outside these directories is also
   allowed under the cleanup rule below.
   The NAKB targets and accuracy criteria in `benchmark/` are frozen; do not
   modify them or the checksum in `test/test_repository_structure.py`.
5. After installing an accepted ladder, publish the successful confirmation
   results already collected during development:

   ```bash
   ~/venvs/kernel-env/bin/python -m kernel_library.nakb_comparison \
     --publish path/to/validation.json
   ```

   Pass multiple result files to publish several accepted updates together.
   The command only reads results and updates the shared measurement record;
   it never replays, compiles, simulates, or profiles kernels. Confirm that each
   result belongs to the installed ladder and backend. Preserve any source
   hashes captured with the measurement; do not stamp today's hashes onto old
   results. Keep older timings as diagnostics when freshness is unverified.
6. Read aggregate progress from the shared record, including after resuming:

   ```bash
   ~/venvs/kernel-env/bin/python -m kernel_library.nakb_comparison
   ```

   Use its run ID, saved mean, coverage, and diagnostic status in summaries.
   Do not maintain a
   separate aggregate in `.cache/equal-weight/current-measurements.json` or
   publish unaccepted candidate timings as measurements of installed ladders.
   Regenerate figures with:

   ```bash
   ~/venvs/kernel-env/bin/python .agents/skills/compare-nakb/scripts/plot_nakb_comparison.py
   ```

   Plotting only reads saved results. Never rerun experiments just to publish
   statistics or produce a plot. Missing or unverified timings must be labeled,
   not automatically remeasured. Incremental results remain diagnostic; the
   full ladder test still owns fresh full-registry performance acceptance.
7. On startup or resume, after each experiment or validation batch, and before
   a checkpoint, promptly delete generated caches and scratch directories
   that no active job, planned run, or unresolved diagnosis needs. This includes
   obsolete local experiment and compiler outputs and completed remote run
   directories on `gym-trn2-1`. Verify that each selected path is unused; do
   not clear shared cache roots. Before deleting cached evidence, publish any
   accepted validation results and preserve required reproduction data and
   measurement provenance. Keep reusable compiler caches and the authoritative
   records under `artifacts/nakb_latency_comparison/`.
8. Call `update_goal` with `complete` only after all repository tests pass.
