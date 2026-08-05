---
name: develop-nkigym
description: Repair existing nkigym bugs until canonical gates pass, then run a user-selected number of focused improvement rounds for IR, operation contracts, code generation, or transforms from measured agentic-tuning evidence. Use for starting or resuming the repository's self-development workflow, fixing baseline or candidate failures in an isolated worktree, and promoting checked candidates.
---

# Develop nkigym

Use `scripts/develop.py` for durable mechanics. Own the diagnosis and source edits yourself.

## Establish the run

1. Work from the repository containing this skill.
2. Use `~/venvs/kernel-env/bin/python` when it exists; otherwise use the active Python interpreter.
3. Read `.agents/rules/learnings.md` before editing.
4. Resume a supplied run with:

   ```bash
   python .agents/skills/develop-nkigym/scripts/develop.py status RUN_DIRECTORY
   ```

5. When no run exists, require a kernel-library workload and profile host. Ask the user how many improvement rounds to
   run; require an explicit positive integer and do not choose a default. Then create the run:

   ```bash
   python .agents/skills/develop-nkigym/scripts/develop.py start WORKLOAD --host HOST --rounds ROUNDS
   ```

   Choose `WORKLOAD` from `attention`, `matmul-lhs`, `matmul-lhs-t`, or
   `rmsnorm-matmul`. The support script loads the module's `WORKLOAD` object
   from `kernel_library`; never redefine its graph or input specifications in
   the skill. Repair cycles do not consume improvement rounds. Runs default to
   `$XDG_STATE_HOME/develop-nkigym/<repository>/` when `XDG_STATE_HOME` is set and
   `~/.local/state/develop-nkigym/<repository>/` otherwise.

Use the `worktree` from status for every inspection and edit. Never edit the source checkout for a run. Do not commit,
clean, reset, or modify files under the run's artifact directories.

## Follow status

Use `mode` to distinguish exhaustive bug repair from focused improvement. Treat `next_action` as authoritative. With an
unchanged worktree, the only exception is retrying `validate` or `tune` after confirming a transient failure:

- `validate`: Run `develop.py validate RUN_DIRECTORY`. This runs the canonical implementation-health gates against the
  unchanged baseline before tuning. A failure changes the run to `mode=repair` and `next_action=edit`.
- `tune`: Run `develop.py tune RUN_DIRECTORY`. Inspect `latest_tuning_log` after a failure. Retry the same command only
  for a transient failure; otherwise repair the reproducible bug.
- `edit` in `repair` mode: Read `failed_gate_log` or `latest_tuning_log`, diagnose the failure, and fix it in any
  implementation file needed under `nkigym/src/nkigym`. Run the next check even when the bug predates the run. If that
  check exposes another bug, fix that bug too. Continue until no canonical gate fails.
- `edit` in `improve` mode: Read the trace at `baseline_tuning_artifact_directory`, compare measured siblings,
  generated kernels, transform decisions, profile results, and compiler failures, and implement the single most
  important evidence-backed improvement. Defer every independent improvement to a later cycle.
- `check`: Run `develop.py check RUN_DIRECTORY`. This command owns the canonical acceptance gates and candidate tuning;
  do not invoke those tests separately. It materializes retained kernel-library endpoints before running the nkigym-only
  MFU test. In repair mode, keep alternating `edit` and `check` until all failures are gone. In improvement mode, limit
  corrections to the selected improvement. If the user forbids tests or hardware work, stop before this command and
  report the pending action.
- `accept`: Run `develop.py accept RUN_DIRECTORY` and report the promoted tree and evidence. If the resulting action is
  not `complete`, continue with the next cycle in the same invocation.
- `complete`: Report `completed_improvement_rounds`, the accepted changes, and the final evidence, then stop.

Run `status` after an interruption or whenever candidate state is uncertain. The command derives resume state from
`run.json`, immutable attempt artifacts, and the current worktree fingerprint. Do not preserve or reconstruct a Codex
thread ID, transcript, prompt, or conversational checkpoint. `historical_best_score` starts from the selected
workload's retained MFU and can only increase from accepted tuning evidence.

## Editing constraints

- Treat the cycle baseline as the starting state. Preserve all baseline content unrelated to the selected repair or
  improvement, including pre-existing staged, unstaged, deleted, and untracked changes; do not clean, reset, restore, or
  overwrite it.
- Make intentional implementation additions, edits, renames, and deletions only under `nkigym/src/nkigym` when required
  by the selected repair or improvement; `kernel_library` supplies workloads and `test` supplies acceptance gates.
- In repair mode, make every focused bug fix required to clear the canonical gates; a pre-existing bug is in scope.
- In improvement mode, make exactly one semantic improvement: the most important change supported by measured evidence.
- The public transform inventory may change as part of the selected repair or improvement: transforms may be added,
  removed, or consolidated. Keep every resulting public transform atomic and workload-generic, do not combine
  independent actions, and keep the public registry synchronized with the source inventory.
- Do not add transform-specific tests or restore deleted tests.
- Let failed checks remain immutable; make a correction and let the script allocate a new check directory.
- Before acceptance, leave the worktree exactly at the candidate that passed its check.
