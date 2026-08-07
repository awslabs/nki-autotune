---
name: self-evolve
description: Execute a user-selected number of focused nkigym refinement rounds directly in the current branch checkout. Use for starting or resuming development on a healthy repository, comparing measured candidates, implementing one evidence-backed refinement, and recording checked candidates without creating detached worktrees. Run a read-only test preflight and abort without repairs if it fails.
---

# Self-evolve nkigym

Use `scripts/develop.py` for durable artifacts and exact-tree checks. Own the evidence analysis and edit the current
branch checkout directly.

## Establish the refinement run

1. Work from the repository containing this skill.
2. Use `~/venvs/kernel-env/bin/python` when it exists; otherwise use the active Python interpreter.
3. Read `.agents/rules/learnings.md` before editing.
4. Inspect `git status --short` and preserve every pre-existing staged, unstaged, deleted, and untracked change.
5. Before starting or resuming a run, execute these standalone tests in order:

   ```bash
   python -m pytest -q test/test_repository_structure.py -s
   python -m pytest -q test/test_random_rollout.py -s
   python -m pytest -q test/test_transform_evaluation.py -s
   ```

   Treat this as a read-only health check. On the first failure, stop immediately and alert the user with the failing
   command and relevant output. Do not diagnose the bug, edit files, or create or resume a run.
6. Resume a supplied run with:

   ```bash
   python .agents/skills/self-evolve/scripts/develop.py status RUN_DIRECTORY
   ```

7. When no run exists, require a kernel-library workload, shape, and profile host. Ask the user how many
   refinement rounds to run; require an explicit positive integer and do not choose a default. Then create the run:

   ```bash
   python .agents/skills/self-evolve/scripts/develop.py start WORKLOAD \
     --shape SHAPE \
     --host HOST \
     --rounds ROUNDS
   ```

   Choose `WORKLOAD` from `attention`, `matmul-lhs`, `matmul-lhs-t`, or
   `rmsnorm-matmul`. Choose `SHAPE` from the tuples registered by
   `kernel_library`. The support script loads that exact tuple;
   never redefine its graph or input specifications in the skill. Runs default to
   `$XDG_STATE_HOME/self-evolve/<repository>/` when `XDG_STATE_HOME` is set and
   `~/.local/state/self-evolve/<repository>/` otherwise. The run stores artifacts there but records the current source
   checkout as `worktree`; it never creates another Git worktree.

Use the current source checkout for every inspection and edit. Require the `worktree` reported by status to equal that
checkout. Do not switch branches, commit, clean, reset, or modify files under the run's artifact directories.

## Follow refinement status

Treat `mode` and `next_action` as authoritative:

- `mode=repair`: Abort the self-evolve session. Report `run_directory`, the failing stage, and `failed_gate_log` or
  `latest_tuning_log`. Make no repair edits.
- `validate`: Run `develop.py validate RUN_DIRECTORY` against the unchanged baseline. If the returned status enters
  repair mode, abort and report it immediately.
- `tune`: Run `develop.py tune RUN_DIRECTORY` to collect baseline feedback. If the returned status enters repair mode,
  abort and report it immediately.
- `edit`: Require `mode=improve`. Read the trace at `baseline_tuning_artifact_directory`; compare measured siblings,
  generated kernels, transform decisions, profile results, and compiler failures; then implement the single most
  important evidence-backed refinement. Defer every independent refinement to a later cycle.
- `check`: Require `mode=improve`, then run `develop.py check RUN_DIRECTORY`. This command owns the canonical acceptance
  gates and candidate tuning; do not invoke those tests separately. Limit corrections after a failed check to the
  selected refinement. If the status enters repair mode, abort and report it. If the user forbids tests or hardware
  work, stop before this command and report the pending action.
- `accept`: Require `mode=improve`, run `develop.py accept RUN_DIRECTORY`, and report the promoted tree and evidence. If
  the resulting action is not `complete`, continue with the next refinement cycle in the same invocation. Acceptance
  records the checked tree as the next baseline but does not commit or rewrite the branch or index.
- `complete`: Report `completed_improvement_rounds`, the accepted changes, and the final evidence, then stop.

Run `status` after an interruption or whenever candidate state is uncertain. The command derives resume state from
`run.json`, immutable attempt artifacts, and the current worktree fingerprint. Do not preserve or reconstruct a Codex
thread ID, transcript, prompt, or conversational checkpoint. `historical_best_score` starts from the selected
workload's retained MFU and can only increase from accepted tuning evidence.

## Editing constraints

- Treat the cycle baseline as the starting state. Preserve all baseline content unrelated to the selected refinement,
  including pre-existing staged, unstaged, deleted, and untracked changes; do not clean, reset, restore, or overwrite it.
- Make intentional implementation additions, edits, renames, and deletions only under `nkigym/src/nkigym` when required
  by the selected refinement; `kernel_library` supplies workloads and `test` supplies acceptance gates.
- Make exactly one semantic refinement: the most important change supported by measured feedback.
- The public transform inventory may change as part of the selected refinement: transforms may be added,
  removed, or consolidated. Keep every resulting public transform atomic and workload-generic, do not combine
  independent actions, and keep the public registry synchronized with the source inventory.
- Do not add transform-specific tests or restore deleted tests.
- Do not diagnose or repair a failing preflight, baseline validation, or baseline tuning run. Stop and alert the user.
- Let failed checks remain immutable; make a correction and let the script allocate a new check directory.
- Before acceptance, leave the source checkout exactly at the candidate that passed its check.
