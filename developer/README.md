# developer

`developer` is a thin, artifact-driven orchestration loop for continuously
improving nkigym IR and transforms for an `f_nkigym` operator graph. Program
packaging, agentic tuning, search policy, and profiling live in
`nkigym.search`; `developer` coordinates worktrees, editing turns, gates, and
promotion between cycles.

The public pipeline is:

1. Validate the `f_nkigym` decorator, source, parameters, and `input_specs`.
2. Persist a standalone `f_nkigym` module, snapshot the current non-ignored
   workspace, and create a detached candidate worktree from that exact tree.
3. Make one Python agentic-tuning call that profiles each state while a
   tool-free Codex policy explores a measured search tree. It may apply up to
   three compatible legal transforms or revisit an earlier branch point.
4. Give the measured trace to a code-editing Codex session for one focused IR,
   operation, codegen, or transform improvement.
5. Evaluate the candidate with five pytest gates: source-size and transform API
   limits, random rollout correctness, an agentic review of every public
   transform, best-known-kernel MFU regression, and target-workload agentic
   tuning.
6. Let the final pytest gate require valid measured evidence within 1.0 MFU
   percentage point of the run's historical best; an increase is recorded but
   is not required.
7. Resume the same editing thread until every check passes and tuning remains
   within the allowed MFU fluctuation.
8. Promote the accepted candidate to the next internal baseline, start a fresh
   editing thread, and repeat steps 3-7 until explicitly stopped.

The accepted source tree and latest tuning trace advance after every cycle, but
the MFU comparison reference never decreases. This prevents tolerated
measurement fluctuations from accumulating across cycles.

## Usage

Run one of the fixed production-shape drivers:

```bash
python -m developer.drivers.matmul --host gym-1
python -m developer.drivers.rmsnorm --host gym-1
python -m developer.drivers.attention --host gym-1
```

Each module owns its `f_nkigym` graph and `INPUT_SPECS`. For a custom workload,
define a decorated `f_nkigym` whose parameter names and order exactly match
`input_specs`, then call `developer`:

```python
from developer import developer
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy

INPUT_SPECS = {
    "lhs_T": ((2048, 2048), "bfloat16"),
    "rhs": ((2048, 2048), "bfloat16"),
}


@nkigym_kernel
def f_nkigym(lhs_T, rhs):
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_product = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_product = NKITensorCopy()(src=psum_product)
    hbm_output = NKIStore()(src=sbuf_product)
    return hbm_output


result = developer(f_nkigym, INPUT_SPECS, "gym-1")
```

Runs are continuous by default. Press Ctrl-C to finish with a durable `stopped`
result after preserving completed cycles, partial-cycle artifacts, and the
candidate worktree.

The fixed drivers print timestamped progress for run setup, each measured
refinement state, Codex editing turns, quality gates, MFU changes, retries, and
accepted cycles. Each update includes the relevant artifact or log path.

## Supported Workloads

The callable has no workload-name catalog. It accepts static, single-output
tensor DAGs expressed with the current `NKIOp` vocabulary.

| Driver | Workload | Current graph |
|---|---|---|
| `developer.drivers.matmul` | `lhs_T.T @ rhs` | load, matmul, drain, store |
| `developer.drivers.rmsnorm` | row-wise RMSNorm followed by matmul | square-reduce, `rsqrt`, broadcast multiply, transpose, matmul |
| `developer.drivers.attention` | scaled dot-product attention | two matmuls plus materialized max, exponent, sum, reciprocal, broadcast, and transpose stages |

Supported operations are load, store, copy, matmul, transpose, activation,
reduction, and tensor-scalar operations. Inputs must use shapes accepted by
the canonical NKI tiler; the target BF16 workloads use two-dimensional extents
of at least 128 with partition and matmul axes aligned to their canonical tile
sizes. General Python control flow, tuple or multiple outputs, and operations
outside the `NKIOp` vocabulary are not represented.

Install and provision the profile host once:

```bash
./install.sh --host gym-1
source ~/venvs/kernel-env/bin/activate
```

The tuning policy and code-editing sessions use the default model through
Bedrock; the controller does not detect or pin a model.

The detached candidate includes the current non-ignored workspace: committed,
staged, unstaged, deleted, and untracked files are captured without changing
the source index or files. Ignored untracked files are excluded. Committing
before a run is not required.

## Agentic Tuning Contract

Each tuning phase is one Python call to
`nkigym.search.run_agentic_tuning(...)`. The function runs the following
isolated command from the candidate worktree:

```text
python -m nkigym.search.agentic_tuning_cli \
  --host HOST \
  --trace-dir TRACE_DIR \
  --program-dir PROGRAM_DIR
```

`PROGRAM_DIR` contains the standalone `f_nkigym` source and structured input
specifications. The command imports the callable and invokes
`nkigym.search.run_profiled_refinement`, which constructs `KernelMDP`, the
transform registry, the Codex policy, and the profile evaluator. `TRACE_DIR`
contains at least:

```text
result.json
nodes/node_000/evaluation.json
```

Every node stores its kernel, evaluation, profile artifacts, parent node, and
incoming transform. One node is active. Each policy turn receives the complete
measured trace plus the active node's full NKI, profile, buffers, and unexplored
actions. The policy may revisit another measured node with unexplored actions
without profiling it; that node's full details become active on the next turn.
Applying a transform then creates a new child, so an irreversible transform
does not make earlier choices final. Already measured parent-transform edges
are omitted, and render-identical states reuse their measured evaluation.

The policy continues until it finishes or no measured node has an unexplored
legal transform. Developer runs do not impose a total transform-count limit;
the controller's tuning subprocess timeout remains the wall-clock bound.
Standalone callers can use `--max-reasoning-steps` for a finite decision limit.

Each tuning call receives a private copy of `PROGRAM_DIR`. The controller
rejects its evidence if the subprocess changes any file in that copy, so
baseline and candidate scores always use the persisted target program.
For candidate attempts, the controller writes the program directory, cycle
baseline tree, serialized agentic tuning command, and historical best MFU to
`agentic-tuning-context.json`. It passes that path to
`test_agentic_tuning.py`, which runs tuning and writes its result under the
gate artifact directory.

## Quality Gates

Candidate changes must include a Python implementation change in the nkigym
execution stack. Checks then run in this order:

1. Deterministic limits on public transform count, one transform per bounded
   source file, the typed `analyze`/`apply` API, and total IR implementation
   size.
2. 500-step random rollouts across five per-run randomized aligned shapes each
   for lhsT matmul, lhs matmul, attention, and RMSNorm+matmul, numerically
   checking every intermediate state against NumPy. The run and rollout seeds
   are logged for replay, and each rollout follows one action path without
   branch fan-out.
3. Read-only agentic evaluation that returns one source-cited assessment for
   every public transform. Both its atomicity and genericity verdicts must
   pass.
4. Hardware MFU regression for the best-known generated matmul,
   RMSNorm+matmul, and attention endpoints.
5. Agentic tuning of the target workload in the edited worktree.

Checks stop at the first failure, but the editing cycle does not: failure
artifacts are returned to the same Codex thread until all checks run and pass.
The five tests under `test/` are the complete repository acceptance suite.
Missing, composite, convenience-wrapper, workload-specific, or indeterminate
transform assessments fail the transform evaluation.
`test_agentic_tuning.py` owns the requirement that tuning complete without
modifying source, produce at least one valid measured score, and remain within
1.0 MFU percentage point of the best MFU observed over the full run. Developer
uses only the five pytest exit statuses for candidate acceptance, then reads
the passing tuning artifact to initialize the next cycle.

## Artifacts

Artifacts default to `$XDG_STATE_HOME/developer/<repository>/` or
`~/.local/state/developer/<repository>/`:

```text
<run-id>/
  run.json
  final.patch
  program/
    f_nkigym.py
    program.json
  worktree/
  agentic-tuning/
    initial/
      program/
        f_nkigym.py
        program.json
      agentic-tuning.log
      result.json
      trace/
        result.json
        observations/
        decisions/
        nodes/
  cycles/000/
    prompt.md
    attempts/000/
      prompt.md
      codex-events.jsonl
      final-message.md
      diff.patch
      changed-files.json
      agentic-tuning-context.json
      gates/
        code-bloat.log
        random-rollout-correctness.log
        transform-evaluation.log
        transform-evaluation-artifacts/
          transform-evaluation.json
          transform-evaluation-response.json
        mfu-regression.log
        mfu-regression-artifacts/
        agentic-tuning.log
        agentic-tuning-artifacts/
          program/
            f_nkigym.py
            program.json
          agentic-tuning.log
          result.json
          trace/
      result.json
```

Inspect a running or stopped run with:

```bash
python -m developer status PATH_TO_RUN
```

`run.json` records both the committed `base_sha` and the exact `initial_tree`
captured from the workspace. `final.patch` is the complete `base_sha`-to-final
candidate patch, including both the initial workspace content and accepted
developer changes.

The initial workspace and accepted candidates become Git tree baselines.
Content-addressed refs under `refs/nki-autotune/baselines/` keep those trees
reachable across Git garbage collection. The controller does not create or move
a branch, commit to the source repository, modify the source index or files, or
clean up the detached worktree.

## Trust Boundary

Transform-policy turns have no tools. The transform evaluator has read-only
repository tools, and the code-editing Codex session uses the `workspace-write`
sandbox. Agentic tuning, SSH profiles, and command gates run candidate code
with the controller user's process permissions. Use trusted repositories,
programs, and profile hosts.
