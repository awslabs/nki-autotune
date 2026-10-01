# Contributing to NKIGym

Start with the [project goal and local installation](README.md), then run the
[usage example](docs/usage.md). Read [AGENTS.md](AGENTS.md) and
[the project learnings](.agents/rules/learnings.md) before changing the backend.

## Architecture

The main path is a PyTorch function, a synthesized operation graph, a canonical
IR, a sequence of transformations, and rendered NKI source.

| Package under `nkigym/src/nkigym/` | Responsibility |
| --- | --- |
| `synthesis/` | Trace supported PyTorch computations and return a generated `f_nkigym` function with input/output adapters. |
| `ops/` | Define operations, their numerical behavior, and their NKI instruction contracts. |
| `ir/` | Build and represent schedule trees, buffers, dimensions, and dependencies. `ir/arith/` provides standalone arithmetic analysis derived from TVM. |
| `transforms/` | Enumerate and apply legal atomic rewrites through `analyze` and `apply`. |
| `codegen/` | Render the IR to NKI and support PyTorch lowering. |
| `profile/` | Simulate generated kernels and compile, execute, and profile them on caller-supplied hardware. |

The IR contains blocks, loops, and instruction nodes. Dependency analysis tracks
the ordering of reads and writes. Code generation follows the resulting
schedule; optimization choices belong in explicit transformations.
TVM is not an installation dependency.

Repository imports have one direction: `benchmark` imports only itself;
`nkigym` imports only itself; `kernel_library` and tests may import both.
The backend must not recognize a benchmark configuration by name or import its
reference computation.

## Development workflow

1. Reproduce the issue using the current checkout. Record the configuration,
   command, seed, and failure artifacts. Keep experiments in an ignored cache
   or outside the source directories.
2. Identify the responsible layer. Fix tracing in synthesis, instruction
   contracts in operations, scheduling legality in transforms/IR, and source
   emission in code generation.
3. Make a focused change. Preserve dtype conversions, rounding, reduction
   order, dependency direction, and every intermediate kernel's semantics.
4. Run the relevant checks below. Add regression coverage for behavior changes,
   including rejected illegal transformations when applicable.
5. For performance work, record the confirmed ladder in
   `kernel_library/_best_nkigym.py` and publish its existing validation result.
   Follow [the validation guide](docs/validation.md) for hardware acceptance
   and measurement provenance.

For optimization, start with the configuration having the largest saved
`NKIGym / NAKB` latency ratio. Compare its generated structure with the matching
expert implementation and express improvements through generic transformations.
Re-rank after accepted changes. Missing or stale timings do not establish the
current performance of a kernel.

## Transform contract

Every public transform directly defines typed `analyze(ir)` and
`apply(ir, option)` methods:

- `analyze` returns legal options for the current IR.
- `apply` rechecks legality, copies the IR, applies one decision, and returns
  the new IR. Invalid options fail explicitly.
- Every intermediate remains correct. A later transform cannot repair an
  earlier semantic violation.
- Each decision is atomic and reusable across workloads. Separate tiling,
  reordering, placement, forwarding, and deletion choices remain separate
  actions.
- Legality covers dependencies and instruction constraints. A legal kernel
  may exceed hardware memory capacity; compilation and profiling determine
  whether it can run efficiently.

Keep one public transform per module, shared helpers under `transforms/helper/`,
and the ordered registry in `transforms/__init__.py` consistent with the
implementations. Operation modules each own one `NKIOp` subclass. Consult
[code-motion legality](nkigym/src/nkigym/transforms/code_motion_legality.md)
when changing movement or dependency rules.

The [repository structure test](test/test_repository_structure.py) enforces
package boundaries, source layout, formatting, and source-size limits,
including at most 30 public transforms and 3,000 IR code lines. Preserve these
gates and the frozen benchmark checksum.

## Testing

Activate the environment and run from the repository root. Select test paths
explicitly for the work being done.

| Tests | Requirements and coverage |
| --- | --- |
| `test/test_repository_structure.py` | Local checks for formatting, imports, layout, source limits, transform APIs, and the frozen benchmark checksum. |
| `test/test_nakb_accuracy.py` | Local checks for the frozen numerical comparison rules. |
| `test/test_nakb_comparison.py` | Local checks for saved-result publication and plotting using synthetic measurements. |
| `test/test_random_torch_synthesis.py` | Local PyTorch synthesis and operation-graph correctness; prints reproduction seeds. |
| `test/test_random_rollout.py` | Requires `--cpu-hosts`. Runs 500 transforms on the first configuration of each selected workload and simulates every 50 steps, within one 600-second deadline. `hf_ffn` is excluded from this rollout coverage. |
| `test/test_nkigym_ladder.py` | Requires `--cpu-hosts` and `--trn2-hosts`. Replays all recorded ladders, validates intermediates and hardware outputs, saves latencies, and enforces the aggregate target. |
| `test/test_transform_evaluation.py` | Requires the `codex` executable configured for model access. Launches model-based reviews of transform atomicity and genericity. |

Run the local checks:

```bash
python -m pytest -q \
  test/test_repository_structure.py \
  test/test_nakb_accuracy.py \
  test/test_nakb_comparison.py \
  test/test_random_torch_synthesis.py
```

For remote coverage, first follow [worker setup](docs/usage.md#worker-setup).
The host variables below are documentation conveniences, passed explicitly to
pytest; they are not automatic configuration.

```bash
python -m pytest -s test/test_random_rollout.py \
  --cpu-hosts "$NKIGYM_CPU_HOST"
```

Run the complete suite only with both kinds of workers and the Codex CLI ready:

```bash
python -m pytest -s test \
  --cpu-hosts "$NKIGYM_CPU_HOST" \
  --trn2-hosts "$NKIGYM_TRN2_HOST"
```

Put test paths before host options: each host option consumes one or more
arguments. Missing required hosts cause a pytest usage error. Keep the backend
and ladders fixed during acceptance, and avoid competing jobs during timed
rollouts. Local checks alone do not establish full-suite acceptance.

## Code style

Use type annotations and accurate docstrings. Keep changes focused, handle
unexpected values explicitly, and keep imports at module scope. Format changed
Python files with Black and isort using the root `pyproject.toml`.

```bash
python -m isort --check-only nkigym/src kernel_library test
python -m black --check nkigym/src kernel_library test
```

The structure test also checks the frozen benchmark and comparison plotting
script. Do not reformat the frozen benchmark as part of backend development.
Optional contributor tools can be installed with
`python -m pip install pre-commit pyright`. Run `pyright` on changed Python
modules; `pre-commit install` enables the formatting hooks.

## Issues and pull requests

For a bug, include a minimal reproduction, commit/revision, dependency versions,
test or profiling command, configuration ID, seed, and relevant error output.
For a performance change, include both latencies and the saved result's
provenance. Remove machine names, personal paths, and access details from shared
logs.

A pull request should explain the problem, resulting behavior, and validation.
State which checks ran and which hardware or model-dependent checks did not.
Keep benchmark revisions separate from backend fixes, and discuss substantial
design changes before implementing them.

## Code of conduct

This project follows the [Amazon Open Source Code of Conduct](CODE_OF_CONDUCT.md).

## Security issue notifications

Report potential vulnerabilities through the
[AWS vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/).
Please do not open a public issue for a security vulnerability.

## Licensing

Contributions are covered by the project's [Apache-2.0 license](LICENSE).
