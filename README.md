# NKIGym

NKIGym is a research system for developing AWS Neuron Kernel Interface (NKI)
kernels for Trainium2 (Trn2). It lowers a PyTorch computation to a canonical
kernel, exposes explicit program transformations, and provides CPU validation
and hardware profiling.

## Target goal

The goal is expert-level NKI performance without expert-level source engineering
for each workload. A developer or profiler-guided agent should be able to
reconstruct efficient kernels using the same small set of reusable
transformations across workloads.

Each transformation makes one semantics-preserving decision, such as splitting
a loop, moving an operation, changing buffer placement, or introducing software
pipelining. A **transform ladder** is an ordered sequence of these decisions,
recorded so it can be replayed from the canonical kernel. Workload-specific
transform implementations and arbitrary edits to generated kernels do not
satisfy this goal.

The evaluation target is:

- Correct intermediate transformations and final hardware outputs under the
  frozen benchmark's accuracy contracts.
- An arithmetic mean of `NKIGym latency / NAKB latency` of at most **0.9** across
  every registered configuration, with equal weight and regressions included.
- Reuse of at most 30 public, atomic, workload-independent transformations.

These are research objectives and acceptance criteria, not a claim that the
current implementation meets them. The project supports a subset of PyTorch
tensor computations. Synthesis and ladder replay are deterministic and do not
invoke a model; an optimization policy can choose transformations externally.
See [validation and saved results](docs/validation.md) for how to assess progress.

## Local quick start

Run commands from the repository root. Use Python 3.10 or newer on a platform
with compatible Neuron wheels. The package currently requires NKI 0.5.x and
neuronx-cc 2.26.x; the constraints live in
[nkigym/pyproject.toml](nkigym/pyproject.toml).

```bash
python3 -m venv ~/venvs/kernel-env
source ~/venvs/kernel-env/bin/activate
python -m pip install --only-binary=:all: \
  --extra-index-url https://pip.repos.neuron.amazonaws.com \
  -e ./nkigym black isort matplotlib pytest pytest-timeout
```

Generate a canonical kernel from a small frozen workload:

```bash
python - <<'PY'
from benchmark import NAKB_WORKLOADS
from nkigym.ir import build_initial_ir
from nkigym.synthesis import synthesize_torch_to_nkigym

workload = NAKB_WORKLOADS["cumsum"][0]
artifact = synthesize_torch_to_nkigym(workload["torch_ref"], workload["input_specs"])
ir = build_initial_ir(artifact.function, artifact.input_specs)
ir.dump(".cache/quickstart")
print("Generated .cache/quickstart/kernel.py and envelope.md")
PY
```

This runs locally without a Trn2 host. It produces NKI source and an
intermediate representation (IR) summary; it does not establish hardware
correctness or performance. The [usage guide](docs/usage.md) covers custom
PyTorch functions, transformation, simulation, and profiling.

Run the local accuracy-contract and saved-result checks:

```bash
python -m pytest -q test/test_nakb_accuracy.py test/test_nakb_comparison.py
```

The full suite has additional CPU worker, Trn2, and Codex requirements. See
[the development guide](CONTRIBUTING.md#testing) before running it.

## Documentation

| Task | Guide |
| --- | --- |
| Synthesize, transform, simulate, and profile a kernel | [Usage](docs/usage.md) |
| Understand the architecture and develop the backend | [Contributing](CONTRIBUTING.md) |
| Validate ladders, publish measurements, and plot saved results | [Validation and results](docs/validation.md) |
| Find upstream documentation and reference implementations | [References](.agents/rules/references.md) |
| Work with repository agents | [Agent guidance](AGENTS.md) |

## Repository layout

| Path | Purpose |
| --- | --- |
| `nkigym/src/nkigym/` | Synthesis, IR, operations, transforms, NKI code generation, simulation, and profiling. |
| `benchmark/` | Frozen PyTorch references, input generators, accuracy contracts, and NAKB baseline latencies: 127 configurations in 26 workload families. |
| `kernel_library/_best_nkigym.py` | Best recorded transform ladders, keyed by workload configuration. |
| `kernel_library/nakb_comparison.py` | Saved-measurement validation, publication, and statistics. |
| `test/` | Structure, accuracy, synthesis, transformation, and hardware acceptance tests. |
| `docs/` | Usage and validation guides. |
| `.agents/` | Repository guidance and optional development/comparison skills. |
| `artifacts/nakb_latency_comparison/` | Generated measurement records and run history; ignored by Git. |

`benchmark/` is independent of the backend and kernel library. Its contents and
the checksum in `test/test_repository_structure.py` are frozen. Revising the
benchmark requires an explicit benchmark change.

## Security and license

See [security issue notifications](CONTRIBUTING.md#security-issue-notifications)
for vulnerability reporting. This project uses the
[Apache-2.0 license](LICENSE).
