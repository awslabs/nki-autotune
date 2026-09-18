## NKI Gym

Build, transform, validate, and remotely profile NKI kernels.

## Installation

Create the local environment and provision an SSH-accessible Trn2 host and CPU
simulation workers:

```bash
cd nki-autotune
./install.sh --host \
  gym-1 \
  gym-cpu-1 \
  gym-cpu-2 \
  gym-cpu-3 \
  gym-cpu-4
source ~/venvs/kernel-env/bin/activate
```

The installer creates `~/venvs/kernel-env` when missing locally and on each
host, installs the local checkout, and installs the profile and simulation
worker dependencies remotely. The `--host` option accepts one or more SSH
destinations. The installer requires `python3`, `ssh`, and `rsync`; the Trn2
host must already have the Neuron driver, runtime, and tools.

## Tests

Host-dependent tests have no repository defaults. Pass CPU SSH destinations
when running remote simulation coverage:

```bash
pytest --cpu-hosts gym-cpu-1 gym-cpu-2
```

The option accepts one or more hosts, and each CPU simulation batch uses all
configured hosts. Tests that do not use remote simulation can run without it.

The complete ladder benchmark replays the current best public-transform ladder
for each registered NAKB workload:

```bash
PYTHONPATH="$PWD:$PWD/nkigym/src" python -m pytest -s \
  test/test_nkigym_ladder.py \
  --cpu-hosts gym-cpu-1 gym-cpu-2 gym-cpu-3 gym-cpu-4 \
  --trn2-hosts gym-trn2-1
```

The development skill owns adaptive exploration and records confirmed ladders
in `kernel_library/_best_nkigym.py`. Pytest starts from the current canonical
IR, requires every recorded option to remain legal, renders the replayed
endpoint, and checks the workload's copied NAKB accuracy contract on Trn2. The
aggregate metric is the arithmetic mean of the per-configuration latency ratios:

```text
mean_relative_latency = mean(confirmed NKIGym latency / NAKB latency)
```

Each registered configuration has equal weight regardless of absolute latency.
Lower is better. Every workload must be correct, and `mean_relative_latency`
must not exceed the fixed target of `0.9`, equivalent to at least 10% average
latency reduction over NAKB. Individual regressions count as negative
improvements and remain in the mean. For example, 10% and 20% latency reductions
average to 15%. The benchmark prints each workload's latencies, the mean relative
latency, and the mean percentage latency reduction.

Hardware correctness uses a fresh random 63-bit input seed on every benchmark
run. The seed is printed. Set
`NKIGYM_NAKB_VALIDATION_SEED` to a recorded value only when reproducing a run.

CPU checks use the official
[`nki.simulate`](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.simulate.html)
API. Hardware profiles upload only the rendered `kernel.py`, stream its
workload metadata, and use
[`neuron-explorer capture`](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/use-neuron-profile.html)
to profile one execution.

Use the metrics API for hardware profiling:

```python
from pathlib import Path

from nkigym.profile import profile_metrics

metrics = profile_metrics(
    host="gym-1",
    kernel=Path("kernel.py").read_text(),
    func_name="nki_kernel",
    input_specs={"x": ((128, 512), "bfloat16")},
    cache_dir="/tmp/kernel-profile",
)

print(metrics.mfu_percent, metrics.latency_ms)
```

The cache directory contains the submitted kernel and request, transport and
compiler logs, `file.neff`, `profile.ntff`, and the JSON profiler summary. A
failed compile or profile raises an exception after preserving available
artifacts.

## Programmatic Synthesis

The synthesis API traces supported NumPy math with shape-only symbolic tensors
and lowers it directly to an `NKIOp` graph. It is deterministic and does not
call a model service:

```python
import numpy as np

from nkigym.synthesis import synthesize_numpy_to_nkigym


def f_numpy(lhs, rhs):
    return lhs.astype(np.float32) @ rhs.astype(np.float32)


kernel = synthesize_numpy_to_nkigym(
    f_numpy,
    {"lhs": ((2048, 2048), "bfloat16"), "rhs": ((2048, 2048), "bfloat16")},
)

source = kernel.source
```

The supported subset includes 2D transpose and matmul, scalar or per-row
broadcast arithmetic, common activations, and free-axis sum, maximum, and mean
reductions. Unsupported operations raise `ValueError`.

## Benchmark

`benchmark/` is created once and frozen. Every target contains a NAKB PyTorch
reference, tensor input specifications, a seeded input generator, a frozen
`AccuracySpec`, and a fixed NAKB baseline in `nakb_latency_ms`. `AccuracySpec`
contains `atol`, `rtol`, the comparison mode, and any output selection, views,
grouping, or per-output tolerances.

`benchmark.NAKB_WORKLOADS` contains 127 complete measured NAKB targets
grouped into 26 flat, self-contained Python modules by workload type. Static
numerical choices are bound into the callable, and configurations with
different callables, input specifications, generators, tolerances, or latency
records remain separate dictionaries. NAKB configurations without every
required field are not included. `benchmark.WORKLOADS` exposes only exact
aliases to entries in `NAKB_WORKLOADS`.

The repository structure test pins the benchmark contents with a SHA-256
checksum and forbids dependencies on `nkigym` or `kernel_library`. Backend
development must leave both the benchmark and its checksum unchanged.

The seeded generators retain NAKB's NumPy input-generation convention.
`TorchReference` applies NAKB's NumPy-to-Torch argument conversion before
calling the copied golden:

```python
from benchmark import NAKB_WORKLOADS

workload = NAKB_WORKLOADS["cumsum"][0]
inputs = workload["input_generator"](workload["input_specs"], seed=0)
outputs = workload["torch_ref"](**inputs)
```

## Kernel Library

`kernel_library/_best_nkigym.py` records the best NKIGym kernels as replayable
public-transform ladders. A missing record means the canonical empty ladder.
Kernel development updates these records while the benchmark stays frozen.

The ladder benchmark synthesizes the current canonical lowering and replays
each recorded ladder, requiring every step to be a currently legal transform
option. Each intermediate state is checked using the target's `AccuracySpec`.
Final acceptance calls `nkigym.profile.profile_metrics` with exact inputs to
compile, capture outputs, and profile the same NEFF, then checks those outputs
against the target's reference and accuracy criteria.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
