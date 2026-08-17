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

Host-dependent acceptance tests have no repository defaults. Pass the Trn2 and
CPU SSH destinations on every run:

```bash
pytest \
  --trn2-hosts gym-trn2-1 gym-trn2-2 \
  --cpu-hosts gym-cpu-1 gym-cpu-2
```

Each option accepts one or more hosts. Search workloads are distributed across
the Trn2 hosts, while each CPU simulation batch uses all CPU hosts. Tests that
do not use remote hosts can run without these options.

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

Use `profile_many()` for independent kernels on the same Trn2 host. It uploads
the batch once, compiles kernels in separate processes, and profiles each
process on a distinct logical NeuronCore:

```python
from nkigym.profile import profile_many

result = profile_many(
    host="gym-1",
    kernels={"candidate-0": Path("kernel.py").read_text()},
    func_name="nki_kernel",
    input_specs={"x": ((128, 512), "bfloat16")},
    cache_dir="/tmp/kernel-profiles",
)
```

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

## Kernel Library

Every workload is an exact seven-field dictionary containing a copied NAKB
PyTorch golden reference, tensor input specifications, seeded input generator,
correctness tolerances, a fixed NAKB baseline in `nakb_latency_ms`, and one
best historical latency.

`kernel_library.NAKB_WORKLOADS` contains 127 complete measured NAKB targets
grouped into 26 flat, self-contained Python modules by workload type. Static
numerical choices are bound into the callable, and configurations with
different callables, input specifications, generators, tolerances, or latency
records remain separate dictionaries. NAKB configurations without every
required field are not included. `kernel_library.WORKLOADS` exposes only exact
aliases to entries in `NAKB_WORKLOADS`.

The seeded generators retain NAKB's NumPy input-generation convention.
`TorchReference` applies NAKB's NumPy-to-Torch argument conversion before
calling the copied golden:

```python
from kernel_library import NAKB_WORKLOADS

workload = NAKB_WORKLOADS["cumsum"][0]
inputs = workload["input_generator"](workload["input_specs"], seed=0)
outputs = workload["torch_ref"](**inputs)
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
