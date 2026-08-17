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

CPU checks use the official
[`nki.simulate`](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.simulate.html)
API. Hardware profiles upload only the rendered `kernel.py`, stream its
workload metadata, and use
[`neuron-explorer capture`](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/use-neuron-profile.html)
to profile one execution.

Use the top-level function for hardware profiling:

```python
from pathlib import Path

from nkigym.profile import profile

mfu_percent, latency_ms = profile(
    host="gym-1",
    kernel=Path("kernel.py").read_text(),
    func_name="nki_kernel",
    input_specs={"x": ((128, 512), "bfloat16")},
    cache_dir="/tmp/kernel-profile",
)
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

from nkigym.synthesis import compile_numpy_to_nkigym


def f_numpy(lhs, rhs):
    return lhs.astype(np.float32) @ rhs.astype(np.float32)


source = compile_numpy_to_nkigym(
    f_numpy,
    {"lhs": ((2048, 2048), "bfloat16"), "rhs": ((2048, 2048), "bfloat16")},
)
```

The supported subset includes 2D transpose and matmul, scalar or per-row
broadcast arithmetic, common activations, and free-axis sum, maximum, and mean
reductions. Unsupported operations raise `ValueError`.

## Kernel Library

Each module exposes one dictionary containing the NumPy reference, fixed input
specifications, seeded input generator, correctness tolerances, historical MFU,
and retained transform trace. `kernel_library.WORKLOADS` discovers modules
automatically. Tests synthesize kernels directly from these dictionaries.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
