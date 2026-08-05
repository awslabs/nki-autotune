## NKI Gym

Build, transform, validate, and remotely profile NKI kernels.

## Installation

Create the local environment and provision an SSH-accessible Trn2 host:

```bash
cd nki-autotune
./install.sh --host gym-1
source ~/venvs/kernel-env/bin/activate
```

The installer creates `~/venvs/kernel-env` when missing on both machines,
installs the local checkout, and installs the profile worker remotely. It
requires `python3`, `ssh`, and `rsync`; the host must already have the Neuron
driver, runtime, and tools.

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

## Developer

The callable developer pipeline accepts an `f_nkigym` operator graph and input
specifications, searches measured transform states, and uses the resulting
evidence to improve IR, operations, code generation, and transforms. Every
candidate must pass all checks and improve the same target; each accepted
candidate becomes the baseline for another cycle until the run is interrupted:

```bash
python -m developer.drivers.matmul --host gym-1
python -m developer.drivers.rmsnorm --host gym-1
python -m developer.drivers.attention --host gym-1
```

Custom workloads use the callable API:

```python
from developer import developer

result = developer(f_nkigym, input_specs, "gym-1")
```

See [`developer/README.md`](developer/README.md) for complete usage and the
artifact and validation contracts.

## Kernel Library

Each kernel-library module defines one `f_nkigym` graph and a fixed transform
ladder. It dumps, CPU-verifies, and profiles every intermediate kernel:

```bash
PYTHONPATH=.:nkigym/src \
  python kernel_library/matmul/lhsT_rhs/ladder.py \
  --host gym-1 \
  --cache /tmp/matmul-lhsT-rhs

PYTHONPATH=.:nkigym/src \
  python kernel_library/matmul/lhs_rhs/ladder.py \
  --host gym-1 \
  --cache /tmp/matmul-lhs-rhs

PYTHONPATH=.:nkigym/src \
  python kernel_library/rmsnorm_matmul/ladder.py \
  --host gym-1 \
  --cache /tmp/rmsnorm-matmul

PYTHONPATH=.:nkigym/src \
  python kernel_library/attention/ladder.py \
  --host gym-1 \
  --cache /tmp/online-fusion-attention
```

Kernels and accuracy results are stored under `kernels/`; remote MFU results
are stored under `mfu/`.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
