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

## Agent Workflows

Use `$debug-nkigym` to run the standalone repository tests directly in the
current checkout and fix implementation bugs until they pass. Use
`$self-evolve` separately to refine IR, operations, code generation, and
transforms from measured search feedback. It first runs the standalone tests as
a read-only health check. If a test fails, it stops and reports the failure
without repairing it or invoking `$debug-nkigym`. Refinement edits happen
directly in the current branch checkout. Run artifacts remain under the external
state directory; no detached Git worktree is created:

```bash
python .agents/skills/self-evolve/scripts/develop.py start matmul-lhs-t \
  --shape m2048_k2048_n2048 \
  --host gym-1 \
  --rounds 3
```

If a run later reports `mode=repair`, `$self-evolve` stops and reports the
failure. Debugging requires a separate, explicit `$debug-nkigym` invocation.

The selectable kernel-library workloads are `attention`, `matmul-lhs`,
`matmul-lhs-t`, and `rmsnorm-matmul`. Every selection also requires one of
the shape keys registered for that workload. See the
[`self-evolve` skill](.agents/skills/self-evolve/SKILL.md) for refinement and
the [`debug-nkigym` skill](.agents/skills/debug-nkigym/SKILL.md) for repair.

## Kernel Library

Each flat kernel-library module owns one exact `(workload, shape)` tuple,
including its NumPy reference, input specifications, seeded random input
generator, validation tolerances, `f_nkigym` graph, and transform ladder. The
module dumps, CPU-verifies, and profiles every intermediate kernel:

```bash
PYTHONPATH=.:nkigym/src \
  python kernel_library/matmul_lhs_t_rhs_m2048_k2048_n2048.py \
  --host gym-1 \
  --cache /tmp/matmul-lhsT-rhs

PYTHONPATH=.:nkigym/src \
  python kernel_library/matmul_lhs_rhs_m2048_k2048_n2048.py \
  --host gym-1 \
  --cache /tmp/matmul-lhs-rhs

PYTHONPATH=.:nkigym/src \
  python kernel_library/rmsnorm_matmul_m2048_k2048_n2048.py \
  --host gym-1 \
  --cache /tmp/rmsnorm-matmul

PYTHONPATH=.:nkigym/src \
  python kernel_library/attention_q16384_kv16384_d128.py \
  --host gym-1 \
  --cache /tmp/online-fusion-attention
```

Kernels and accuracy results are stored under `kernels/`; remote MFU results
are stored under `mfu/`.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
