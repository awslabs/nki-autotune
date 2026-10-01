# Using NKIGym

Complete the [local installation](../README.md#local-quick-start) first. Run
commands from the repository root with the virtual environment activated.

## Synthesize a PyTorch function

NKIGym accepts supported PyTorch tensor computations and explicit input shapes
and dtypes. The input specification's names and order must match the function
parameters. Synthesis traces the computation, lowers it to an operation graph,
and checks that graph numerically on deterministic development inputs.

Create `.cache/` with `mkdir -p .cache`. Save the following as
`.cache/tanh_add.py` and run `python .cache/tanh_add.py`:

```python
import numpy as np
import torch

from nkigym.ir import build_initial_ir
from nkigym.synthesis import synthesize_torch_to_nkigym


def f_torch(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Add two tensors and apply a hyperbolic tangent."""
    return torch.tanh(lhs + rhs)


input_specs = {
    "lhs": ((128, 512), "float32"),
    "rhs": ((128, 512), "float32"),
}
artifact = synthesize_torch_to_nkigym(f_torch, input_specs)
ir = build_initial_ir(artifact.function, artifact.input_specs)
ir.dump(".cache/tanh_add/canonical")

generator = torch.Generator().manual_seed(0)
reference_inputs = {
    name: torch.randn(shape, generator=generator)
    for name, (shape, _) in input_specs.items()
}
inputs = artifact.adapt_inputs(reference_inputs)
expected = artifact.adapt_output(f_torch(**reference_inputs))
np.testing.assert_allclose(
    artifact.function(**inputs), expected, atol=5e-3, rtol=5e-3
)
```

Use a Python file so source inspection can recover the function. Supported
operations include tensor arithmetic, activations, matrix multiplication,
reductions, and selected indexing/convolution patterns; support depends on the
operation and shape. Unsupported graphs fail explicitly.

The returned `SynthesizedKernel` contains:

| Field | Meaning |
| --- | --- |
| `source` | Generated Python operation-graph source defining `f_nkigym`. |
| `function` | The generated callable, used for CPU graph checks and IR construction. |
| `input_specs` | The normalized shapes and dtypes expected by the generated kernel. |
| `adapt_inputs`, `adapt_output` | Convert reference inputs and outputs to the kernel's argument and output layout. |

Always use the artifact's normalized specifications and adapters. The physical
layout may differ from the original PyTorch layout. NumPy arrays are used for
validation and transport; the synthesis frontend is PyTorch.

`ir.dump(directory)` writes `envelope.md` and a Black-formatted `kernel.py`.
The latter is executable NKI source. To obtain the source without writing
files, call `nkigym.codegen.render(ir)`. The operation-graph check above does
not execute the rendered NKI kernel.

## Apply a transformation

Append the following to the example:

```python
from nkigym.transforms import Split

transform = Split()
options = transform.analyze(ir)
if not options:
    raise RuntimeError("No legal Split option for this kernel")

option = options[0]
candidate = transform.apply(ir, option)
candidate.dump(".cache/tanh_add/split")
print(option)
```

This chooses one legal split for demonstration. It does not imply a latency
improvement. `apply` returns a new IR and rechecks the option's legality.
Re-run `analyze` after each change; node IDs and available choices can change.
`nkigym.transforms.public_transforms()` returns the complete ordered action set.

To validate this small rendered kernel locally, append:

```python
from runpy import run_path

from nkigym.profile import simulate_fp32

namespace = run_path(".cache/tanh_add/split/kernel.py")
nki_kernel = namespace[f"nki_{candidate.func_name}"]
actual = simulate_fp32(nki_kernel)(**inputs)
np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)
```

`simulate_fp32` runs the NKI simulator with floating-point storage promoted to
FP32. It helps detect scheduling errors on small examples. Final dtype behavior
and latency require hardware validation.

## Use a frozen workload

`benchmark.NAKB_WORKLOADS` maps workload families to configurations. A
configuration ID combines the family name and zero-based index, such as
`cumsum_0`. Each configuration supplies a PyTorch reference, input contract,
seeded generator, accuracy specification, and baseline latency.

The following can be run from a Python session at the repository root:

```python
from benchmark import NAKB_WORKLOADS
from nkigym.ir import build_initial_ir
from nkigym.synthesis import synthesize_torch_to_nkigym

workload = NAKB_WORKLOADS["cumsum"][0]
artifact = synthesize_torch_to_nkigym(workload["torch_ref"], workload["input_specs"])
reference_inputs = workload["input_generator"](workload["input_specs"], seed=0)
inputs = artifact.adapt_inputs(
    {name: value.copy() for name, value in reference_inputs.items()}
)
expected = artifact.adapt_output(workload["torch_ref"](**reference_inputs))
ir = build_initial_ir(artifact.function, artifact.input_specs)
```

For these workloads, use the frozen `AccuracySpec` through
`benchmark.accuracy_validation` and `benchmark.validate_nakb_outputs`.
The example's `assert_allclose` is not a replacement for the benchmark's
output selection, grouping, or top-k rules. See
[ladder validation](validation.md#validate-recorded-ladders).

## Worker setup

Remote simulation and profiling accept caller-supplied SSH destinations.
Arrange access and host provisioning through your environment's normal
process. No hostnames, credentials, SSH configuration, or login procedures are
provided by this repository.

Workers need Python and compatible package wheels. CPU simulation uses host
CPUs; hardware profiling requires a Trn2 host with the Neuron driver, runtime,
and tools, including `neuron-explorer`. Consult the public
[NKI environment setup guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/setup-env.html)
for platform prerequisites.

Replace the example destinations with workers you already have configured:

```bash
export NKIGYM_CPU_HOST="cpu-worker.example.org"
export NKIGYM_TRN2_HOST="trn2-worker.example.org"
./install.sh --host "$NKIGYM_CPU_HOST" "$NKIGYM_TRN2_HOST"
source ~/venvs/kernel-env/bin/activate
```

These variables are conveniences for the examples, not settings read
automatically by NKIGym. A Trn2 host can also serve as a CPU worker; in that case
install it once and pass its destination to both host options.

The installer requires Bash, `python3`, `ssh`, and `rsync`. It creates
`~/venvs/kernel-env` locally and on each supplied host, installs the local
checkout in editable mode with development tools, and installs the worker
package remotely. It does not install system drivers or configure access.
Simulation and profiling submit the current backend with each run; reinstall
workers when package dependencies change.

## Profile on Trn2

Continue from either example with its `artifact`, `ir`, and `inputs`. To profile
the transformed custom example, first set `ir = candidate`.

```python
import os
from tempfile import mkdtemp

from nkigym.codegen import render
from nkigym.ir.program_sharding import configured_program_shards
from nkigym.profile import profile_metrics

profile_dir = mkdtemp(prefix="nkigym-profile-")
print(f"Profile artifacts: {profile_dir}")
metrics = profile_metrics(
    host=os.environ["NKIGYM_TRN2_HOST"],
    kernel=render(ir),
    func_name=f"nki_{ir.func_name}",
    input_specs=artifact.input_specs,
    cache_dir=profile_dir,
    lnc=max((1, *configured_program_shards(ir).values())),
    confirmation=True,
    inputs=inputs,
)
print(metrics.latency_ms, metrics.mfu_percent)
```

The result includes latency in milliseconds, estimated model FLOP utilization
(MFU) in percent, the profiler summary, and output arrays when exact inputs are
supplied. Compare `metrics.outputs` against the adapted reference using the
appropriate accuracy contract. Profiling alone does not establish correctness.

For a frozen workload, validate the returned outputs with:

```python
from benchmark import accuracy_validation, validate_nakb_outputs

validate_nakb_outputs(
    metrics.outputs, expected, inputs, accuracy_validation(workload)[1]
)
```

Use a dedicated cache directory: `profile_metrics` replaces its contents.
It retains the submitted request, compiler and transport logs, compiled
`file.neff`, execution traces, and profiler results for diagnosis. Failures
raise exceptions while preserving available artifacts.

For batched intermediate checks, `nkigym.profile.batch_simulate_fp32` accepts
`FP32SimulationCase` objects and explicit CPU destinations. The
[ladder test](../test/test_nkigym_ladder.py) demonstrates this with the frozen
accuracy contract. Record accepted ladders in
`kernel_library/_best_nkigym.py`; follow the
[results guide](validation.md#publish-an-accepted-result) to publish measurements.
