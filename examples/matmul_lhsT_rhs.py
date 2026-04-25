"""Matmul ``lhs_T @ rhs`` — sampling many IR knob configurations.

Samples ``num_samples`` random knob configurations from ``build_ir``'s
default IR, renders each one, and profiles them in parallel across
hosts.

A plain-numpy equivalent ships through nkipy + neuronx-cc
(``baremetal_jit``'s path) as the zero-NKI reference baseline.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/matmul_lhsT_rhs.py
"""

import shutil
from pathlib import Path

import numpy as np

from autotune.runner.api import remote_profile
from autotune.runner.remote import remote_numpy_baseline
from autotune.runner.types import KernelJob
from nkigym.codegen import render_ir
from nkigym.kernel_ir import build_ir
from nkigym.kernel_ir.sample import knob_signature, sample
from nkigym.ops.matmul import NKIMatmul
from nkigym.search import dump_ir, func_source_with_imports, inline_gadgets
from nkigym.search.mac import compute_mac_count


def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    """nkigym math function for ``lhs_T.T @ rhs``."""
    output = NKIMatmul()(stationary=lhs_T, moving=rhs)
    return output


def matmul_lhsT_rhs_numpy(lhs_T: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Plain-numpy equivalent for the nkipy baremetal_jit baseline."""
    return lhs_T.T @ rhs


if __name__ == "__main__":
    K, M, N = 2048, 2048, 2048
    HOSTS = ["gym-2", "gym-3"]
    ATOL, RTOL = 1e-2, 1e-2

    INPUT_SPECS = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}
    CACHE_ROOT = Path("/home/ubuntu/cache/matmul_lhsT_rhs_sampling")

    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    base_ir = build_ir(matmul_lhsT_rhs_nkigym, INPUT_SPECS)

    mac_count = compute_mac_count(matmul_lhsT_rhs_nkigym, INPUT_SPECS)
    nkigym_source = func_source_with_imports(matmul_lhsT_rhs_nkigym)
    output_shape = tuple(base_ir.logical_tensors[base_ir.return_name].shape)

    num_samples = 100
    seen: set[tuple] = set()
    kernels: dict[str, KernelJob] = {}
    for i in range(num_samples):
        candidate = sample(base_ir)
        sig = knob_signature(candidate)
        if sig in seen:
            continue
        seen.add(sig)
        source = inline_gadgets(render_ir(candidate))
        kernel_name = f"kernel_{i}.py"
        kernels[kernel_name] = KernelJob(
            source=source,
            func_name=candidate.func_name,
            output_shape=output_shape,
            input_specs=INPUT_SPECS,
            nkigym_source=nkigym_source,
            nkigym_func_name=matmul_lhsT_rhs_nkigym.__name__,
            mac_count=mac_count,
            atol=ATOL,
            rtol=RTOL,
            neuronx_cc_args=("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false"),
        )
        dump_ir(CACHE_ROOT, kernel_name, candidate)

    output = remote_profile(kernels=kernels, hosts=HOSTS, cache_dir=str(CACHE_ROOT))

    baseline = remote_numpy_baseline(
        func=matmul_lhsT_rhs_numpy,
        input_specs=INPUT_SPECS,
        mac_count=mac_count,
        host=HOSTS[0],
        kernel_name="nkipy_baseline",
    )
    print(f"{baseline.kernel_name}: min_ms={baseline.min_ms}  MFU={baseline.mfu}")
