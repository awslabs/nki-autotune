"""Matmul ``lhs @ rhs`` — with and without the ``LoadTranspose`` rewrite.

Two variants share the same math function and tuned knobs:

1. **Base** — explicit ``NKITranspose`` op + ``transpose_block`` gadget
   (``nisa.nc_transpose`` via Tensor Engine).
2. **Fused** — ``LoadTranspose`` rewrite merges the transpose into the
   ``lhs`` load, producing one ``nisa.dma_transpose`` per leaf and
   dropping the intermediate ``sbuf_lhs`` buffer.

A third entry ships the plain-numpy equivalent through nkipy +
neuronx-cc (``baremetal_jit``'s path) as the zero-NKI reference
baseline.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/matmul_lhs_rhs.py
"""

import shutil
from pathlib import Path

import numpy as np

from autotune.runner.api import remote_profile
from autotune.runner.remote import remote_numpy_baseline
from autotune.runner.types import KernelJob
from nkigym.codegen import render_ir
from nkigym.kernel_ir import build_ir
from nkigym.kernel_ir.rewrites import LoadTranspose
from nkigym.kernel_ir.sample import knob_signature, sample
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.transpose import NKITranspose
from nkigym.search import dump_ir, func_source_with_imports, inline_gadgets
from nkigym.search.mac import compute_mac_count


def matmul_lhs_rhs_nkigym(lhs, rhs):
    """nkigym math function for ``lhs @ rhs`` (lhs not pre-transposed)."""
    lhs_T = NKITranspose()(data=lhs)
    output = NKIMatmul()(stationary=lhs_T, moving=rhs)
    return output


def matmul_lhs_rhs_numpy(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Plain-numpy equivalent for the nkipy baremetal_jit baseline."""
    return lhs @ rhs


if __name__ == "__main__":
    M, K, N = 2048, 2048, 2048
    HOSTS = ["gym-2", "gym-3"]
    ATOL, RTOL = 1e-2, 1e-2

    INPUT_SPECS = {"lhs": ((M, K), "bfloat16"), "rhs": ((K, N), "bfloat16")}
    CACHE_ROOT = Path("/home/ubuntu/cache/matmul_lhs_rhs_sampling")

    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    base_ir = build_ir(matmul_lhs_rhs_nkigym, INPUT_SPECS)
    fused_ir = LoadTranspose()(base_ir)

    mac_count = compute_mac_count(matmul_lhs_rhs_nkigym, INPUT_SPECS)
    nkigym_source = func_source_with_imports(matmul_lhs_rhs_nkigym)
    output_shape = tuple(base_ir.logical_tensors[base_ir.return_name].shape)

    num_samples = 100
    seen: set[tuple] = set()
    kernels: dict[str, KernelJob] = {}
    for variant, ir in [("base", base_ir), ("fused", fused_ir)]:
        for i in range(num_samples):
            candidate = sample(ir)
            sig = knob_signature(candidate)
            if sig in seen:
                continue
            seen.add(sig)
            source = inline_gadgets(render_ir(candidate))
            kernel_name = f"{variant}_{i}.py"
            kernels[kernel_name] = KernelJob(
                source=source,
                func_name=candidate.func_name,
                output_shape=output_shape,
                input_specs=INPUT_SPECS,
                nkigym_source=nkigym_source,
                nkigym_func_name=matmul_lhs_rhs_nkigym.__name__,
                mac_count=mac_count,
                atol=ATOL,
                rtol=RTOL,
                neuronx_cc_args=("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false"),
            )
            dump_ir(CACHE_ROOT, kernel_name, candidate)

    output = remote_profile(kernels=kernels, hosts=HOSTS, cache_dir=str(CACHE_ROOT))

    baseline = remote_numpy_baseline(
        func=matmul_lhs_rhs_numpy,
        input_specs=INPUT_SPECS,
        mac_count=mac_count,
        host=HOSTS[0],
        kernel_name="nkipy_baseline",
    )
    print(f"{baseline.kernel_name}: min_ms={baseline.min_ms}  MFU={baseline.mfu}")
