"""Double matmul ``(Q @ K^T) @ V`` — attention-style, sampling IR knobs.

For every IR-rewrite variant (see ``enumerate_rewrite_combinations``)
samples ``num_samples`` random knob configurations from the variant
IR, renders each one, and profiles them in parallel across hosts.

A plain-numpy equivalent ships through nkipy + neuronx-cc
(``baremetal_jit``'s path) as the zero-NKI reference baseline.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/double_matmul.py
"""

import shutil
from pathlib import Path

import numpy as np

from autotune.runner.api import remote_profile
from autotune.runner.remote import remote_numpy_baseline
from autotune.runner.types import KernelJob
from nkigym.codegen import render_ir
from nkigym.kernel_ir import build_ir
from nkigym.kernel_ir.rewrites import LoadTranspose, enumerate_rewrite_combinations
from nkigym.kernel_ir.sample import knob_signature, sample
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.transpose import NKITranspose
from nkigym.search import dump_ir, func_source_with_imports, inline_gadgets
from nkigym.search.mac import compute_mac_count


def double_matmul_nkigym(q, k, v):
    """nkigym math function for attention-style double matmul ``(Q @ K^T) @ V``."""
    q_T = NKITranspose()(data=q)
    k_T = NKITranspose()(data=k)
    scores = NKIMatmul()(stationary=q_T, moving=k_T)
    scores_T = NKITranspose()(data=scores)
    output = NKIMatmul()(stationary=scores_T, moving=v)
    return output


def double_matmul_numpy(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Plain-numpy equivalent for the nkipy baremetal_jit baseline."""
    return (q @ k.T) @ v


if __name__ == "__main__":
    SEQLEN_Q, SEQLEN_KV, D = 2048, 2048, 128
    HOSTS = ["gym-1"]
    ATOL, RTOL = 1e-2, 1e-2

    INPUT_SPECS = {
        "q": ((SEQLEN_Q, D), "bfloat16"),
        "k": ((SEQLEN_KV, D), "bfloat16"),
        "v": ((SEQLEN_KV, D), "bfloat16"),
    }
    CACHE_ROOT = Path("/home/ubuntu/cache/double_matmul_sampling")

    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    base_ir = build_ir(double_matmul_nkigym, INPUT_SPECS)
    variants = dict(enumerate_rewrite_combinations(base_ir, [LoadTranspose()]))

    mac_count = compute_mac_count(double_matmul_nkigym, INPUT_SPECS)
    nkigym_source = func_source_with_imports(double_matmul_nkigym)
    output_shape = tuple(base_ir.logical_tensors[base_ir.return_name].shape)

    num_samples = 10
    seen: set[tuple] = set()
    kernels: dict[str, KernelJob] = {}
    for variant, ir in variants.items():
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
                nkigym_func_name=double_matmul_nkigym.__name__,
                mac_count=mac_count,
                atol=ATOL,
                rtol=RTOL,
                neuronx_cc_args=("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false"),
            )
            dump_ir(CACHE_ROOT, kernel_name, candidate)

    output = remote_profile(kernels=kernels, hosts=HOSTS, cache_dir=str(CACHE_ROOT))

    baseline = remote_numpy_baseline(
        func=double_matmul_numpy,
        input_specs=INPUT_SPECS,
        mac_count=mac_count,
        host=HOSTS[0],
        kernel_name="nkipy_baseline",
    )
    print(f"{baseline.kernel_name}: min_ms={baseline.min_ms}  MFU={baseline.mfu}")
