"""Vanilla rmsnorm(lhs) @ rhs — KernelIR + renderer end-to-end with sampling.

Defines the nkigym math function for rmsnorm+matmul, builds the
baseline IR, samples N random knob configurations, renders each one,
and profiles them in parallel across hosts. A plain-numpy equivalent
ships through nkipy + neuronx-cc as the zero-NKI reference baseline.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/rmsnorm_matmul.py
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
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose
from nkigym.search import dump_ir, func_source_with_imports, inline_gadgets
from nkigym.search.mac import compute_mac_count

EPS = 1e-6


def rmsnorm_matmul_nkigym(lhs, rhs):
    """``rmsnorm(lhs) @ rhs`` math function.

    rmsnorm(lhs) = lhs * rsqrt(mean(lhs², axis=K) + eps)
    output      = rmsnorm(lhs) @ rhs

    NKIMatmul.stationary expects (K, M), so NKITranspose converts
    lhs_rms(M, K) → lhs_T(K, M) before the matmul.
    """
    rms_inv = NKIActivationReduce(op="square", reduce_op="add", post_op="rsqrt", scale=1 / 2048, bias=EPS)(data=lhs)
    lhs_rms = NKITensorScalar(op="multiply")(data=lhs, operand0=rms_inv)
    lhs_T = NKITranspose()(data=lhs_rms)
    output = NKIMatmul()(stationary=lhs_T, moving=rhs)
    return output


def rmsnorm_matmul_numpy(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Plain-numpy golden for the nkipy baremetal_jit baseline."""
    m = np.mean(np.square(lhs.astype(np.float32)), axis=1, keepdims=True)
    rms_inv = 1.0 / np.sqrt(m + 1e-6)
    normed = (lhs.astype(np.float32) * rms_inv).astype(lhs.dtype)
    return (normed @ rhs).astype(lhs.dtype)


if __name__ == "__main__":
    M, K, N = 2048, 2048, 2048
    HOSTS = ["gym-2"]
    ATOL, RTOL = 1e-2, 1e-2

    INPUT_SPECS = {"lhs": ((M, K), "bfloat16"), "rhs": ((K, N), "bfloat16")}
    CACHE_ROOT = Path("/home/ubuntu/cache/rmsnorm_matmul_sampling")
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    base_ir = build_ir(rmsnorm_matmul_nkigym, INPUT_SPECS)

    mac_count = compute_mac_count(rmsnorm_matmul_nkigym, INPUT_SPECS)
    nkigym_source = func_source_with_imports(rmsnorm_matmul_nkigym)
    output_shape = tuple(base_ir.logical_tensors[base_ir.return_name].shape)

    num_samples = 20
    seen: set[tuple] = set()
    kernels: dict[str, KernelJob] = {}
    render_failures: list[tuple[int, str]] = []
    for i in range(num_samples):
        try:
            candidate = sample(base_ir)
        except Exception as e:
            render_failures.append((i, f"sample: {type(e).__name__}: {e}"))
            continue
        sig = knob_signature(candidate)
        if sig in seen:
            continue
        seen.add(sig)
        try:
            source = inline_gadgets(render_ir(candidate))
        except Exception as e:
            render_failures.append((i, f"render: {type(e).__name__}: {e}"))
            continue
        kernel_name = f"sample_{i}.py"
        kernels[kernel_name] = KernelJob(
            source=source,
            func_name=candidate.func_name,
            output_shape=output_shape,
            input_specs=INPUT_SPECS,
            nkigym_source=nkigym_source,
            nkigym_func_name=rmsnorm_matmul_nkigym.__name__,
            mac_count=mac_count,
            atol=ATOL,
            rtol=RTOL,
            neuronx_cc_args=("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false"),
        )
        dump_ir(CACHE_ROOT, kernel_name, candidate)

    print(f"sampled {len(kernels)} unique IRs, {len(render_failures)} render failures")
    for i, err in render_failures[:10]:
        print(f"  sample {i}: {err}")

    output = remote_profile(kernels=kernels, hosts=HOSTS, cache_dir=str(CACHE_ROOT))

    baseline = remote_numpy_baseline(
        func=rmsnorm_matmul_numpy,
        input_specs=INPUT_SPECS,
        mac_count=mac_count,
        host=HOSTS[0],
        kernel_name="nkipy_baseline",
    )
    print(f"{baseline.kernel_name}: min_ms={baseline.min_ms}  MFU={baseline.mfu}")
