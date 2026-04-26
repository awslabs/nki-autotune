"""Online-fused rmsnorm(lhs) @ rhs through the nkigym IR + renderer.

Hand-builds the online-fusion :class:`KernelIR` (bypassing ``build_ir``
because the recurrence's cross-iteration state on ``sbuf_m_state``
can't be expressed as a stateless DAG), renders it through the normal
``render_ir`` path, and ships to a remote Trainium host via the
standard worker transport. Compares:

* Correctness — CPU-sim via numpy vs golden ``rmsnorm(lhs) @ rhs``.
* MFU — profiled on HW, compared against the nkipy numpy-baseline and
  the hand-written online-fused kernel in ``kernel_library/``.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/rmsnorm_matmul_online_fused.py
"""

import inspect
import shutil
from pathlib import Path

import numpy as np

from autotune.runner.api import remote_profile
from autotune.runner.remote import remote_numpy_baseline
from autotune.runner.types import KernelJob
from nkigym.codegen import render_ir
from nkigym.kernel_ir.online_fused import build_rmsnorm_matmul_online_ir
from nkigym.search import dump_ir, inline_gadgets


def rmsnorm_matmul_numpy(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Plain-numpy ``rmsnorm(lhs) @ rhs`` — CPU-sim golden + nkipy baseline."""
    m = np.mean(np.square(lhs.astype(np.float32)), axis=1, keepdims=True)
    rms_inv = 1.0 / np.sqrt(m + 1e-6)
    normed = (lhs.astype(np.float32) * rms_inv).astype(lhs.dtype)
    return (normed @ rhs).astype(lhs.dtype)


def _nkigym_source_shim() -> tuple[str, str]:
    """Build a ``(source, func_name)`` shim for the worker's CPU-sim golden.

    The hand-built online-fused IR doesn't go through ``build_ir`` — it
    has no corresponding nkigym math function — so the worker's golden
    path reuses ``rmsnorm_matmul_numpy`` directly.
    """
    src = "import numpy as np\n\n" + inspect.getsource(rmsnorm_matmul_numpy)
    return src, rmsnorm_matmul_numpy.__name__


if __name__ == "__main__":
    M, K, N = 2048, 2048, 2048
    HOSTS = ["gym-2"]
    ATOL, RTOL = 1e-2, 1e-2

    INPUT_SPECS = {"lhs": ((M, K), "bfloat16"), "rhs": ((K, N), "bfloat16")}
    CACHE_ROOT = Path("/home/ubuntu/cache/rmsnorm_matmul_online_fused")
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    ir = build_rmsnorm_matmul_online_ir(INPUT_SPECS)
    source = inline_gadgets(render_ir(ir))

    nkigym_source, nkigym_func_name = _nkigym_source_shim()
    kernel_name = "kernel.py"
    kernels = {
        kernel_name: KernelJob(
            source=source,
            func_name=ir.func_name,
            output_shape=(M, N),
            input_specs=INPUT_SPECS,
            nkigym_source=nkigym_source,
            nkigym_func_name=nkigym_func_name,
            mac_count=M * K * N,
            atol=ATOL,
            rtol=RTOL,
            neuronx_cc_args=("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false"),
        )
    }
    dump_ir(CACHE_ROOT, kernel_name, ir)

    output = remote_profile(kernels=kernels, hosts=HOSTS, cache_dir=str(CACHE_ROOT))
    for r in output.results:
        sim = r.cpu_sim.get("passed")
        print(
            f"{r.kernel_name}: sim={sim}  min_ms={r.min_ms}  MFU={r.mfu}  "
            f"mbu={r.mbu_estimated_percent}  roof={r.roofline_efficiency}"
        )

    baseline = remote_numpy_baseline(
        func=rmsnorm_matmul_numpy,
        input_specs=INPUT_SPECS,
        mac_count=M * K * N,
        host=HOSTS[0],
        kernel_name="nkipy_baseline",
    )
    print(f"{baseline.kernel_name}: min_ms={baseline.min_ms}  MFU={baseline.mfu}")
