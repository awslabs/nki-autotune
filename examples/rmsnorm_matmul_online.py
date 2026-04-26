"""Run the hand-written online-fused rmsnorm+matmul kernel on a gym host.

Loads ``kernel_library/rmsnorm_matmul/kernel_hand_online.py`` verbatim
as the NKI source, ships it to a remote Trainium host via the standard
worker transport, and compares:

* Correctness — CPU-sim via numpy vs golden ``rmsnorm(lhs) @ rhs``.
* MFU — profiled on HW, compared against the nkipy numpy-baseline.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/rmsnorm_matmul_online.py
"""

import inspect
import shutil
from pathlib import Path

import numpy as np

from autotune.runner.api import remote_profile
from autotune.runner.remote import remote_numpy_baseline
from autotune.runner.types import KernelJob

_KERNEL_PATH = Path(__file__).resolve().parent.parent / "kernel_library" / "rmsnorm_matmul" / "kernel_hand_online.py"


EPS = 1e-6


def rmsnorm_matmul_numpy(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Plain-numpy ``rmsnorm(lhs) @ rhs`` — the golden reference.

    Written as a standalone function so ``remote_numpy_baseline`` can
    ship its source to the worker and push it through nkipy + neuronx-cc.
    """
    m = np.mean(np.square(lhs.astype(np.float32)), axis=1, keepdims=True)
    rms_inv = 1.0 / np.sqrt(m + 1e-6)
    normed = (lhs.astype(np.float32) * rms_inv).astype(lhs.dtype)
    return (normed @ rhs).astype(lhs.dtype)


def _nkigym_source_shim() -> tuple[str, str]:
    """Build a minimal ``nkigym_source`` + func name for the worker's golden pass.

    The worker's CPU-sim path calls ``nkigym_func_name`` in a fresh
    namespace using only ``numpy``. The hand kernel isn't nkigym-driven,
    but the worker still wants a numpy function that matches the kernel
    signature and returns the expected output shape/dtype. We reuse
    ``rmsnorm_matmul_numpy`` above as that shim.
    """
    src = "import numpy as np\n\n" + inspect.getsource(rmsnorm_matmul_numpy)
    return src, rmsnorm_matmul_numpy.__name__


if __name__ == "__main__":
    M, K, N = 2048, 2048, 2048
    HOSTS = ["gym-2"]
    ATOL, RTOL = 1e-2, 1e-2

    INPUT_SPECS = {"lhs": ((M, K), "bfloat16"), "rhs": ((K, N), "bfloat16")}
    CACHE_ROOT = Path("/home/ubuntu/cache/rmsnorm_matmul_online")
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    """MACs: one matmul of (M, K) × (K, N). rmsnorm adds a small
    multiply-add per element — negligible next to the MK·N matmul, so
    the MFU denominator uses matmul MACs only."""
    mac_count = M * K * N

    kernel_source = _KERNEL_PATH.read_text()

    nkigym_source, nkigym_func_name = _nkigym_source_shim()

    kernels = {
        "kernel_hand_online.py": KernelJob(
            source=kernel_source,
            func_name="rmsnorm_matmul_online",
            output_shape=(M, N),
            input_specs=INPUT_SPECS,
            nkigym_source=nkigym_source,
            nkigym_func_name=nkigym_func_name,
            mac_count=mac_count,
            atol=ATOL,
            rtol=RTOL,
            neuronx_cc_args=("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false"),
        )
    }

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
        mac_count=mac_count,
        host=HOSTS[0],
        kernel_name="nkipy_baseline",
    )
    print(f"{baseline.kernel_name}: min_ms={baseline.min_ms}  MFU={baseline.mfu}")
