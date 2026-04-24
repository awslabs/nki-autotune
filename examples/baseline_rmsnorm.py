"""Baseline RMSNorm + matmul via nkipy: plain numpy -> HLO -> neuronx-cc -> Trn2.

Mirrors ``examples/rmsnorm_matmul.py`` but without any NKI kernel authoring.
``nkipy`` traces the numpy function to HLO, ``neuronx-cc`` lowers the HLO
to a NEFF, and ``BaremetalExecutor`` runs it on a Neuron core. The reported
MFU is the compiler's out-of-the-box result with no autotuning.

Math: ``RMSNorm(a) @ b = (a / sqrt(mean(a^2) + eps)) @ b``

MFU is computed against the matmul's MAC count (``M*K*N``) to stay
comparable with ``examples/rmsnorm_matmul.py``; the elementwise norm
term adds negligible FLOPs relative to the matmul.

This script must run on a Trn2 host (e.g. ``gym-1``) because it imports
``nkipy``, ``spike``, and ``nki``::

    rsync examples/baseline_rmsnorm.py gym-1:/tmp/
    ssh gym-1 '/home/ubuntu/venvs/kernel-env/bin/python /tmp/baseline_rmsnorm.py'
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

_VENV_BIN = os.path.dirname(sys.executable)
if _VENV_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _VENV_BIN + os.pathsep + os.environ.get("PATH", "")

import ml_dtypes
import numpy as np
from nkipy.core.trace import NKIPyKernel
from nkipy.runtime import BaremetalExecutor, CompiledKernel
from nkipy.runtime.execute import _compile_kernel

_PE_FREQ_HZ = 2.4e9
_BF16_FLOPS_PER_CYCLE = 2 * 128 * 128
EPS = 1e-6


def rmsnorm_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute ``RMSNorm(a) @ b`` in plain numpy semantics.

    Args:
        a: Input tensor of shape (M, K).
        b: Weight tensor of shape (K, N).

    Returns:
        Output tensor of shape (M, N).
    """
    k = a.shape[1]
    sum_sq = np.sum(a * a, axis=1, keepdims=True)
    scaled = sum_sq * (1.0 / k) + EPS
    rsqrt_val = np.reciprocal(np.sqrt(scaled))
    a_normed = a * rsqrt_val
    output = a_normed @ b
    return output


def calculate_mfu_bf16(mac_count: int, time_ms: float) -> float:
    """Return MFU percentage on Trn2 NeuronCore-v3 TensorEngine for bf16."""
    flops = 2 * mac_count
    actual_pe_cycles = (time_ms / 1000) * _PE_FREQ_HZ
    theoretical_pe_cycles = flops / _BF16_FLOPS_PER_CYCLE
    return 100.0 * theoretical_pe_cycles / actual_pe_cycles


def run_variant(
    name: str, inputs: dict[str, np.ndarray], mac_count: int, cache_dir: Path, warmup: int = 10, iters: int = 100
) -> dict[str, Any]:
    """Trace + compile + benchmark the rmsnorm+matmul kernel on the local Neuron core.

    Args:
        name: Variant label used for cache subdirectory naming.
        inputs: Kwargs passed to ``rmsnorm_matmul``. Values are numpy arrays.
        mac_count: Theoretical MAC count for MFU computation.
        cache_dir: Parent directory for compilation artifacts.
    """
    variant_dir = cache_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)

    traced = NKIPyKernel.trace(rmsnorm_matmul)
    args = list(inputs.values())
    neff_path, _kname, _ir, _boundargs, _orig = _compile_kernel(traced, *args, artifacts_dir=str(variant_dir))
    compiled = CompiledKernel(traced, neff_path)

    with BaremetalExecutor() as spike:
        stats = spike.benchmark(compiled, *args, warmup_iterations=warmup, benchmark_iterations=iters, mode="device")
    min_ms = stats.min_ms
    mean_ms = stats.mean_ms
    mfu = calculate_mfu_bf16(mac_count, min_ms)
    record = {
        "variant": name,
        "min_ms": min_ms,
        "mean_ms": mean_ms,
        "mac_count": mac_count,
        "mfu": mfu,
        "neff_path": neff_path,
    }
    print(f"{name:<16} min={min_ms:.3f}ms  mean={mean_ms:.3f}ms  MFU={mfu:.2f}%")
    return record


if __name__ == "__main__":
    M, K, N = 2048, 2048, 2048
    MAC_COUNT = M * K * N

    CACHE_ROOT = Path("/home/ubuntu/cache/baseline_rmsnorm")
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    bf16 = ml_dtypes.bfloat16
    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K)).astype(bf16)
    b = rng.standard_normal((K, N)).astype(bf16)

    record = run_variant("rmsnorm_matmul", {"a": a, "b": b}, MAC_COUNT, CACHE_ROOT)

    with open(CACHE_ROOT / "results.json", "w") as fh:
        json.dump(
            {"shape": {"M": M, "K": K, "N": N}, "dtype": "bfloat16", "eps": EPS, "results": [record]}, fh, indent=2
        )
    print(f"\nResults -> {CACHE_ROOT / 'results.json'}")
