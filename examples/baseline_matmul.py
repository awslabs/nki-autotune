"""Baseline matrix multiplication via nkipy: plain numpy -> HLO -> neuronx-cc -> Trn2.

Mirrors the four input-layout variants of ``examples/matmul.py`` but without
any NKI kernel authoring. ``nkipy`` traces the numpy function to HLO,
``neuronx-cc`` lowers the HLO to a NEFF, and ``BaremetalExecutor`` runs
it on a Neuron core. The reported MFU is the compiler's out-of-the-box
result with no autotuning.

This script must run on a Trn2 host (e.g. ``gym-1``) because it imports
``nkipy``, ``spike``, and ``nki``. On the coordinator::

    rsync examples/baseline_matmul.py gym-1:/tmp/
    ssh gym-1 '/home/ubuntu/venvs/kernel-env/bin/python /tmp/baseline_matmul.py'
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

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


def matmul_lhsT_rhs(lhs_T: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Compute ``lhs_T.T @ rhs``."""
    output = lhs_T.T @ rhs
    return output


def matmul_lhs_rhs(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Compute ``lhs @ rhs``."""
    output = lhs @ rhs
    return output


def matmul_lhs_rhsT(lhs: np.ndarray, rhs_T: np.ndarray) -> np.ndarray:
    """Compute ``lhs @ rhs_T.T``."""
    output = lhs @ rhs_T.T
    return output


def matmul_lhsT_rhsT(lhs_T: np.ndarray, rhs_T: np.ndarray) -> np.ndarray:
    """Compute ``lhs_T.T @ rhs_T.T``."""
    output = lhs_T.T @ rhs_T.T
    return output


def calculate_mfu_bf16(mac_count: int, time_ms: float) -> float:
    """Return MFU percentage on Trn2 NeuronCore-v3 TensorEngine for bf16."""
    flops = 2 * mac_count
    actual_pe_cycles = (time_ms / 1000) * _PE_FREQ_HZ
    theoretical_pe_cycles = flops / _BF16_FLOPS_PER_CYCLE
    return 100.0 * theoretical_pe_cycles / actual_pe_cycles


def run_variant(
    name: str,
    func: Callable[..., np.ndarray],
    inputs: dict[str, np.ndarray],
    mac_count: int,
    cache_dir: Path,
    warmup: int = 10,
    iters: int = 100,
) -> dict[str, Any]:
    """Trace + compile + benchmark one variant on the local Neuron core.

    Args:
        name: Variant label used for cache subdirectory naming.
        func: Plain numpy function; arithmetic ops are traced to HLO.
        inputs: Kwargs passed to ``func``. Values are numpy arrays.
        mac_count: Theoretical MAC count for MFU computation.
        cache_dir: Parent directory for compilation artifacts.
    """
    variant_dir = cache_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)

    traced = NKIPyKernel.trace(func)
    args = list(inputs.values())
    neff_path, kname, _ir, _boundargs, _orig = _compile_kernel(traced, *args, artifacts_dir=str(variant_dir))
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
    print(f"{name:<12} min={min_ms:.3f}ms  mean={mean_ms:.3f}ms  MFU={mfu:.2f}%")
    return record


if __name__ == "__main__":
    K, M, N = 2048, 2048, 2048
    MAC_COUNT = M * N * K

    CACHE_ROOT = Path("/home/ubuntu/cache/baseline_matmul")
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    bf16 = ml_dtypes.bfloat16
    rng = np.random.default_rng(0)
    lhs = rng.standard_normal((M, K)).astype(bf16)
    lhs_T = rng.standard_normal((K, M)).astype(bf16)
    rhs = rng.standard_normal((K, N)).astype(bf16)
    rhs_T = rng.standard_normal((N, K)).astype(bf16)

    variants = [
        ("lhsT_rhs", matmul_lhsT_rhs, {"lhs_T": lhs_T, "rhs": rhs}),
        ("lhs_rhs", matmul_lhs_rhs, {"lhs": lhs, "rhs": rhs}),
        ("lhs_rhsT", matmul_lhs_rhsT, {"lhs": lhs, "rhs_T": rhs_T}),
        ("lhsT_rhsT", matmul_lhsT_rhsT, {"lhs_T": lhs_T, "rhs_T": rhs_T}),
    ]

    results = [run_variant(n, f, i, MAC_COUNT, CACHE_ROOT) for n, f, i in variants]

    with open(CACHE_ROOT / "results.json", "w") as fh:
        json.dump({"shape": {"K": K, "M": M, "N": N}, "dtype": "bfloat16", "results": results}, fh, indent=2)
    print(f"\nResults -> {CACHE_ROOT / 'results.json'}")
