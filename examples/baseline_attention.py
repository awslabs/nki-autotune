"""Baseline causal attention via nkipy: plain numpy -> HLO -> neuronx-cc -> Trn2.

Mirrors ``examples/baseline_matmul.py`` but for causal self-attention. The
tensors match the convention used by ``attention_cte_golden``:

    Q: (seq_len, d_k)
    K: (seq_len, d_k)
    V: (seq_len, d_v)

    output = softmax(causal(scale * Q @ K^T)) @ V

No NKI kernel authoring: ``nkipy`` traces the numpy function to HLO,
``neuronx-cc`` lowers it, and ``BaremetalExecutor`` runs it on a Neuron
core. The reported MFU is the compiler's out-of-the-box result.

This script must run on a Trn2 host (e.g. ``gym-2``)::

    rsync examples/baseline_attention.py gym-2:/tmp/
    ssh gym-2 '/home/ubuntu/venvs/kernel-env/bin/python /tmp/baseline_attention.py'
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


def attention_causal(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Causal self-attention on 2D tensors.

    Args:
        Q: Query tensor of shape (seq_len, d_k).
        K: Key tensor of shape (seq_len, d_k).
        V: Value tensor of shape (seq_len, d_v).

    Returns:
        Output tensor of shape (seq_len, d_v).
    """
    scale = 1.0 / np.sqrt(Q.shape[-1])
    s = (Q @ K.T) * scale
    causal = np.triu(np.full_like(s, -np.inf), k=1)
    s = s + causal
    s = s - s.max(axis=-1, keepdims=True)
    exp_s = np.exp(s)
    p = exp_s / exp_s.sum(axis=-1, keepdims=True)
    output = p @ V
    return output


def calculate_mfu_bf16(mac_count: int, time_ms: float) -> float:
    """Return MFU percentage on Trn2 NeuronCore-v3 TensorEngine for bf16."""
    flops = 2 * mac_count
    actual_pe_cycles = (time_ms / 1000) * _PE_FREQ_HZ
    theoretical_pe_cycles = flops / _BF16_FLOPS_PER_CYCLE
    return 100.0 * theoretical_pe_cycles / actual_pe_cycles


def run_attention(
    inputs: dict[str, np.ndarray], mac_count: int, cache_dir: Path, warmup: int = 10, iters: int = 100
) -> dict[str, Any]:
    """Trace + compile + benchmark the causal attention kernel on the local Neuron core.

    Args:
        inputs: Kwargs passed to ``attention_causal``. Values are numpy arrays.
        mac_count: Theoretical MAC count for MFU computation.
        cache_dir: Directory for compilation artifacts.
        warmup: Warmup iterations.
        iters: Benchmark iterations.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    traced = NKIPyKernel.trace(attention_causal)
    args = list(inputs.values())
    neff_path, kname, _ir, _boundargs, _orig = _compile_kernel(traced, *args, artifacts_dir=str(cache_dir))
    compiled = CompiledKernel(traced, neff_path)

    with BaremetalExecutor() as spike:
        stats = spike.benchmark(compiled, *args, warmup_iterations=warmup, benchmark_iterations=iters, mode="device")
    min_ms = stats.min_ms
    mean_ms = stats.mean_ms
    mfu = calculate_mfu_bf16(mac_count, min_ms)
    record = {"min_ms": min_ms, "mean_ms": mean_ms, "mac_count": mac_count, "mfu": mfu, "neff_path": neff_path}
    print(f"attention    min={min_ms:.3f}ms  mean={mean_ms:.3f}ms  MFU={mfu:.2f}%")
    return record


if __name__ == "__main__":
    seq_len, d_k, d_v = 2048, 128, 128
    mac_count = seq_len * seq_len * d_k + seq_len * seq_len * d_v

    cache_root = Path("/home/ubuntu/cache/baseline_attention")
    shutil.rmtree(cache_root, ignore_errors=True)
    cache_root.mkdir(parents=True)

    bf16 = ml_dtypes.bfloat16
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((seq_len, d_k)).astype(bf16)
    K = rng.standard_normal((seq_len, d_k)).astype(bf16)
    V = rng.standard_normal((seq_len, d_v)).astype(bf16)

    record = run_attention({"Q": Q, "K": K, "V": V}, mac_count, cache_root)

    with open(cache_root / "results.json", "w") as fh:
        json.dump(
            {"shape": {"seq_len": seq_len, "d_k": d_k, "d_v": d_v}, "dtype": "bfloat16", "result": record}, fh, indent=2
        )
    print(f"\nResults -> {cache_root / 'results.json'}")
