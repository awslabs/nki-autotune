"""Vanilla (un-fused) attention nkigym math function.

Expresses ``O = softmax(Q @ K^T) @ V`` as a stateless DAG of ``NKIOp``
calls. Serves two purposes:

1. CPU-sim reference for the online-fused variant — ``build_ir`` parses
   this function, and the rendered kernel can be simulated against the
   plain-numpy golden ``attention_numpy``.
2. Source material for the ``flash_attention`` online-fusion recipe
   (Phase 2). The recipe pattern-matches the op chain here and
   rewrites it to a single-pass online softmax + scaled matmul drain.

Shapes follow the matmul operand convention ``stationary.T @ moving``:

* ``Q_T``: ``(d_head, seq_q)`` — stationary for MM1 so
  ``S = Q_T.T @ K_T = Q @ K^T`` has shape ``(seq_q, seq_k)``.
* ``K_T``: ``(d_head, seq_k)`` — moving for MM1.
* ``V``: ``(seq_k, d_head)`` — moving for MM2.
* Output ``O``: ``(seq_q, d_head)``.

MM2 needs ``stationary=(seq_k, seq_q)``, so an inline transpose flips
``P_norm (seq_q, seq_k)`` to ``P_norm_T`` before the matmul — the same
shape hygiene pattern as rmsnorm+matmul.
"""

import shutil
from pathlib import Path

import numpy as np

from autotune.runner.api import remote_profile
from autotune.runner.remote import remote_numpy_baseline
from autotune.runner.types import KernelJob
from nkigym.codegen import render_ir
from nkigym.kernel_ir import build_ir
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose
from nkigym.search import dump_ir, func_source_with_imports, inline_gadgets
from nkigym.search.mac import compute_mac_count


def attention_nkigym(Q_T, K_T, V):
    """``O = softmax(Q @ K^T) @ V`` as a stateless DAG.

    The softmax chain is classic two-pass: compute per-row max,
    subtract to shift for numerical stability, exp and reduce for
    the partition function, then divide to normalize. Each step is
    a single ``NKIOp`` — no hidden state, no cross-iteration reads.
    """
    S = NKIMatmul()(stationary=Q_T, moving=K_T)
    m = NKIActivationReduce(op="copy", reduce_op="max")(data=S)
    S_shifted = NKITensorScalar(op="subtract")(data=S, operand0=m)
    l = NKIActivationReduce(op="exp", reduce_op="add")(data=S_shifted)
    P = NKIActivation(op="exp")(data=S_shifted)
    l_inv = NKIActivation(op="reciprocal")(data=l)
    P_norm = NKITensorScalar(op="multiply")(data=P, operand0=l_inv)
    P_norm_T = NKITranspose()(data=P_norm)
    O = NKIMatmul()(stationary=P_norm_T, moving=V)
    return O


def attention_numpy(Q_T: np.ndarray, K_T: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Plain-numpy ``softmax(Q @ K^T) @ V`` — CPU-sim golden."""
    Q = Q_T.astype(np.float32).T
    K = K_T.astype(np.float32).T
    V32 = V.astype(np.float32)
    S = Q @ K.T
    m = np.max(S, axis=1, keepdims=True)
    P = np.exp(S - m)
    l = np.sum(P, axis=1, keepdims=True)
    P_norm = P / l
    return (P_norm @ V32).astype(Q_T.dtype)


if __name__ == "__main__":
    SEQ_Q, SEQ_K, D_HEAD = 512, 512, 128
    HOSTS = ["gym-2"]
    ATOL, RTOL = 1e-2, 1e-2

    INPUT_SPECS = {
        "Q_T": ((D_HEAD, SEQ_Q), "bfloat16"),
        "K_T": ((D_HEAD, SEQ_K), "bfloat16"),
        "V": ((SEQ_K, D_HEAD), "bfloat16"),
    }
    CACHE_ROOT = Path("/home/ubuntu/cache/attention_vanilla")
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    ir = build_ir(attention_nkigym, INPUT_SPECS)
    source = inline_gadgets(render_ir(ir))
    mac_count = compute_mac_count(attention_nkigym, INPUT_SPECS)
    nkigym_source = func_source_with_imports(attention_nkigym)

    kernels = {
        "kernel.py": KernelJob(
            source=source,
            func_name=ir.func_name,
            output_shape=(SEQ_Q, D_HEAD),
            input_specs=INPUT_SPECS,
            nkigym_source=nkigym_source,
            nkigym_func_name=attention_nkigym.__name__,
            mac_count=mac_count,
            atol=ATOL,
            rtol=RTOL,
            neuronx_cc_args=("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false"),
        )
    }
    dump_ir(CACHE_ROOT, "kernel.py", ir)

    output = remote_profile(kernels=kernels, hosts=HOSTS, cache_dir=str(CACHE_ROOT))
    for r in output.results:
        print(f"{r.kernel_name}: sim={r.cpu_sim.get('passed')}  " f"min_ms={r.min_ms}  MFU={r.mfu}")

    baseline = remote_numpy_baseline(
        func=attention_numpy, input_specs=INPUT_SPECS, mac_count=mac_count, host=HOSTS[0], kernel_name="nkipy_baseline"
    )
    print(f"{baseline.kernel_name}: min_ms={baseline.min_ms}  MFU={baseline.mfu}")
