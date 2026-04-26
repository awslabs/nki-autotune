"""Online-fused attention through the nkigym IR + flash_attention recipe.

Builds the vanilla ``attention`` IR via :func:`build_ir` parsing the
``attention_nkigym`` math function, then applies :class:`OnlineFusion`
— which matches the ``flash_attention`` recipe and rewrites the 9-op
chain into ``NKIOnlineFlashAttention`` + finalize. Renders the result
and ships to a remote Trainium host for CPU-sim + HW profiling.

Correctness is the Phase-3 target. Performance tuning (scope,
rotation, emission depth on the new scratch buffers) is a follow-on —
the canonical defaults here are intentionally naive.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/attention_online_fused.py
"""

import inspect
import shutil
from pathlib import Path

import numpy as np

from autotune.runner.api import remote_profile
from autotune.runner.remote import remote_numpy_baseline
from autotune.runner.types import KernelJob
from nkigym.codegen import render_ir
from nkigym.kernel_ir import build_ir
from nkigym.kernel_ir.rewrites import OnlineFusion
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose
from nkigym.search import dump_ir, inline_gadgets
from nkigym.search.mac import compute_mac_count


def attention_nkigym(Q_T, K_T, V):
    """Vanilla attention: ``O = softmax(Q @ K^T) @ V`` as a stateless DAG.

    Parsed by :func:`build_ir`; the ``flash_attention`` recipe matches
    the 9-op chain and rewrites it to the online-fused equivalent.
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
    return ((P / l) @ V32).astype(Q_T.dtype)


def _nkigym_source_shim() -> tuple[str, str]:
    """Build a ``(source, func_name)`` shim for the worker's CPU-sim golden.

    ``attention_numpy`` is the plain-numpy reference — the worker runs
    it in a fresh namespace using only ``numpy`` to compare against
    the simulated kernel output.
    """
    src = "import numpy as np\n\n" + inspect.getsource(attention_numpy)
    return src, attention_numpy.__name__


if __name__ == "__main__":
    SEQ_Q, SEQ_K, D_HEAD = 128, 128, 128
    HOSTS = ["gym-2"]
    ATOL, RTOL = 1e-2, 1e-2

    INPUT_SPECS = {
        "Q_T": ((D_HEAD, SEQ_Q), "bfloat16"),
        "K_T": ((D_HEAD, SEQ_K), "bfloat16"),
        "V": ((SEQ_K, D_HEAD), "bfloat16"),
    }
    CACHE_ROOT = Path("/home/ubuntu/cache/attention_online_fused")
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    vanilla = build_ir(attention_nkigym, INPUT_SPECS)
    rewrite = OnlineFusion()
    matches = rewrite.analyze(vanilla)
    if not matches:
        raise RuntimeError("OnlineFusion found no flash_attention match in the vanilla attention IR")
    ir = rewrite.apply(vanilla, matches)
    source = inline_gadgets(render_ir(ir))
    """``compute_mac_count`` expects the nkigym math function and counts
    only ``NKIMatmul`` MACs — that's exactly the attention MM1 + MM2
    cost, which is what we want for MFU denominators."""
    mac_count = compute_mac_count(attention_nkigym, INPUT_SPECS)

    nkigym_source, nkigym_func_name = _nkigym_source_shim()

    kernels = {
        "kernel.py": KernelJob(
            source=source,
            func_name=ir.func_name,
            output_shape=(SEQ_Q, D_HEAD),
            input_specs=INPUT_SPECS,
            nkigym_source=nkigym_source,
            nkigym_func_name=nkigym_func_name,
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
