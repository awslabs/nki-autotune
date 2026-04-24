"""RMSNorm + matmul: build KernelIR from nkigym function, tune knobs, render, profile.

Demos the agentic backend flow on a multi-op workload:

1. Define the nkigym math function (RMSNorm + matmul).
2. ``build_ir`` produces a baseline ``KernelIR`` — one ``NKILoad`` per
   input, one ``Op`` per nkigym call, one ``NKIStore`` at the tail.
3. Overlay champion knobs (73% MFU at 2048³ bf16, strict IR-driven —
   no in-place reuse, no online fusion). These are the tunable knobs
   the agent would propose after exploring against the profiler.
4. ``remote_run`` renders + profiles on a Trainium host.

Math: ``output = RMSNorm(a) @ b = (a / sqrt(mean(a^2) + eps)) @ b``.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/rmsnorm_matmul.py
"""

import shutil
from pathlib import Path

from nkigym.kernel_ir import BufferScope, KernelIR, NumBuffers, build_ir
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose
from nkigym.search import remote_run

EPS = 1e-6


def rmsnorm_matmul_nkigym(a, b):
    """nkigym math function — the source of truth for the IR."""
    k = a.shape[1]
    sq, sum_sq = NKIActivationReduce()(data=a, op="square", reduce_op="add")
    scaled = NKITensorScalar()(data=sum_sq, op0="multiply", operand0=1.0 / k, op1="add", operand1=EPS)
    rsqrt_val = NKIActivation()(data=scaled, op="rsqrt")
    a_normed = NKITensorScalar()(data=a, op0="multiply", operand0=rsqrt_val)
    a_t = NKITranspose()(data=a_normed)
    result = NKIMatmul()(stationary=a_t, moving=b)
    return result


def build_tuned_rmsnorm_matmul_ir(input_specs: dict[str, tuple[tuple[int, ...], str]]) -> KernelIR:
    """Baseline IR from ``build_ir`` + champion knobs (≈73% MFU at 2048³).

    ``build_ir`` picks dim ids in discovery order:
    ``d0`` = a's partition (M), ``d1`` = a's free / K (ACCUMULATION),
    ``d2`` = b's free (N). Champion loop order is ``[M, N, K]`` →
    ``[d0, d2, d1]``.
    """
    ir = build_ir(rmsnorm_matmul_nkigym, input_specs)
    ir.dim_order = ["d0", "d2", "d1"]
    ir.ltiles_per_block = {"d0": 4, "d1": 8, "d2": 1}
    ir.buffer_scopes = {
        "sbuf_a": BufferScope.MIDDLE,
        "sbuf_b": BufferScope.INNER,
        "sbuf_a_normed": BufferScope.MIDDLE,
        "sbuf_a_t": BufferScope.MIDDLE,
    }
    ir.num_buffers = {
        "sbuf_b": NumBuffers(num_p_buffers=4, num_f_buffers=2),
        "sbuf_result": NumBuffers(num_p_buffers=None, num_f_buffers=4),
    }
    ir.emission_depth = {"sbuf_b": 1, "sbuf_result": 0}
    for op in ir.ops:
        if op.kind == "NKITranspose":
            op.attrs["mode"] = "dma_transpose"
    return ir


if __name__ == "__main__":
    M, K, N = 2048, 2048, 2048
    INPUT_SPECS = {"a": ((M, K), "bfloat16"), "b": ((K, N), "bfloat16")}
    HOSTS = ["gym-1"]
    ATOL, RTOL = 1e-2, 1e-2

    CACHE_ROOT = Path("/home/ubuntu/cache/rmsnorm_matmul")
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    ir = build_tuned_rmsnorm_matmul_ir(INPUT_SPECS)
    output = remote_run(
        ir=ir,
        func=rmsnorm_matmul_nkigym,
        input_specs=INPUT_SPECS,
        hosts=HOSTS,
        cache_dir=str(CACHE_ROOT),
        atol=ATOL,
        rtol=RTOL,
        kernel_name="rmsnorm_matmul.py",
    )
    for r in output.results:
        print(f"{r.kernel_name}: sim={r.cpu_sim.get('passed')}  min_ms={r.min_ms}  MFU={r.mfu}")
