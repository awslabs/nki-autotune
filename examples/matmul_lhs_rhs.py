"""Matrix multiplication with inline transpose: ``lhs @ rhs``.

Mirrors ``examples/matmul.py`` but starts from the non-transposed
``lhs(M, K)`` and inserts an explicit ``NKITranspose`` to satisfy the
``NKIMatmul.stationary=(K, M)`` layout contract.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/matmul_lhs_rhs.py
"""

import shutil
from pathlib import Path

from nkigym.kernel_ir import BufferScope, KernelIR, NumBuffers, build_ir
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.transpose import NKITranspose
from nkigym.search import remote_run


def matmul_lhs_rhs_nkigym(lhs, rhs):
    """nkigym math function for ``lhs @ rhs`` (lhs not pre-transposed)."""
    lhs_T = NKITranspose()(data=lhs)
    output = NKIMatmul()(stationary=lhs_T, moving=rhs)
    return output


def build_tuned_matmul_lhs_rhs_ir(input_specs: dict[str, tuple[tuple[int, ...], str]]) -> KernelIR:
    """Baseline IR from ``build_ir`` + the knobs from ``matmul_lhs_rhs.md``."""
    ir = build_ir(matmul_lhs_rhs_nkigym, input_specs)
    ir.dim_order = ["d2", "d0", "d1"]
    ir.ltiles_per_block = {"d0": 8, "d1": 4, "d2": 1}
    ir.buffer_scopes = {
        "sbuf_lhs": BufferScope.INNER,
        "sbuf_lhs_T": BufferScope.INNER,
        "sbuf_rhs": BufferScope.INNER,
        "sbuf_output": BufferScope.MIDDLE,
    }
    ir.num_buffers = {
        "sbuf_lhs": NumBuffers(num_p_buffers=2, num_f_buffers=4),
        "sbuf_lhs_T": NumBuffers(num_p_buffers=2, num_f_buffers=4),
        "sbuf_rhs": NumBuffers(num_p_buffers=2, num_f_buffers=None),
        "sbuf_output": NumBuffers(num_p_buffers=None, num_f_buffers=4),
    }
    ir.emission_depth = {"sbuf_lhs": 1, "sbuf_lhs_T": 1, "sbuf_rhs": 1, "sbuf_output": 0}
    return ir


if __name__ == "__main__":
    M, K, N = 2048, 2048, 2048
    HOSTS = ["gym-1"]
    ATOL, RTOL = 1e-2, 1e-2

    INPUT_SPECS = {"lhs": ((M, K), "bfloat16"), "rhs": ((K, N), "bfloat16")}
    CACHE_ROOT = Path("/home/ubuntu/cache/matmul_lhs_rhs")

    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    ir = build_tuned_matmul_lhs_rhs_ir(INPUT_SPECS)
    output = remote_run(
        ir=ir,
        func=matmul_lhs_rhs_nkigym,
        input_specs=INPUT_SPECS,
        hosts=HOSTS,
        cache_dir=str(CACHE_ROOT),
        atol=ATOL,
        rtol=RTOL,
        kernel_name="matmul_lhs_rhs.py",
    )
    for r in output.results:
        print(f"{r.kernel_name}: sim={r.cpu_sim.get('passed')}  min_ms={r.min_ms}  MFU={r.mfu}")
