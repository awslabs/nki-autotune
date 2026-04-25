"""Matrix multiplication: build KernelIR from nkigym function, tune knobs, render, profile.

Demos the agentic backend flow:

1. Define the nkigym math function (chain of ``NKIOp`` calls).
2. ``build_ir`` produces a baseline ``KernelIR`` with naive defaults
   (``dim_order`` = non-ACC first then ACC, ``ltiles_per_block = 1``,
   empty ``buffer_scopes`` / ``num_buffers`` / ``emission_depth``).
3. Overlay champion knobs — this is what the agent's iteration loop
   would write back after exploring variants against the profiler.
4. ``remote_run`` calls ``render_ir`` on the tuned IR, ships to a
   Trainium host, and reports MFU.
5. ``remote_numpy_baseline`` ships a plain-numpy ``lhs_T.T @ rhs`` to
   the same host through nkipy + neuronx-cc — the zero-NKI reference
   MFU to compare the tuned kernel against.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/matmul_lhsT_rhs.py
"""

import shutil
from pathlib import Path

import numpy as np

from autotune.runner.remote import remote_numpy_baseline
from nkigym.kernel_ir import BufferScope, KernelIR, NumBuffers, build_ir
from nkigym.ops.matmul import NKIMatmul
from nkigym.search import remote_run
from nkigym.search.mac import compute_mac_count


def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    """nkigym math function — the source of truth for the IR."""
    output = NKIMatmul()(stationary=lhs_T, moving=rhs)
    return output


def matmul_lhsT_rhs_numpy(lhs_T: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Plain-numpy equivalent for the nkipy baremetal_jit baseline."""
    return lhs_T.T @ rhs


def build_tuned_matmul_ir(input_specs: dict[str, tuple[tuple[int, ...], str]]) -> KernelIR:
    """Baseline IR from ``build_ir`` + champion knobs (90.9% MFU at 2048³)."""
    ir = build_ir(matmul_lhsT_rhs_nkigym, input_specs)
    ir.dim_order = ["d2", "d0", "d1"]
    ir.ltiles_per_block = {"d0": 8, "d1": 4, "d2": 1}
    ir.buffer_scopes = {"sbuf_lhs_T": BufferScope.INNER, "sbuf_rhs": BufferScope.INNER}
    ir.num_buffers = {
        "sbuf_lhs_T": NumBuffers(num_p_buffers=2, num_f_buffers=4),
        "sbuf_rhs": NumBuffers(num_p_buffers=2, num_f_buffers=None),
        "sbuf_output": NumBuffers(num_p_buffers=None, num_f_buffers=4),
    }
    ir.emission_depth = {"sbuf_lhs_T": 1, "sbuf_rhs": 1, "sbuf_output": 0}
    return ir


if __name__ == "__main__":
    K, M, N = 2048, 2048, 2048
    HOSTS = ["gym-1"]
    ATOL, RTOL = 1e-2, 1e-2

    INPUT_SPECS = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}
    CACHE_ROOT = Path("/home/ubuntu/cache/matmul_lhsT_rhs")

    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    ir = build_tuned_matmul_ir(INPUT_SPECS)
    output = remote_run(
        ir=ir,
        func=matmul_lhsT_rhs_nkigym,
        input_specs=INPUT_SPECS,
        hosts=HOSTS,
        cache_dir=str(CACHE_ROOT),
        atol=ATOL,
        rtol=RTOL,
        kernel_name="matmul_lhsT_rhs.py",
    )
    for r in output.results:
        print(f"{r.kernel_name}: sim={r.cpu_sim.get('passed')}  min_ms={r.min_ms}  MFU={r.mfu}")

    mac_count = compute_mac_count(matmul_lhsT_rhs_nkigym, INPUT_SPECS)
    baseline = remote_numpy_baseline(
        func=matmul_lhsT_rhs_numpy,
        input_specs=INPUT_SPECS,
        mac_count=mac_count,
        host=HOSTS[0],
        kernel_name="nkipy_baseline",
    )
    print(f"{baseline.kernel_name}: min_ms={baseline.min_ms}  MFU={baseline.mfu}  ({baseline.hardware_output})")
