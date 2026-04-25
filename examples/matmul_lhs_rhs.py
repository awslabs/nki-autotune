"""Matmul ``lhs @ rhs`` — with and without the ``LoadTranspose`` rewrite.

Two variants share the same math function and tuned knobs:

1. **Base** — explicit ``NKITranspose`` op + ``transpose_block`` gadget
   (``nisa.nc_transpose`` via Tensor Engine).
2. **Fused** — ``LoadTranspose`` rewrite merges the transpose into the
   ``lhs`` load, producing one ``nisa.dma_transpose`` per leaf and
   dropping the intermediate ``sbuf_lhs`` buffer.

A third entry ships the plain-numpy equivalent through nkipy +
neuronx-cc (``baremetal_jit``'s path) as the zero-NKI reference
baseline.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/matmul_lhs_rhs.py
"""

import shutil
from pathlib import Path

import numpy as np

from autotune.runner.remote import remote_numpy_baseline
from nkigym.kernel_ir import BufferScope, KernelIR, NumBuffers, build_ir
from nkigym.kernel_ir.rewrites import LoadTranspose
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.transpose import NKITranspose
from nkigym.search import remote_run
from nkigym.search.mac import compute_mac_count


def matmul_lhs_rhs_nkigym(lhs, rhs):
    """nkigym math function for ``lhs @ rhs`` (lhs not pre-transposed)."""
    lhs_T = NKITranspose()(data=lhs)
    output = NKIMatmul()(stationary=lhs_T, moving=rhs)
    return output


def matmul_lhs_rhs_numpy(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Plain-numpy equivalent for the nkipy baremetal_jit baseline."""
    return lhs @ rhs


def build_tuned_ir(input_specs: dict[str, tuple[tuple[int, ...], str]], fuse_load_transpose: bool) -> KernelIR:
    """Build the baseline IR, tune knobs, optionally apply ``LoadTranspose``."""
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
    if fuse_load_transpose:
        ir = LoadTranspose()(ir)
    return ir


if __name__ == "__main__":
    M, K, N = 2048, 2048, 2048
    HOSTS = ["gym-1"]
    ATOL, RTOL = 1e-2, 1e-2

    INPUT_SPECS = {"lhs": ((M, K), "bfloat16"), "rhs": ((K, N), "bfloat16")}
    CACHE_ROOT = Path("/home/ubuntu/cache/matmul_lhs_rhs")

    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    for fuse, kernel_name in ((False, "base.py"), (True, "load_transpose.py")):
        ir = build_tuned_ir(INPUT_SPECS, fuse_load_transpose=fuse)
        output = remote_run(
            ir=ir,
            func=matmul_lhs_rhs_nkigym,
            input_specs=INPUT_SPECS,
            hosts=HOSTS,
            cache_dir=str(CACHE_ROOT),
            atol=ATOL,
            rtol=RTOL,
            kernel_name=kernel_name,
        )
        for r in output.results:
            print(f"{r.kernel_name}: sim={r.cpu_sim.get('passed')}  min_ms={r.min_ms}  MFU={r.mfu}")

    mac_count = compute_mac_count(matmul_lhs_rhs_nkigym, INPUT_SPECS)
    baseline = remote_numpy_baseline(
        func=matmul_lhs_rhs_numpy,
        input_specs=INPUT_SPECS,
        mac_count=mac_count,
        host=HOSTS[0],
        kernel_name="nkipy_baseline",
    )
    print(f"{baseline.kernel_name}: min_ms={baseline.min_ms}  MFU={baseline.mfu}  ({baseline.hardware_output})")
