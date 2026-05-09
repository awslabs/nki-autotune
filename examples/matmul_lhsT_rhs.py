"""Render the eager ``lhs_T @ rhs`` kernel and CPU-sim it against numpy.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/matmul_lhsT_rhs.py
"""

import shutil
from pathlib import Path

import nki
import numpy as np

from nkigym.codegen import render_eager
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy

K, M, N = 2048, 2048, 2048


@nkigym_kernel
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    """``lhs_T.T @ rhs`` with first-class buffer declarations."""
    lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(M, N), dtype="bfloat16")()

    NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


if __name__ == "__main__":
    INPUT_SPECS = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}
    CACHE_ROOT = Path("/home/ubuntu/cache/matmul_lhsT_rhs_eager")
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    source = render_eager(matmul_lhsT_rhs_nkigym, INPUT_SPECS)
    (CACHE_ROOT / "kernel.py").write_text(source)

    sim_source = source.replace("nl.bfloat16", "nl.float32")
    sim_ns: dict = {}
    exec(sim_source, sim_ns)
    kernel_fn = sim_ns["matmul_lhsT_rhs_nkigym"]
    rng = np.random.default_rng(0)
    lhs_T = rng.standard_normal((K, M)).astype(np.float32)
    rhs = rng.standard_normal((K, N)).astype(np.float32)
    actual = nki.simulate(kernel_fn)(lhs_T=lhs_T, rhs=rhs)
    if isinstance(actual, tuple):
        actual = actual[0]
    expected = lhs_T.T @ rhs
    print(f"[matmul_lhsT_rhs] max_abs={float(np.abs(actual - expected).max()):.3e}")
    assert np.allclose(actual, expected, atol=5e-3, rtol=5e-3)
    print(f"[matmul_lhsT_rhs] kernel written to {CACHE_ROOT / 'kernel.py'}")
