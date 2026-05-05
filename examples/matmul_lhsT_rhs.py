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
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore


@nkigym_kernel
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    """``lhs_T.T @ rhs`` as an nkigym op DAG."""
    lhs_T_sbuf = NKILoad()(data=lhs_T)
    rhs_sbuf = NKILoad()(data=rhs)
    prod = NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out


if __name__ == "__main__":
    K, M, N = 2048, 2048, 2048
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
