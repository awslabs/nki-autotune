"""Render the eager ``lhs @ rhs`` kernel (with inline Transpose) and CPU-sim.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/matmul_lhs_rhs.py
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
from nkigym.ops.transpose import NKITranspose


@nkigym_kernel
def matmul_lhs_rhs_nkigym(lhs, rhs):
    """``lhs @ rhs`` via a Transpose staging pass."""
    lhs_sbuf = NKILoad()(data=lhs)
    rhs_sbuf = NKILoad()(data=rhs)
    lhs_T = NKITranspose()(data=lhs_sbuf)
    prod = NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out


if __name__ == "__main__":
    M, K, N = 2048, 2048, 2048
    INPUT_SPECS = {"lhs": ((M, K), "bfloat16"), "rhs": ((K, N), "bfloat16")}
    CACHE_ROOT = Path("/home/ubuntu/cache/matmul_lhs_rhs_eager")
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    source = render_eager(matmul_lhs_rhs_nkigym, INPUT_SPECS)
    (CACHE_ROOT / "kernel.py").write_text(source)

    sim_source = source.replace("nl.bfloat16", "nl.float32")
    sim_ns: dict = {}
    exec(sim_source, sim_ns)
    kernel_fn = sim_ns["matmul_lhs_rhs_nkigym"]
    rng = np.random.default_rng(1)
    lhs = rng.standard_normal((M, K)).astype(np.float32)
    rhs = rng.standard_normal((K, N)).astype(np.float32)
    actual = nki.simulate(kernel_fn)(lhs=lhs, rhs=rhs)
    if isinstance(actual, tuple):
        actual = actual[0]
    expected = lhs @ rhs
    print(f"[matmul_lhs_rhs] max_abs={float(np.abs(actual - expected).max()):.3e}")
    assert np.allclose(actual, expected, atol=5e-3, rtol=5e-3)
    print(f"[matmul_lhs_rhs] kernel written to {CACHE_ROOT / 'kernel.py'}")
