"""Numpy matmul → ``f_nkigym`` → dim unification.

The ``f_nkigym`` body below is the output of
:func:`nkigym.synthesis.numpy_to_nkigym.compile_numpy_to_nkigym`
pasted verbatim — re-run the synthesiser manually whenever the op
surface or workload changes.

Usage::

    source ~/venvs/kernel-env/bin/activate
    PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python examples/matmul_lhsT_rhs.py
"""

import importlib.util
import os
import shutil
from pathlib import Path

import numpy as np

from nkigym.ir import build_initial_ir
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.synthesis.simulate_nki import simulate_fp32

K, M, N = 2048, 2048, 2048
INPUT_SPECS: dict[str, tuple[int, ...]] = {"lhs_T": (K, M), "rhs": (K, N)}


def f_numpy(lhs_T: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """``lhs_T.T @ rhs`` — plain numpy reference (synthesis source)."""
    return lhs_T.T @ rhs


@nkigym_kernel
def f_nkigym(lhs_T, rhs):
    """Cached output of ``compile_numpy_to_nkigym(f_numpy, ...)``."""
    sbuf_lhs_T = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
    sbuf_rhs = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
    psum_prod = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(M, N), dtype="bfloat16")()

    NKILoad()(src=lhs_T, dst=sbuf_lhs_T)
    NKILoad()(src=rhs, dst=sbuf_rhs)
    NKIMemset(value=0.0)(dst=psum_prod)
    NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs, dst=psum_prod)
    NKITensorCopy()(src=psum_prod, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


def _check_numerics(kernel_path: Path, seed: int = 0, atol: float = 5e-3, rtol: float = 5e-3) -> None:
    """Simulate the dumped kernel under fp32 and compare against ``f_numpy``."""
    rng = np.random.default_rng(seed)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, shape in INPUT_SPECS.items()}
    expected = f_numpy(**inputs)
    spec = importlib.util.spec_from_file_location("dumped_kernel", kernel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = np.asarray(simulate_fp32(module.nki_f_nkigym)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    print(f"[numerics] PASS (atol={atol}, rtol={rtol})")


if __name__ == "__main__":
    CACHE_DIR = Path("/home/ubuntu/cache/matmul_lhsT_rhs")
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    ir = build_initial_ir(f_nkigym, INPUT_SPECS)
    ir.dump(CACHE_DIR)
    # _check_numerics(CACHE_DIR / "kernel.py")
