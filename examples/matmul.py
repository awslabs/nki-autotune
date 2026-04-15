"""Matrix multiplication: numpy golden, nkigym simulation, and comparison.

Demonstrates that NKIMatmul produces identical results to numpy
at float64 precision.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/matmul.py
"""

import shutil
from pathlib import Path

import numpy as np

from autotune.runner.compare import assert_close
from nkigym.codegen import build_ir, render_ir
from nkigym.ops.matmul import NKIMatmul

CACHE_DIR = Path("/home/ubuntu/cache/matmul")


def matmul_numpy(lhs_T: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Matrix multiply with numpy: lhs_T.T @ rhs.

    Args:
        lhs_T: Stationary tensor of shape (K, M).
        rhs: Moving tensor of shape (K, N).

    Returns:
        Output tensor of shape (M, N).
    """
    return lhs_T.T @ rhs


def matmul_nkigym(lhs_T: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Matrix multiply using nkigym logical ops.

    Args:
        lhs_T: Stationary tensor of shape (K, M).
        rhs: Moving tensor of shape (K, N).

    Returns:
        Output tensor of shape (M, N).
    """
    result = NKIMatmul()(stationary=lhs_T, moving=rhs)
    return result


if __name__ == "__main__":
    K, M, N = 8192, 8192, 8192

    rng = np.random.default_rng(42)
    lhs_T = rng.standard_normal((K, M))
    rhs = rng.standard_normal((K, N))

    out_np = matmul_numpy(lhs_T, rhs)
    out_gym = matmul_nkigym(lhs_T, rhs)
    status = assert_close(out_gym, out_np, atol=1e-10, rtol=1e-10)
    print(f"matmul: {status}")

    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    CACHE_DIR.mkdir(parents=True)

    input_specs = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}

    ir = build_ir(matmul_nkigym, input_specs)
    (CACHE_DIR / "dim_analysis.txt").write_text(repr(ir.dim_analysis))

    ir.op_graph.render(CACHE_DIR / "op_graph")

    source = render_ir(ir)
    (CACHE_DIR / "kernel.py").write_text(source)
