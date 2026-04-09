"""Matrix multiplication: numpy reference, nkigym simulation, and remote profiling.

Demonstrates that nkigym nc_matmul produces identical results
to numpy at float64 precision, renders the naive NKI kernel,
then compiles and benchmarks it on remote Trainium workers.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/matmul.py
"""

import inspect
import shutil
from pathlib import Path

import numpy as np

from autotune.runner.compare import assert_close
from nkigym.codegen.render import build_ir
from nkigym.ops.matmul import NKIMatmul
from nkigym.search.api import remote_search


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

    lhs_T = np.random.randn(K, M)
    rhs = np.random.randn(K, N)

    out_np = matmul_numpy(lhs_T, rhs)
    out_gym = matmul_nkigym(lhs_T, rhs)
    status = assert_close(out_gym, out_np, atol=1e-10, rtol=1e-10)
    print(status)
    input_specs = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}
    ir = build_ir(matmul_nkigym, input_specs=input_specs)

    golden_source = inspect.getsource(matmul_numpy)
    cache_dir = Path("/home/ubuntu/cache/matmul_test")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    remote_search(
        initial_kernel=ir,
        golden_source=golden_source,
        golden_func_name="matmul_numpy",
        hosts=["gym-1", "gym-2", "gym-3", "gym-4", "gym-5", "gym-6"],
        cache_dir=str(cache_dir),
        num_variants=10,
        transforms=[],
        atol=1e-3,
        rtol=1e-3,
        warmup=10,
        iters=100,
    )
