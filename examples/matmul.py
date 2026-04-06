"""Matrix multiplication: numpy reference, nkigym simulation, and remote profiling.

Demonstrates that nkigym nc_matmul produces identical results
to numpy at float64 precision, renders the naive NKI kernel,
then compiles and benchmarks it on remote Trainium workers.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/matmul.py
"""

import inspect

import numpy as np

from autotune.runner.api import remote_profile
from autotune.runner.compare import assert_close
from autotune.runner.types import KernelJob
from nkigym.codegen.render import render
from nkigym.ops.matmul import NKIMatmul


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
    haha = NKIMatmul()(stationary=lhs_T, moving=rhs)
    return haha


if __name__ == "__main__":

    K, M, N = 2048, 2048, 2048

    lhs_T = np.random.randn(K, M)
    rhs = np.random.randn(K, N)

    out_np = matmul_numpy(lhs_T, rhs)
    out_gym = matmul_nkigym(lhs_T, rhs)
    status = assert_close(out_gym, out_np, atol=1e-10, rtol=1e-10)
    print(status)
    input_specs = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}
    kernel_src = render(matmul_nkigym, input_specs=input_specs)

    golden_source = inspect.getsource(matmul_numpy)
    kernels = {
        "matmul_nkigym": KernelJob(
            source=kernel_src,
            input_specs=input_specs,
            golden_source=golden_source,
            golden_func_name="matmul_numpy",
            atol=1e-3,
            rtol=1e-3,
        )
    }

    output = remote_profile(
        kernels=kernels,
        hosts=["gym-1", "gym-2", "gym-3", "gym-4", "gym-5", "gym-6"],
        cache_dir="/home/ubuntu/cache/matmul_test",
        warmup=10,
        iters=100,
    )
