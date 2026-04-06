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

import nkigym
from autotune.runner.api import remote_profile
from autotune.runner.types import KernelJob


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
    output = nkigym.nc_matmul(lhs_T, rhs)
    return output


if __name__ == "__main__":

    K, M, N = 2048, 2048, 2048

    lhs_T = np.random.randn(K, M)
    rhs = np.random.randn(K, N)

    out_np = matmul_numpy(lhs_T, rhs)
    out_gym = matmul_nkigym(lhs_T, rhs)
    max_diff = np.max(np.abs(out_np - out_gym))
    print(f"max |diff|: {max_diff:.2e}")
    np.testing.assert_allclose(out_gym, out_np, rtol=1e-10, atol=1e-10)
    print("PASS: matmul nkigym matches numpy")
    kernel_src = nkigym.render(matmul_nkigym, lhs_T=lhs_T, rhs=rhs)

    golden_source = inspect.getsource(matmul_numpy)
    kernels = {
        "matmul_nkigym": KernelJob(
            source=kernel_src,
            input_specs={"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")},
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
    print(output)
