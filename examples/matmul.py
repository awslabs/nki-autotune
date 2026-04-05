"""Matrix multiplication: numpy reference, nkigym simulation, and remote profiling.

Demonstrates that nkigym nc_matmul produces identical results
to numpy at float64 precision, renders the naive NKI kernel,
then compiles and benchmarks it on remote Trainium workers.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/matmul.py --hosts gym-1 gym-2
"""

import numpy as np

import nkigym
from autotune.runner.api import remote_profile
from autotune.runner.types import ProfileConfig

CACHE_DIR = "/home/ubuntu/cache/matmul_test"

K, M, N = 2048, 2048, 2048
MAC_COUNT = K * M * N

GOLDEN_SOURCE = """\
import numpy as np


def golden_matmul(a, b):
    return a.astype(np.float64).T @ b.astype(np.float64)
"""


def matmul_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiply with numpy: a.T @ b.

    Args:
        a: Stationary tensor of shape (K, M).
        b: Moving tensor of shape (K, N).

    Returns:
        Output tensor of shape (M, N).
    """
    return a.T @ b


def matmul_nkigym(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiply using nkigym logical ops.

    Args:
        a: Stationary tensor of shape (K, M).
        b: Moving tensor of shape (K, N).

    Returns:
        Output tensor of shape (M, N).
    """
    output = nkigym.nc_matmul(a, b)
    return output


HOSTS = ["gym-1", "gym-2", "gym-3", "gym-4", "gym-5"]


if __name__ == "__main__":

    a = np.random.randn(K, M)
    b = np.random.randn(K, N)

    out_np = matmul_numpy(a, b)
    out_gym = matmul_nkigym(a, b)
    max_diff = np.max(np.abs(out_np - out_gym))
    print(f"max |diff|: {max_diff:.2e}")
    np.testing.assert_allclose(out_gym, out_np, rtol=1e-10, atol=1e-10)
    print("PASS: matmul nkigym matches numpy")

    kernel_src = nkigym.render(matmul_nkigym, a=a, b=b)
    output = remote_profile(
        kernels={"matmul_kernel.py": kernel_src},
        input_specs={"a": ((K, M), "bfloat16"), "b": ((K, N), "bfloat16")},
        hosts=HOSTS,
        cache_dir=CACHE_DIR,
        config=ProfileConfig(
            mac_count=MAC_COUNT,
            golden_source=GOLDEN_SOURCE,
            golden_func_name="golden_matmul",
            atol=1e-3,
            rtol=1e-3,
            warmup=10,
            iters=100,
        ),
    )
    print(output)
