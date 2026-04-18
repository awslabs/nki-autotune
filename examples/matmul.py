"""Matrix multiplication: remote search over sampled KernelIR variants.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/matmul.py
"""

import inspect
import shutil
from pathlib import Path

import numpy as np

from nkigym.codegen import build_ir
from nkigym.ops.matmul import NKIMatmul
from nkigym.search import remote_search

CACHE_DIR = Path("/home/ubuntu/cache/matmul")
HOSTS = ["gym-1", "gym-2", "gym-3"]
NUM_VARIANTS = 50


def matmul_numpy(lhs_T: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Matrix multiply with numpy: lhs_T.T @ rhs."""
    return lhs_T.T @ rhs


def matmul_nkigym(lhs_T: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Matrix multiply using nkigym logical ops."""
    result = NKIMatmul()(stationary=lhs_T, moving=rhs)
    return result


if __name__ == "__main__":
    K, M, N = 2048, 2048, 2048
    input_specs = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}

    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    CACHE_DIR.mkdir(parents=True)

    ir = build_ir(matmul_nkigym, input_specs)

    output = remote_search(
        initial_kernel=ir,
        input_specs=input_specs,
        golden_source=inspect.getsource(matmul_numpy),
        golden_func_name="matmul_numpy",
        hosts=HOSTS,
        cache_dir=str(CACHE_DIR),
        num_variants=NUM_VARIANTS,
        atol=1e-2,
        rtol=1e-2,
    )
    print(output)
