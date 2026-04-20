"""Matrix multiplication: remote search over sampled KernelIR variants.

Three input-layout variants:
  * ``lhs_T @ rhs``  -- lhs_T(K, M), rhs(K, N) -> direct nc_matmul
  * ``lhs @ rhs``    -- lhs(M, K), rhs(K, N)   -> transpose lhs, then nc_matmul
  * ``lhs @ rhs_T``  -- lhs(M, K), rhs_T(N, K) -> transpose both, then nc_matmul

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/matmul.py
"""

import inspect
import shutil
from pathlib import Path

import numpy as np

from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.transpose import NKITranspose
from nkigym.search import remote_search


def matmul_lhsT_rhs_numpy(lhs_T: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Matrix multiply with numpy: lhs_T.T @ rhs."""
    return lhs_T.T @ rhs


def matmul_lhsT_rhs_nkigym(lhs_T: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Matrix multiply using nkigym logical ops: lhs_T.T @ rhs."""
    output = NKIMatmul()(stationary=lhs_T, moving=rhs)
    return output


def matmul_lhs_rhs_numpy(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Matrix multiply with numpy: lhs @ rhs."""
    return lhs @ rhs


def matmul_lhs_rhs_nkigym(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Matrix multiply using nkigym logical ops: lhs @ rhs."""
    lhs_T = NKITranspose()(data=lhs)
    output = NKIMatmul()(stationary=lhs_T, moving=rhs)
    return output


def matmul_lhs_rhsT_numpy(lhs: np.ndarray, rhs_T: np.ndarray) -> np.ndarray:
    """Matrix multiply with numpy: lhs @ rhs_T.T."""
    return lhs @ rhs_T.T


def matmul_lhs_rhsT_nkigym(lhs: np.ndarray, rhs_T: np.ndarray) -> np.ndarray:
    """Matrix multiply using nkigym logical ops: lhs @ rhs_T.T."""
    lhs_T = NKITranspose()(data=lhs)
    rhs = NKITranspose()(data=rhs_T)
    output = NKIMatmul()(stationary=lhs_T, moving=rhs)
    return output


if __name__ == "__main__":
    K, M, N = 2048, 2048, 2048
    HOSTS = ["gym-1", "gym-2", "gym-3"]
    NUM_VARIANTS = 50
    ATOL, RTOL = 1e-2, 1e-2

    CACHE_ROOT = Path("/home/ubuntu/cache/matmul")
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    remote_search(
        func=matmul_lhsT_rhs_nkigym,
        input_specs={"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")},
        golden_source=inspect.getsource(matmul_lhsT_rhs_numpy),
        golden_func_name="matmul_lhsT_rhs_numpy",
        hosts=HOSTS,
        cache_dir=str(CACHE_ROOT / "lhsT_rhs"),
        num_variants=NUM_VARIANTS,
        atol=ATOL,
        rtol=RTOL,
        seed=0,
    )

    remote_search(
        func=matmul_lhs_rhs_nkigym,
        input_specs={"lhs": ((M, K), "bfloat16"), "rhs": ((K, N), "bfloat16")},
        golden_source=inspect.getsource(matmul_lhs_rhs_numpy),
        golden_func_name="matmul_lhs_rhs_numpy",
        hosts=HOSTS,
        cache_dir=str(CACHE_ROOT / "lhs_rhs"),
        num_variants=NUM_VARIANTS,
        atol=ATOL,
        rtol=RTOL,
        seed=0,
    )

    remote_search(
        func=matmul_lhs_rhsT_nkigym,
        input_specs={"lhs": ((M, K), "bfloat16"), "rhs_T": ((N, K), "bfloat16")},
        golden_source=inspect.getsource(matmul_lhs_rhsT_numpy),
        golden_func_name="matmul_lhs_rhsT_numpy",
        hosts=HOSTS,
        cache_dir=str(CACHE_ROOT / "lhs_rhsT"),
        num_variants=NUM_VARIANTS,
        atol=ATOL,
        rtol=RTOL,
        seed=0,
    )
