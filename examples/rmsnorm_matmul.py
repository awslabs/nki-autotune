"""RMSNorm + matmul: remote search over sampled KernelIR variants.

Math: RMSNorm(a) @ b = (a / sqrt(mean(a^2) + eps)) @ b

Expressed as NKI ops:
  sq, sum_sq = activation_reduce(a, square, add)
  scaled     = tensor_scalar(sum_sq * (1/K) + eps)
  rsqrt_val  = activation(scaled, rsqrt)
  a_normed   = tensor_scalar(a * rsqrt_val)
  a_t        = nc_transpose(a_normed)
  result     = nc_matmul(a_t, b)

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/rmsnorm_matmul.py
"""

import shutil
from pathlib import Path

import numpy as np

from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose
from nkigym.search import remote_search

EPS = 1e-6


def rmsnorm_matmul_nkigym(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """RMSNorm(a) @ b using nkigym NKIOp classes.

    Args:
        a: Input tensor of shape (M, K).
        b: Weight tensor of shape (K, N).

    Returns:
        Output tensor of shape (M, N).
    """
    k = a.shape[1]
    sq, sum_sq = NKIActivationReduce()(data=a, op="square", reduce_op="add")
    scaled = NKITensorScalar()(data=sum_sq, op0="multiply", operand0=1.0 / k, op1="add", operand1=EPS)
    rsqrt_val = NKIActivation()(data=scaled, op="rsqrt")
    a_normed = NKITensorScalar()(data=a, op0="multiply", operand0=rsqrt_val)
    a_t = NKITranspose()(data=a_normed)
    result = NKIMatmul()(stationary=a_t, moving=b)
    return result


if __name__ == "__main__":
    M, K, N = 2048, 2048, 2048
    input_specs = {"a": ((M, K), "bfloat16"), "b": ((K, N), "bfloat16")}

    CACHE_DIR = Path("/home/ubuntu/cache/rmsnorm_matmul")
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    CACHE_DIR.mkdir(parents=True)

    output = remote_search(
        func=rmsnorm_matmul_nkigym,
        input_specs=input_specs,
        hosts=["gym-1", "gym-2", "gym-3"],
        cache_dir=str(CACHE_DIR),
        num_variants=100,
        atol=1e-2,
        rtol=1e-2,
        seed=0,
    )
