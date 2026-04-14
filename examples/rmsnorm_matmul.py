"""RMSNorm + matmul: numpy golden, nkigym simulation, and comparison.

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

import numpy as np

from autotune.runner.compare import assert_close
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose

EPS = 1e-6


def rmsnorm_matmul_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """RMSNorm(a) @ b with numpy.

    Args:
        a: Input tensor of shape (M, K).
        b: Weight tensor of shape (K, N).

    Returns:
        Output tensor of shape (M, N).
    """
    k = a.shape[1]
    rms = np.sqrt(np.mean(a**2, axis=1, keepdims=True) + EPS)
    a_normed = a / rms
    return a_normed @ b


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
    M, K, N = 1024, 1024, 1024

    rng = np.random.default_rng(42)
    a = rng.standard_normal((M, K))
    b = rng.standard_normal((K, N))

    out_np = rmsnorm_matmul_numpy(a, b)
    out_gym = rmsnorm_matmul_nkigym(a, b)
    status = assert_close(out_gym, out_np, atol=1e-10, rtol=1e-10)
    print(f"rmsnorm_matmul: {status}")
