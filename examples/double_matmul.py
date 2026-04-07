"""Double matmul: simplified attention backbone (transpose + matmul only).

Math: output = (Q @ K.T) @ V

Expressed as NKI ops:
  Q_t  = nc_transpose(Q)        -- Q(d0, d1) -> Q_t(d1, d0)
  K_t  = nc_transpose(K)        -- K(d2, d1) -> K_t(d1, d2)
  S    = nc_matmul(Q_t, K_t)    -- Q_t.T @ K_t = Q @ K.T -> S(d0, d2)
  S_t  = nc_transpose(S)        -- S(d0, d2) -> S_t(d2, d0)
  out  = nc_matmul(S_t, V)      -- S_t.T @ V = S @ V -> out(d0, d3)

CPU sim should pass. Hardware run will likely fail due to oversized
inter-op SBUF buffers — that is expected and will be fixed by transforms.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/double_matmul.py
"""

import inspect

import numpy as np

from autotune.runner.api import remote_profile
from autotune.runner.compare import assert_close
from autotune.runner.types import KernelJob
from nkigym.codegen.render import render
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.transpose import NKITranspose


def double_matmul_numpy(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Reference: (Q @ K.T) @ V.

    Args:
        Q: Shape (seq_q, d_k).
        K: Shape (seq_k, d_k).
        V: Shape (seq_k, d_v).

    Returns:
        Output of shape (seq_q, d_v).
    """
    return (Q @ K.T) @ V


def double_matmul_nkigym(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Double matmul using nkigym ops.

    Args:
        Q: Shape (seq_q, d_k).
        K: Shape (seq_k, d_k).
        V: Shape (seq_k, d_v).

    Returns:
        Output of shape (seq_q, d_v).
    """
    Q_t = NKITranspose()(data=Q)
    K_t = NKITranspose()(data=K)
    S = NKIMatmul()(stationary=Q_t, moving=K_t)
    S_t = NKITranspose()(data=S)
    output = NKIMatmul()(stationary=S_t, moving=V)
    return output


if __name__ == "__main__":
    seq_q = 2048
    d_k = 128
    seq_k = 2048
    d_v = 128

    rng = np.random.default_rng(42)
    Q = rng.standard_normal((seq_q, d_k))
    K = rng.standard_normal((seq_k, d_k))
    V = rng.standard_normal((seq_k, d_v))

    out_np = double_matmul_numpy(Q, K, V)
    out_gym = double_matmul_nkigym(Q, K, V)
    status = assert_close(out_gym, out_np, atol=1e-6, rtol=1e-6)
    print(f"CPU sim: {status}")

    input_specs = {"Q": ((seq_q, d_k), "bfloat16"), "K": ((seq_k, d_k), "bfloat16"), "V": ((seq_k, d_v), "bfloat16")}
    kernel_src = render(double_matmul_nkigym, input_specs=input_specs)

    golden_source = inspect.getsource(double_matmul_numpy)
    worker_specs = {name: (spec[0], spec[1]) for name, spec in input_specs.items()}
    kernels = {
        "double_matmul_nkigym": KernelJob(
            source=kernel_src,
            input_specs=worker_specs,
            golden_source=golden_source,
            golden_func_name="double_matmul_numpy",
            atol=0.5,
            rtol=0.1,
        )
    }

    remote_profile(
        kernels=kernels,
        hosts=["gym-1", "gym-2", "gym-3", "gym-4", "gym-5", "gym-6"],
        cache_dir="/home/ubuntu/cache/double_matmul_test",
        warmup=10,
        iters=100,
    )
