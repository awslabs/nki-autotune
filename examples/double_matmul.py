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
import shutil
from pathlib import Path

import numpy as np

from autotune.runner.compare import assert_close
from nkigym.codegen.render import build_ir
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.transpose import NKITranspose
from nkigym.search.api import remote_search


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
    ir = build_ir(double_matmul_nkigym, input_specs=input_specs)

    golden_source = inspect.getsource(double_matmul_numpy)
    cache_dir = Path("/home/ubuntu/cache/double_matmul_test")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    remote_search(
        initial_kernel=ir,
        golden_source=golden_source,
        golden_func_name="double_matmul_numpy",
        hosts=["gym-1", "gym-2", "gym-3", "gym-4", "gym-5", "gym-6"],
        cache_dir=str(cache_dir),
        num_variants=10,
        atol=1e-3,
        rtol=1e-3,
        warmup=10,
        iters=100,
    )
