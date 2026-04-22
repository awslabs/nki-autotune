"""Double matmul: remote search over sampled KernelContext variants.

Math: output = (Q @ K.T) @ V

Expressed as NKI ops:
  Q_t  = nc_transpose(Q)        -- Q(d0, d1) -> Q_t(d1, d0)
  K_t  = nc_transpose(K)        -- K(d2, d1) -> K_t(d1, d2)
  S    = nc_matmul(Q_t, K_t)    -- Q_t.T @ K_t = Q @ K.T -> S(d0, d2)
  S_t  = nc_transpose(S)        -- S(d0, d2) -> S_t(d2, d0)
  out  = nc_matmul(S_t, V)      -- S_t.T @ V = S @ V -> out(d0, d3)

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/double_matmul.py
"""

import shutil
from pathlib import Path

import numpy as np

from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.transpose import NKITranspose
from nkigym.search import remote_search


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
    seq_q, d_k, seq_k, d_v = 2048, 2048, 2048, 2048
    input_specs = {"Q": ((seq_q, d_k), "bfloat16"), "K": ((seq_k, d_k), "bfloat16"), "V": ((seq_k, d_v), "bfloat16")}

    CACHE_DIR = Path("/home/ubuntu/cache/double_matmul")
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    CACHE_DIR.mkdir(parents=True)

    output = remote_search(
        func=double_matmul_nkigym,
        input_specs=input_specs,
        hosts=["gym-1", "gym-2", "gym-3"],
        cache_dir=str(CACHE_DIR),
        num_variants=50,
        atol=1e-2,
        rtol=1e-2,
        seed=3279,
    )
