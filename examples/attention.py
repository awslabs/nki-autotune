"""Causal attention: remote search over sampled KernelIR variants.

softmax(mask(scale * Q @ K^T)) @ V with lower-triangular causal mask.

Expressed as NKI ops (design doc section 1):
  Q_t      = nc_transpose(Q)
  K_t      = nc_transpose(K)
  S        = nc_matmul(Q_t, K_t)          Q @ K^T
  masked_S = affine_select(S, causal)
  scaled_S = tensor_scalar(masked_S * scale)
  neg_max  = tensor_reduce(scaled_S, max, negate)
  exp_S, sum_exp = activation_reduce(scaled_S, exp, add, bias=neg_max)
  inv_sum  = activation(sum_exp, reciprocal)
  exp_S_t  = nc_transpose(exp_S)
  attn     = nc_matmul(exp_S_t, V)        exp_S @ V
  output   = tensor_scalar(attn * inv_sum)

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/attention.py
"""

import shutil
from pathlib import Path

import numpy as np

from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.affine_select import NKIAffineSelect
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose
from nkigym.search import remote_search


def attention_nkigym(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Causal attention using nkigym NKIOp classes.

    Args:
        Q: Query tensor of shape (seq_q, d_k).
        K: Key tensor of shape (seq_k, d_k).
        V: Value tensor of shape (seq_k, d_v).

    Returns:
        Output tensor of shape (seq_q, d_v).
    """
    d_k = Q.shape[1]
    scale = 1.0 / np.sqrt(d_k)
    Q_t = NKITranspose()(data=Q)
    K_t = NKITranspose()(data=K)
    S = NKIMatmul()(stationary=Q_t, moving=K_t)
    masked_S = NKIAffineSelect()(
        on_true_tile=S, pattern=[[-1, K.shape[0]]], channel_multiplier=1, on_false_value=-np.inf, cmp_op="greater_equal"
    )
    scaled_S = NKITensorScalar()(data=masked_S, op0="multiply", operand0=scale)
    neg_max = NKITensorReduce()(data=scaled_S, op="maximum", axis=1, negate=True)
    exp_S, sum_exp = NKIActivationReduce()(data=scaled_S, op="exp", reduce_op="add", bias=neg_max)
    inv_sum = NKIActivation()(data=sum_exp, op="reciprocal")
    exp_S_t = NKITranspose()(data=exp_S)
    attn = NKIMatmul()(stationary=exp_S_t, moving=V)
    output = NKITensorScalar()(data=attn, op0="multiply", operand0=inv_sum)
    return output


if __name__ == "__main__":
    seq_len, d_k, d_v = 512, 128, 128
    input_specs = {
        "Q": ((seq_len, d_k), "bfloat16"),
        "K": ((seq_len, d_k), "bfloat16"),
        "V": ((seq_len, d_v), "bfloat16"),
    }

    CACHE_DIR = Path("/home/ubuntu/cache/attention")
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    CACHE_DIR.mkdir(parents=True)

    output = remote_search(
        func=attention_nkigym,
        input_specs=input_specs,
        hosts=["gym-1", "gym-2", "gym-3"],
        cache_dir=str(CACHE_DIR),
        num_variants=50,
        atol=1e-2,
        rtol=1e-2,
        seed=0,
    )
