"""Causal attention: numpy golden, nkigym simulation, and comparison.

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

import nki
import numpy as np

from autotune.runner.compare import assert_close
from autotune.runner.compile import load_kernel
from nkigym.codegen import build_ir, render_ir
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.affine_select import NKIAffineSelect
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose

CACHE_DIR = Path("/home/ubuntu/cache/attention")


def attention_numpy(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Causal attention with numpy.

    Args:
        Q: Query tensor of shape (seq_q, d_k).
        K: Key tensor of shape (seq_k, d_k).
        V: Value tensor of shape (seq_k, d_v).

    Returns:
        Output tensor of shape (seq_q, d_v).
    """
    d_k = Q.shape[1]
    scale = 1.0 / np.sqrt(d_k)
    seq_q = Q.shape[0]
    seq_k = K.shape[0]
    scores = scale * (Q @ K.T)
    row_idx = np.arange(seq_q)[:, np.newaxis]
    col_idx = np.arange(seq_k)[np.newaxis, :]
    causal_mask = row_idx >= col_idx
    scores = np.where(causal_mask, scores, -np.inf)
    row_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - row_max)
    row_sum = exp_scores.sum(axis=-1, keepdims=True)
    weights = exp_scores / row_sum
    return weights @ V


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
    neg_max = NKITensorReduce()(data=scaled_S, op="max", axis=1, negate=True)
    exp_S, sum_exp = NKIActivationReduce()(data=scaled_S, op="exp", reduce_op="add", bias=neg_max)
    inv_sum = NKIActivation()(data=sum_exp, op="reciprocal")
    exp_S_t = NKITranspose()(data=exp_S)
    attn = NKIMatmul()(stationary=exp_S_t, moving=V)
    output = NKITensorScalar()(data=attn, op0="multiply", operand0=inv_sum)
    return output


if __name__ == "__main__":
    seq_len, d_k, d_v = 2048, 128, 128

    rng = np.random.default_rng(42)
    Q = rng.standard_normal((seq_len, d_k))
    K = rng.standard_normal((seq_len, d_k))
    V = rng.standard_normal((seq_len, d_v))

    out_np = attention_numpy(Q, K, V)
    out_gym = attention_nkigym(Q, K, V)
    status = assert_close(out_gym, out_np, atol=1e-10, rtol=1e-10)
    print(f"attention: {status}")

    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    CACHE_DIR.mkdir(parents=True)

    input_specs = {
        "Q": ((seq_len, d_k), "bfloat16"),
        "K": ((seq_len, d_k), "bfloat16"),
        "V": ((seq_len, d_v), "bfloat16"),
    }
    ir = build_ir(attention_nkigym, input_specs)
    (CACHE_DIR / "dim_analysis.txt").write_text(repr(ir.dim_analysis))

    ir.op_graph.render(CACHE_DIR / "op_graph")

    source = render_ir(ir)
    (CACHE_DIR / "kernel.py").write_text(source)

    kernel_func = load_kernel(str(CACHE_DIR / "kernel.py"), "attention_nkigym")
    golden = attention_numpy(Q.astype(np.float32), K.astype(np.float32), V.astype(np.float32))
    sim_result = nki.simulate(kernel_func)(Q=Q.astype(np.float32), K=K.astype(np.float32), V=V.astype(np.float32))
    sim_status = assert_close(sim_result, golden, atol=1e-1, rtol=1e-1)
    print(f"attention cpu_sim: {sim_status}")
