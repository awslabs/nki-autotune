"""Causal attention: numpy reference, nkigym simulation, and remote profiling.

Demonstrates that nkigym tile-level NKI ops produce identical results
to numpy at float64 precision for causal masked attention, renders
the naive NKI kernel, then compiles and benchmarks it on remote
Trainium workers.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/attention.py
"""

import inspect
import shutil
from pathlib import Path

import numpy as np

from autotune.runner.compare import assert_close
from nkigym.codegen.render import build_ir
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.affine_select import NKIAffineSelect
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose
from nkigym.search.api import remote_search


def attention_numpy(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Causal attention with numpy.

    softmax(mask(scale * Q @ K^T)) @ V with lower-triangular causal mask.
    scale = 1/sqrt(d_k) derived from Q.shape[1].

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

    Same math as attention_numpy, expressed as NKI tile operations:
        1. Q_t, K_t = transpose Q and K
        2. S = Q_t^T @ K_t = Q @ K^T
        3. scaled_S = S * scale
        4. masked_S = affine_select causal mask
        5. neg_max = -max(masked_S, axis=1)
        6. exp_S, sum_exp = exp(masked_S + neg_max), sum(exp_S)
        7. inv_sum = 1/sum_exp
        8. exp_S_t = transpose exp_S
        9. attn = exp_S_t^T @ V
       10. output = attn * inv_sum

    Args:
        Q: Query tensor of shape (seq_q, d_k).
        K: Key tensor of shape (seq_k, d_k).
        V: Value tensor of shape (seq_k, d_v).

    Returns:
        Output tensor of shape (seq_q, d_v).
    """
    d_k = Q.shape[1]
    scale = 1.0 / np.sqrt(d_k)
    seq_k = K.shape[0]
    Q_t = NKITranspose()(data=Q)
    K_t = NKITranspose()(data=K)
    S = NKIMatmul()(stationary=Q_t, moving=K_t)
    scaled_S = NKITensorScalar()(data=S, scalar0=scale, op0="multiply")
    masked_S = NKIAffineSelect()(
        data=scaled_S, pattern=[[-1, seq_k]], channel_multiplier=1, on_false_value=-np.inf, cmp_op="greater_equal"
    )
    neg_max = NKITensorReduce()(data=masked_S, op="max", negate=True)
    exp_S, sum_exp = NKIActivationReduce()(data=masked_S, op="exp", reduce_op="add", bias=neg_max)
    inv_sum = NKIActivation()(data=sum_exp, op="reciprocal")
    exp_S_t = NKITranspose()(data=exp_S)
    attn = NKIMatmul()(stationary=exp_S_t, moving=V)
    output = NKITensorScalar()(data=attn, scalar0=inv_sum, op0="multiply")
    return output


if __name__ == "__main__":

    seq_len, d_k, d_v = 2048, 128, 128

    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_v)

    out_np = attention_numpy(Q, K, V)
    out_gym = attention_nkigym(Q, K, V)
    status = assert_close(out_gym, out_np, atol=1e-10, rtol=1e-10)
    print(status)

    input_specs = {
        "Q": ((seq_len, d_k), "bfloat16"),
        "K": ((seq_len, d_k), "bfloat16"),
        "V": ((seq_len, d_v), "bfloat16"),
    }
    ir = build_ir(attention_nkigym, input_specs=input_specs)

    golden_source = inspect.getsource(attention_numpy)
    cache_dir = Path("/home/ubuntu/cache/attention_test")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    remote_search(
        initial_kernel=ir,
        golden_source=golden_source,
        golden_func_name="attention_numpy",
        hosts=["gym-1", "gym-2", "gym-3", "gym-4", "gym-5", "gym-6"],
        cache_dir=str(cache_dir),
        num_variants=10,
        transforms=[],
        atol=1e-2,
        rtol=1e-2,
        warmup=10,
        iters=100,
    )
