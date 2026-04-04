"""Batched causal attention: numpy reference vs nkigym simulation.

Demonstrates that nkigym tile-level NKI ops produce identical results
to numpy at float64 precision for causal masked attention with a batch
dimension.
"""

import numpy as np

import nkigym


def attention_numpy(Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: float) -> np.ndarray:
    """Batched causal attention with numpy.

    softmax(mask(scale * Q @ K^T)) @ V with lower-triangular causal mask.

    Args:
        Q: Query tensor of shape (batch, seq_q, d).
        K: Key tensor of shape (batch, seq_k, d).
        V: Value tensor of shape (batch, seq_k, d_v).
        scale: Scalar multiplier.

    Returns:
        Output tensor of shape (batch, seq_q, d_v).
    """
    batch, seq_q, _ = Q.shape
    seq_k = K.shape[1]
    row_idx = np.arange(seq_q)[:, np.newaxis]
    col_idx = np.arange(seq_k)[np.newaxis, :]
    causal_mask = row_idx >= col_idx

    out = np.empty((batch, seq_q, V.shape[2]), dtype=Q.dtype)
    for b in range(batch):
        scores = scale * (Q[b] @ K[b].T)
        scores = np.where(causal_mask, scores, -np.inf)
        row_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - row_max)
        row_sum = exp_scores.sum(axis=-1, keepdims=True)
        weights = exp_scores / row_sum
        out[b] = weights @ V[b]
    return out


def attention_nkigym(Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: float) -> np.ndarray:
    """Batched causal attention using nkigym logical ops.

    Same math as attention_numpy, expressed as NKI tile operations.
    The batch dimension is an explicit loop since nkigym ops are 2D tiles.

    Per-batch steps:
        1. Q_t, K_t = transpose Q[b] and K[b]
        2. S = Q_t^T @ K_t = Q[b] @ K[b]^T
        3. scaled_S = S * scale
        4. masked_S = affine_select causal mask
        5. neg_max = -max(masked_S, axis=1)
        6. exp_S, sum_exp = exp(masked_S + neg_max), sum(exp_S)
        7. inv_sum = 1/sum_exp
        8. exp_S_t = transpose exp_S
        9. attn = exp_S_t^T @ V[b]
       10. output = attn * inv_sum

    Args:
        Q: Query tensor of shape (batch, seq_q, d).
        K: Key tensor of shape (batch, seq_k, d).
        V: Value tensor of shape (batch, seq_k, d_v).
        scale: Scalar multiplier.

    Returns:
        Output tensor of shape (batch, seq_q, d_v).
    """
    batch, seq_q, _ = Q.shape
    seq_k = K.shape[1]

    out = np.empty((batch, seq_q, V.shape[2]), dtype=Q.dtype)
    for b in range(batch):
        Q_t = nkigym.nc_transpose(Q[b])
        K_t = nkigym.nc_transpose(K[b])
        S = nkigym.nc_matmul(Q_t, K_t)
        scaled_S = nkigym.tensor_scalar(S, scale, op0="multiply")
        masked_S = nkigym.affine_select(
            scaled_S, pattern=[[-1, seq_k]], channel_multiplier=1, on_false_value=-np.inf, cmp_op="greater_equal"
        )
        neg_max = nkigym.tensor_reduce(masked_S, op="max", negate=True)
        exp_S, sum_exp = nkigym.activation_reduce(masked_S, op="exp", reduce_op="add", bias=neg_max)
        inv_sum = nkigym.activation(sum_exp, op="reciprocal")
        exp_S_t = nkigym.nc_transpose(exp_S)
        attn = nkigym.nc_matmul(exp_S_t, V[b])
        out[b] = nkigym.tensor_scalar(attn, inv_sum, op0="multiply")
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    batch, seq_len, d_k, d_v = 4, 128, 64, 64
    Q = rng.standard_normal((batch, seq_len, d_k))
    K = rng.standard_normal((batch, seq_len, d_k))
    V = rng.standard_normal((batch, seq_len, d_v))
    scale = 1.0 / np.sqrt(d_k)

    out_np = attention_numpy(Q, K, V, scale)
    out_gym = attention_nkigym(Q, K, V, scale)

    print(f"Q: {Q.shape}  K: {K.shape}  V: {V.shape}")
    print(f"scale: {scale:.6f}")
    print(f"numpy  sample [0]: {out_np[0, 0, :4]}")
    print(f"nkigym sample [0]: {out_gym[0, 0, :4]}")
    max_diff = np.max(np.abs(out_np - out_gym))
    print(f"max |diff|: {max_diff:.2e}")
    np.testing.assert_allclose(out_gym, out_np, rtol=1e-10, atol=1e-10)
    print("PASS: causal batched nkigym matches numpy")
