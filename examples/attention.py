"""Standard attention computed with numpy."""

import numpy as np


def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: float) -> np.ndarray:
    """
    Compute scaled dot-product attention.

    Output = softmax(scale * Q @ K^T) @ V

    Steps (mirroring the NKI reference kernel):
        1. scores = scale * Q @ K^T              (MM1)
        2. row_max = max(scores, axis=-1)         (reduce_max)
        3. exp_scores = exp(scores - row_max)     (subtract + exp)
        4. row_sum = sum(exp_scores, axis=-1)     (reduce_sum)
        5. weights = exp_scores / row_sum          (normalize)
        6. output = weights @ V                   (MM2)

    Args:
        Q: Query tensor of shape (batch, seq_q, d).
        K: Key tensor of shape (batch, seq_k, d).
        V: Value tensor of shape (batch, seq_k, d).
        scale: Scalar multiplier applied to Q @ K^T.

    Returns:
        Output tensor of shape (batch, seq_q, d).
    """
    scores = scale * (Q @ np.swapaxes(K, -2, -1))
    row_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - row_max)
    row_sum = exp_scores.sum(axis=-1, keepdims=True)
    weights = exp_scores / row_sum
    return weights @ V


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    batch, seq_q, seq_k, d = 2, 4096, 4096, 128
    Q = rng.standard_normal((batch, seq_q, d))
    K = rng.standard_normal((batch, seq_k, d))
    V = rng.standard_normal((batch, seq_k, d))

    scale = 1.0 / np.sqrt(d)
    out = attention(Q, K, V, scale)
    print(f"Q:   {Q.shape}  dtype={Q.dtype}")
    print(f"K:   {K.shape}  dtype={K.dtype}")
    print(f"V:   {V.shape}  dtype={V.dtype}")
    print(f"Out: {out.shape} dtype={out.dtype}")
    print(f"Out sample (batch=0, row=0, first 8 cols):\n{out[0, 0, :8]}")
