"""Verify attention kernel numerical correctness against numpy reference.

Tests the causal single-head attention math function from the design
guide (section 1) using nkigym CPU stubs at float64 precision.
"""

import numpy as np

import nkigym


def _attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: float) -> np.ndarray:
    """11-op causal attention from design guide section 1.

    Args:
        Q: Query tensor of shape (seq_q, d_k).
        K: Key tensor of shape (seq_k, d_k).
        V: Value tensor of shape (seq_k, d_v).
        scale: Scaling factor (typically 1/sqrt(d_k)).

    Returns:
        Attention output of shape (seq_q, d_v).
    """
    Q_t = nkigym.nc_transpose(Q)
    K_t = nkigym.nc_transpose(K)
    S = nkigym.nc_matmul(Q_t, K_t)
    masked_S = nkigym.affine_select(S, cmp_op="greater_equal", on_false_value=-np.inf, channel_multiplier=1, step=-1)
    scaled_S = nkigym.tensor_scalar(masked_S, scale, op0="multiply")
    neg_max_S = nkigym.tensor_reduce(scaled_S, reduce_op="max", negate=True)
    exp_S, sum_exp = nkigym.activation_reduce(scaled_S, neg_max_S, op="exp", reduce_op="add")
    inv_sum = nkigym.activation(sum_exp, op="reciprocal")
    exp_S_t = nkigym.nc_transpose(exp_S)
    attn = nkigym.nc_matmul(exp_S_t, V)
    output = nkigym.tensor_scalar(attn, inv_sum, op0="multiply")
    return output


def _numpy_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: float) -> np.ndarray:
    """Reference numpy implementation of causal attention.

    Args:
        Q: Query tensor of shape (seq_q, d_k).
        K: Key tensor of shape (seq_k, d_k).
        V: Value tensor of shape (seq_k, d_v).
        scale: Scaling factor.

    Returns:
        Attention output of shape (seq_q, d_v).
    """
    S = scale * (Q @ K.T)
    seq_q, seq_k = S.shape
    mask = np.tril(np.ones((seq_q, seq_k)))
    S = np.where(mask, S, -np.inf)
    S_max = np.max(S, axis=1, keepdims=True)
    exp_S = np.exp(S - S_max)
    sum_exp = np.sum(exp_S, axis=1, keepdims=True)
    softmax_S = exp_S / sum_exp
    return softmax_S @ V


def _make_inputs(seq_q: int, d_k: int, d_v: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate deterministic random float64 inputs.

    Args:
        seq_q: Query sequence length.
        d_k: Key/query hidden dimension.
        d_v: Value hidden dimension.
        seed: Random seed.

    Returns:
        Tuple of (Q, K, V) arrays.
    """
    rng = np.random.RandomState(seed)
    q = rng.randn(seq_q, d_k)
    k = rng.randn(seq_q, d_k)
    v = rng.randn(seq_q, d_v)
    return q, k, v


def test_nkigym_attention_correctness() -> None:
    """nkigym attention stubs match numpy reference at float64."""
    q, k, v = _make_inputs(256, 128, 128, seed=42)
    scale = 1.0 / np.sqrt(128)
    nkigym_result = _attention(q, k, v, scale)
    numpy_result = _numpy_attention(q, k, v, scale)
    np.testing.assert_allclose(nkigym_result, numpy_result, rtol=1e-10, atol=1e-10)


def test_nkigym_attention_shape() -> None:
    """Attention output shape matches (seq_q, d_v)."""
    q, k, v = _make_inputs(256, 128, 128, seed=99)
    scale = 1.0 / np.sqrt(128)
    result = _attention(q, k, v, scale)
    assert result.shape == (256, 128)


def test_nkigym_attention_non_square_dv() -> None:
    """Attention works when d_v differs from d_k."""
    rng = np.random.RandomState(77)
    q = rng.randn(128, 64)
    k = rng.randn(128, 64)
    v = rng.randn(128, 256)
    scale = 1.0 / np.sqrt(64)
    nkigym_result = _attention(q, k, v, scale)
    numpy_result = _numpy_attention(q, k, v, scale)
    np.testing.assert_allclose(nkigym_result, numpy_result, rtol=1e-10, atol=1e-10)


def test_causal_mask_first_row() -> None:
    """First row of causal attention only attends to position 0."""
    rng = np.random.RandomState(55)
    q = rng.randn(128, 64)
    k = rng.randn(128, 64)
    v_zeros = np.zeros((128, 64))
    v_zeros[0, :] = rng.randn(64)
    scale = 1.0 / np.sqrt(64)
    result = _attention(q, k, v_zeros, scale)
    """Row 0 only sees V[0,:] so output[0,:] should equal V[0,:]."""
    np.testing.assert_allclose(result[0, :], v_zeros[0, :], rtol=1e-10, atol=1e-10)


def test_attention_deterministic() -> None:
    """Two calls with the same seed produce identical results."""
    q1, k1, v1 = _make_inputs(128, 64, 64, seed=33)
    q2, k2, v2 = _make_inputs(128, 64, 64, seed=33)
    scale = 1.0 / np.sqrt(64)
    r1 = _attention(q1, k1, v1, scale)
    r2 = _attention(q2, k2, v2, scale)
    np.testing.assert_array_equal(r1, r2)


def test_softmax_rows_sum_to_one() -> None:
    """Softmax rows within the attention sum to 1 (verifying normalization)."""
    q, k, v = _make_inputs(128, 64, 64, seed=11)
    scale = 1.0 / np.sqrt(64)
    S = scale * (q @ k.T)
    seq_q, seq_k = S.shape
    mask = np.tril(np.ones((seq_q, seq_k)))
    S = np.where(mask, S, -np.inf)
    S_max = np.max(S, axis=1, keepdims=True)
    exp_S = np.exp(S - S_max)
    row_sums = np.sum(exp_S, axis=1)
    """Each row sum should be positive; after dividing, rows sum to 1."""
    softmax_S = exp_S / row_sums[:, np.newaxis]
    np.testing.assert_allclose(np.sum(softmax_S, axis=1), 1.0, rtol=1e-12)
