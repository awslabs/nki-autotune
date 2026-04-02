"""Tests for attention function numerical correctness.

Verifies that the nkigym attention stubs produce output matching
a numpy reference implementation of causal single-head attention:
softmax(mask(scale * Q @ K^T)) @ V.
"""

import numpy as np

import nkigym


def _causal_mask(scores: np.ndarray) -> np.ndarray:
    """Apply causal mask: set upper-triangular entries to -inf.

    Args:
        scores: Attention score matrix of shape [seq_q, seq_k].

    Returns:
        Masked scores with -inf above the diagonal.
    """
    seq_q, seq_k = scores.shape
    mask = np.tril(np.ones((seq_q, seq_k), dtype=bool))
    return np.where(mask, scores, -np.inf)


def _safe_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis.

    Args:
        x: Input array.

    Returns:
        Softmax probabilities.
    """
    x_max = np.max(x, axis=-1, keepdims=True)
    """
    Handle rows that are all -inf (masked out entirely).
    Replace -inf max with 0 so exp(-inf - 0) = 0 rather than nan.
    """
    x_max = np.where(np.isinf(x_max), 0.0, x_max)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def _attention_reference(Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: float) -> np.ndarray:
    """Compute causal attention using numpy.

    Reference: softmax(mask(scale * Q @ K^T)) @ V.

    Args:
        Q: Query tensor of shape [seq_q, d_k].
        K: Key tensor of shape [seq_k, d_k].
        V: Value tensor of shape [seq_k, d_v].
        scale: Scaling factor.

    Returns:
        Output tensor of shape [seq_q, d_v].
    """
    scores = scale * (Q @ K.T)
    masked = _causal_mask(scores)
    probs = _safe_softmax(masked)
    return probs @ V


def _attention_nkigym(Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: float) -> np.ndarray:
    """Compute attention using nkigym stubs.

    Mirrors the design guide section 1 math function.

    Args:
        Q: Query tensor.
        K: Key tensor.
        V: Value tensor.
        scale: Scaling factor.

    Returns:
        Attention output.
    """
    Q_t = nkigym.transpose(Q)
    K_t = nkigym.transpose(K)
    S = nkigym.nc_matmul(Q_t, K_t)
    scaled_S = nkigym.tensor_scalar(S, op0=np.multiply, operand0=scale)
    masked_S = nkigym.affine_select(
        scaled_S, cmp_op="greater_equal", on_false_value=-np.inf, channel_multiplier=1, step=-1
    )
    max_S = nkigym.tensor_reduce(masked_S, op=np.maximum)
    shifted_S = nkigym.tensor_scalar(masked_S, max_S, op0=np.subtract)
    exp_S = nkigym.activation(shifted_S, op=np.exp)
    sum_exp = nkigym.tensor_reduce(exp_S, op=np.add)
    inv_sum = nkigym.activation(sum_exp, op="reciprocal")
    exp_S_t = nkigym.transpose(exp_S)
    attn = nkigym.nc_matmul(exp_S_t, V)
    output = nkigym.tensor_scalar(attn, inv_sum, op0=np.multiply)
    return output


def test_attention_small_square() -> None:
    """Attention 128x128 Q/K, 128x128 V matches numpy reference."""
    rng = np.random.RandomState(42)
    q = rng.randn(128, 128)
    k = rng.randn(128, 128)
    v = rng.randn(128, 128)
    scale = 1.0 / np.sqrt(128)
    expected = _attention_reference(q, k, v, scale)
    actual = _attention_nkigym(q, k, v, scale)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_attention_256x128() -> None:
    """Attention 256x128 Q/K, 256x128 V matches numpy reference."""
    rng = np.random.RandomState(99)
    q = rng.randn(256, 128)
    k = rng.randn(256, 128)
    v = rng.randn(256, 128)
    scale = 1.0 / np.sqrt(128)
    expected = _attention_reference(q, k, v, scale)
    actual = _attention_nkigym(q, k, v, scale)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_attention_rectangular_v() -> None:
    """Attention 128x64 Q/K, 128x256 V produces [128, 256] output."""
    rng = np.random.RandomState(77)
    q = rng.randn(128, 64)
    k = rng.randn(128, 64)
    v = rng.randn(128, 256)
    scale = 1.0 / np.sqrt(64)
    expected = _attention_reference(q, k, v, scale)
    actual = _attention_nkigym(q, k, v, scale)
    assert actual.shape == (128, 256)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_causal_mask_upper_triangle() -> None:
    """Causal mask sets upper-triangular entries to -inf."""
    scores = np.ones((4, 4))
    masked = _causal_mask(scores)
    assert masked[0, 1] == -np.inf
    assert masked[0, 3] == -np.inf
    assert masked[1, 0] == 1.0
    assert masked[3, 3] == 1.0


def test_affine_select_matches_causal() -> None:
    """nkigym.affine_select reproduces the causal mask pattern."""
    rng = np.random.RandomState(55)
    data = rng.randn(8, 8)
    result = nkigym.affine_select(data, cmp_op="greater_equal", on_false_value=-np.inf, channel_multiplier=1, step=-1)
    expected = _causal_mask(data)
    np.testing.assert_array_equal(result, expected)


def test_activation_reciprocal() -> None:
    """nkigym.activation with op='reciprocal' computes 1/x."""
    data = np.array([2.0, 4.0, 0.5, 10.0])
    result = nkigym.activation(data, op="reciprocal")
    expected = np.array([0.5, 0.25, 2.0, 0.1])
    np.testing.assert_allclose(result, expected, rtol=1e-15)


def test_tensor_reduce_max() -> None:
    """nkigym.tensor_reduce with op=np.maximum computes row-wise max."""
    data = np.array([[1.0, 3.0, 2.0], [5.0, 0.0, 4.0]])
    result = nkigym.tensor_reduce(data, op=np.maximum)
    expected = np.array([3.0, 5.0])
    np.testing.assert_array_equal(result, expected)


def test_nc_transpose_stub() -> None:
    """nkigym.nc_transpose is equivalent to .T."""
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = nkigym.nc_transpose(data)
    np.testing.assert_array_equal(result, data.T)
