import numpy as np


def silu(x):
    return x / (1 + np.exp(-x))


def cpu_golden_result(hidden, gate, gamma, qkv_weights, eps):
    if gate is not None:
        hidden = hidden * silu(gate.astype(np.float32))
    rms = np.sqrt(np.mean(np.square(hidden), axis=-1, keepdims=True) + eps)
    output = hidden * np.reciprocal(rms)
    if gamma is not None:
        output *= gamma
    if qkv_weights is not None:
        output = output @ qkv_weights
    return output
