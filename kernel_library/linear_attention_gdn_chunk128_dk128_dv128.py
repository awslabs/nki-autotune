"""One Qwen3.5 Gated DeltaNet chunk workload."""

from __future__ import annotations

import numpy as np

from nkigym.search.types import InputSpecs

_BATCH = 1
_HEADS = 1
_CHUNK = 128
_KEY_DIM = 128
_VALUE_DIM = 128


def numpy_gdn_prefill_ref(query, key, value, g_log, beta, state):
    """Step-by-step GDN recurrence in fp32 (the CPU ground truth)."""
    B, H, S, Dk = query.shape
    Dv = value.shape[-1]
    out = np.zeros((B, H, S, Dv), dtype=np.float32)
    state = state.astype(np.float32).copy()
    for b in range(B):
        for h in range(H):
            st = state[b, h]
            for i in range(S):
                g_t = np.exp(g_log[b, h, i])
                bt = beta[b, h, i]
                st = st * g_t
                kv_mem = np.einsum("kv,k->v", st, key[b, h, i])
                delta = (value[b, h, i] - kv_mem) * bt
                st = st + np.outer(key[b, h, i], delta)
                out[b, h, i] = np.einsum("kv,k->v", st, query[b, h, i])
            state[b, h] = st
    return out, state


def _l2norm(values: np.ndarray) -> np.ndarray:
    """Normalize the final dimension using the source test's floor."""
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-6)


def _input_generator(input_specs: InputSpecs, seed: int) -> dict[str, np.ndarray]:
    """Generate the production-like distribution from the source numerical test."""
    rng = np.random.default_rng(seed)
    inputs = {
        "query": _l2norm(rng.standard_normal((_BATCH, _HEADS, _CHUNK, _KEY_DIM)).astype(np.float32))
        / np.sqrt(np.float32(_KEY_DIM)),
        "key": _l2norm(rng.standard_normal((_BATCH, _HEADS, _CHUNK, _KEY_DIM)).astype(np.float32)),
        "value": rng.standard_normal((_BATCH, _HEADS, _CHUNK, _VALUE_DIM)).astype(np.float32) * 0.05,
        "g_log": -np.abs(rng.standard_normal((_BATCH, _HEADS, _CHUNK)).astype(np.float32)) * 0.3,
        "beta": (1.0 / (1.0 + np.exp(-rng.standard_normal((_BATCH, _HEADS, _CHUNK)).astype(np.float32)))).astype(
            np.float32
        ),
        "state": rng.standard_normal((_BATCH, _HEADS, _KEY_DIM, _VALUE_DIM)).astype(np.float32) * 0.01,
    }
    return {name: inputs[name] for name in input_specs}


WORKLOAD = {
    "numpy_ref": numpy_gdn_prefill_ref,
    "input_specs": {
        "query": ((_BATCH, _HEADS, _CHUNK, _KEY_DIM), "float32"),
        "key": ((_BATCH, _HEADS, _CHUNK, _KEY_DIM), "float32"),
        "value": ((_BATCH, _HEADS, _CHUNK, _VALUE_DIM), "float32"),
        "g_log": ((_BATCH, _HEADS, _CHUNK), "float32"),
        "beta": ((_BATCH, _HEADS, _CHUNK), "float32"),
        "state": ((_BATCH, _HEADS, _KEY_DIM, _VALUE_DIM), "float32"),
    },
    "input_generator": _input_generator,
    "atol": 1e-2,
    "rtol": 0.0,
    "best_historical_mfu": 9.37,
}
