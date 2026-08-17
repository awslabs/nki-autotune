"""One Qwen3.5 routed MoE block workload."""

from __future__ import annotations

import numpy as np

from nkigym.search.types import InputSpecs

T = 128
H = 2048
I_TP = 64
E = 1
SIGMOID_GAIN = 1.702


def numpy_golden(hidden, gate_up, down, affinity, block_expert, block_tokens):
    """fp32 block-by-block reference: gate/up -> act(gate)*up -> down -> affinity.

    act(g) = g * sigmoid(SIGMOID_GAIN * g): gain 1.702 for Swish/gelu_apprx_sigmoid
    (what we ship), 1.0 for true SiLU. Getting this constant wrong shows up as a
    uniform ~19% relative error on EVERY token, which is how it was found.

    POST_SCALE: affinity multiplies AFTER the down projection. Shared by every
    arm, so an arm that skips blocks (e.g. LNC=1 on a ping-pong kernel) shows up
    as a cosine collapse rather than a speedup.
    """
    x_all = hidden.astype(np.float32)
    gup = gate_up.astype(np.float32)
    dwn = down.astype(np.float32)
    out = np.zeros((T, H), dtype=np.float32)
    for blk in range(block_expert.shape[0]):
        ids = block_tokens[blk]
        ids = ids[ids >= 0]
        if ids.size == 0:
            continue
        e = int(block_expert[blk])
        x = x_all[ids]
        g = x @ gup[e, :, 0, :]
        u = x @ gup[e, :, 1, :]
        h = (g / (1.0 + np.exp(-SIGMOID_GAIN * g))) * u
        o = (h @ dwn[e]) * affinity[ids, e][:, None]
        # np.add.at, not out[ids] += : with top_k=8 a token appears in 8 blocks
        # and fancy-index += would keep only the last write.
        np.add.at(out, ids, o)
    return out


def _input_generator(input_specs: InputSpecs, seed: int) -> dict[str, np.ndarray]:
    """Generate one populated source block with replayable expert tensors."""
    rng = np.random.default_rng(seed)
    inputs = {
        "hidden": rng.random((T, H), dtype=np.float32),
        "gate_up": rng.uniform(-0.1, 0.1, (E, H, 2, I_TP)).astype(np.float32),
        "down": rng.uniform(-0.1, 0.1, (E, I_TP, H)).astype(np.float32),
        "affinity": np.ones((T, E), dtype=np.float32),
        "block_expert": np.zeros((1,), dtype=np.int32),
        "block_tokens": np.arange(T, dtype=np.int32).reshape(1, T),
    }
    return {name: inputs[name] for name in input_specs}


WORKLOAD = {
    "numpy_ref": numpy_golden,
    "input_specs": {
        "hidden": ((T, H), "bfloat16"),
        "gate_up": ((E, H, 2, I_TP), "bfloat16"),
        "down": ((E, I_TP, H), "bfloat16"),
        "affinity": ((T, E), "bfloat16"),
        "block_expert": ((1,), "int32"),
        "block_tokens": ((1, T), "int32"),
    },
    "input_generator": _input_generator,
    "atol": 2e-2,
    "rtol": 2e-2,
    "best_historical_mfu": 2.82,
}
