"""Numpy matmul → ``f_nkigym`` → canonical IR → random-policy rollouts.

The ``f_nkigym`` body below is the output of
:func:`nkigym.synthesis.numpy_to_nkigym.compile_numpy_to_nkigym`
pasted verbatim — re-run the synthesiser manually whenever the op
surface or workload changes.

After dumping the canonical IR (``step_0``) and checking numerics, this
script wraps ``f_nkigym`` in :class:`nkigym.environment.KernelMDP`,
samples random ``(transform, option)`` actions over ``Split + Fuse + Reorder``
for ``NUM_ROLLOUTS`` independent rollouts, and dumps every step's IR.

Usage::

    source ~/venvs/kernel-env/bin/activate
    PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python examples/matmul_lhsT_rhs.py
"""

import importlib.util
import os
import random
import shutil

import numpy as np

from nkigym.environment import KernelMDP
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.synthesis.simulate_nki import simulate_fp32
from nkigym.transforms import Fuse, Reorder, Split

K, M, N = 2048, 2048, 2048
INPUT_SPECS: dict[str, tuple[tuple[int, ...], str]] = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}
NUM_ROLLOUTS = 4
MAX_STEPS = 5
SEED = 42


def f_numpy(lhs_T: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """``lhs_T.T @ rhs`` — plain numpy reference (synthesis source)."""
    return lhs_T.T @ rhs


@nkigym_kernel
def f_nkigym(lhs_T, rhs):
    """Cached output of ``compile_numpy_to_nkigym(f_numpy, ...)``."""
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


def _check_numerics(kernel_path: str, seed: int = 0, atol: float = 5e-3, rtol: float = 5e-3) -> None:
    """Simulate the dumped kernel under fp32 and compare against ``f_numpy``."""
    rng = np.random.default_rng(seed)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dtype) in INPUT_SPECS.items()}
    expected = f_numpy(**inputs)
    spec = importlib.util.spec_from_file_location("dumped_kernel", kernel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = np.asarray(simulate_fp32(module.nki_f_nkigym)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    print(f"[numerics] PASS (atol={atol}, rtol={rtol})")


if __name__ == "__main__":
    CACHE_DIR = "/home/ubuntu/cache/matmul_lhsT_rhs"
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    """Random-policy rollouts via the KernelMDP environment."""
    env = KernelMDP(f_nkigym, INPUT_SPECS, transforms=[Split(), Fuse(), Reorder()])
    rng = random.Random(SEED)
    for k in range(NUM_ROLLOUTS):
        state = env.reset()
        cache_0 = f"{CACHE_DIR}/rollout_{k}/step_0"
        state.dump(cache_0)
        _check_numerics(f"{cache_0}/kernel.py")
        for t in range(1, MAX_STEPS + 1):
            actions = env.legal_actions(state)
            if not actions:
                break
            action = rng.choice(actions)
            state = env.step(state, action)
            cache_t = f"{CACHE_DIR}/rollout_{k}/step_{t}"
            state.dump(cache_t)
            _check_numerics(f"{cache_t}/kernel.py")
