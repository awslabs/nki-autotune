"""Random-policy MDP rollouts — the transform composability + correctness test.

Wraps the canonical matmul ``f_nkigym`` in :class:`nkigym.environment.KernelMDP`
and samples random ``(transform, option)`` actions over the full shipped set
(``Split + Fuse + Reorder + CodeMotion + RFactor + SoftwarePipeline +
BufferLayout + BufferCompaction``). It generates and dumps every rollout first,
then CPU-sim-checks all gathered kernels in one pass with shared inputs and a
single numpy reference result. This is the fuzz test that every legal transform
composition stays correctness-preserving: a divergence at any step is a
transform (or legality) bug, independent of any hand-authored ladder.

The ``f_nkigym`` body below is the output of
:func:`nkigym.synthesis.numpy_to_nkigym.compile_numpy_to_nkigym`
pasted verbatim — re-run the synthesiser manually whenever the op
surface or workload changes.

Usage::

    source ~/venvs/kernel-env/bin/activate
    PYTHONPATH=nkigym/src python examples/matmul_lhsT_rhs.py --cache /tmp/autotune_cache
"""

import argparse
import importlib.util
import os
import random
import shutil

import numpy as np

from nkigym.environment import KernelMDP
from nkigym.ir import KernelIR
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.synthesis.simulate_nki import simulate_fp32
from nkigym.transforms import (
    BufferCompaction,
    BufferLayout,
    CodeMotion,
    Fuse,
    Reorder,
    RFactor,
    SoftwarePipeline,
    Split,
)

K, M, N = 2048, 2048, 2048
INPUT_SPECS: dict[str, tuple[tuple[int, ...], str]] = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}
NUM_ROLLOUTS = 1
MAX_STEPS = 100
SEED = 0
"""Rollout ``k`` uses ``random.Random(SEED + k)`` so divergences replay exactly."""
TRANSFORMS = [
    Split(),
    Fuse(),
    Reorder(),
    CodeMotion(),
    RFactor(),
    SoftwarePipeline(),
    BufferLayout(),
    BufferCompaction(),
]


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


def _dump_kernel(state: KernelIR, cache_dir: str, rollout: int, step: int) -> tuple[str, str]:
    """Dump one rollout state and return its label and generated kernel path."""
    label = f"rollout_{rollout}/step_{step}"
    state_dir = os.path.join(cache_dir, label)
    state.dump(state_dir)
    return label, os.path.join(state_dir, "kernel.py")


def _gather_kernels(cache_dir: str) -> list[tuple[str, str]]:
    """Run every seeded random rollout and gather all dumped kernels."""
    kernels: list[tuple[str, str]] = []
    env = KernelMDP(f_nkigym, INPUT_SPECS, transforms=TRANSFORMS)
    for rollout in range(NUM_ROLLOUTS):
        rng = random.Random(SEED + rollout)
        state = env.reset()
        kernels.append(_dump_kernel(state, cache_dir, rollout, 0))
        for step in range(1, MAX_STEPS + 1):
            actions = env.legal_actions(state)
            if not actions:
                break
            action = rng.choice(actions)
            print(f"[rollout {rollout}] step {step}: {type(action[0]).__name__} {action[1]}")
            state = env.step(state, action)
            kernels.append(_dump_kernel(state, cache_dir, rollout, step))
    print(f"[rollouts] gathered {len(kernels)} kernel(s)")
    return kernels


def _check_numerics(kernels: list[tuple[str, str]], seed: int = 0, atol: float = 5e-3, rtol: float = 5e-3) -> None:
    """Simulate all gathered kernels against one shared numpy reference."""
    rng = np.random.default_rng(seed)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dtype) in INPUT_SPECS.items()}
    expected = f_numpy(**inputs)
    for index, (label, kernel_path) in enumerate(kernels):
        spec = importlib.util.spec_from_file_location(f"dumped_kernel_{index}", kernel_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load {label} from {kernel_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        actual = np.asarray(simulate_fp32(module.nki_f_nkigym)(**inputs))
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=label)
        print(f"[numerics] {label}: PASS (atol={atol}, rtol={rtol})")
    print(f"[numerics] all {len(kernels)} kernel(s) PASS")


def _main() -> None:
    """Generate all rollout kernels, then validate the gathered batch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    cache_dir = os.path.join(args.cache, "matmul_lhsT_rhs")
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)

    """Random-policy rollouts via the KernelMDP environment. Each rollout is SEEDED
    (``SEED + k``) so a numerics divergence is REPRODUCIBLE, and every action is
    printed so the exact ``(transform, option)`` that broke correctness is
    recoverable from the log — a fuzz test is only useful if its failures replay."""
    kernels = _gather_kernels(cache_dir)
    _check_numerics(kernels)


if __name__ == "__main__":
    _main()
