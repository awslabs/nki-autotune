"""Random-policy transform rollouts for both canonical matmul workloads.

Each workload starts from its canonical SSA graph and samples legal actions
from every shipped transform. All generated states are dumped before one
shared-input fp32 simulation pass, so a failing transform sequence can be
replayed from its printed seed.

By default the driver runs both ``lhs_T.T @ rhs`` and ``lhs @ rhs``. Select one
with ``--workload`` when debugging a specific graph.

Usage::

    source ~/venvs/kernel-env/bin/activate
    PYTHONPATH=.:nkigym/src:autotune/src \
      python examples/random_rollout.py --cache /tmp/autotune_cache
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nkigym.environment import KernelMDP
from nkigym.ir import KernelIR
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.transpose import NKITranspose
from nkigym.synthesis.simulate_nki import simulate_fp32
from nkigym.transforms import (
    BufferCompaction,
    BufferLayout,
    CodeMotion,
    Fuse,
    LoadTranspose,
    MatmulTranspose,
    Reorder,
    RFactor,
    SoftwarePipeline,
    Split,
)

InputSpecs = dict[str, tuple[tuple[int, ...], str]]
KernelArtifact = tuple[str, Path]
K, M, N = 2048, 2048, 2048
VALIDATION_SEED = 0
ATOL = 5e-3
RTOL = 5e-3


@dataclass(frozen=True)
class Workload:
    """Canonical graph and replay settings for one rollout workload."""

    name: str
    input_specs: InputSpecs
    f_numpy: Callable[..., np.ndarray]
    f_nkigym: Callable[..., np.ndarray]
    rollout_seeds: tuple[int, ...]
    max_steps: int


def f_lhs_t_rhs_numpy(lhs_T: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Return the NumPy reference for ``lhs_T.T @ rhs``."""
    return lhs_T.T @ rhs


@nkigym_kernel
def f_lhs_t_rhs_nkigym(lhs_T, rhs):
    """Return the canonical SSA graph for ``lhs_T.T @ rhs``."""
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


def f_lhs_rhs_numpy(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Return the NumPy reference for ``lhs @ rhs``."""
    return lhs @ rhs


@nkigym_kernel
def f_lhs_rhs_nkigym(lhs, rhs):
    """Return the canonical SSA graph for ``lhs @ rhs``."""
    sbuf_lhs = NKILoad()(src=lhs)
    psum_lhs_T = NKITranspose()(data=sbuf_lhs)
    sbuf_lhs_T = NKITensorCopy()(src=psum_lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


LHS_T_RHS = Workload(
    name="matmul_lhsT_rhs",
    input_specs={"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")},
    f_numpy=f_lhs_t_rhs_numpy,
    f_nkigym=f_lhs_t_rhs_nkigym,
    rollout_seeds=(0,),
    max_steps=100,
)
LHS_RHS = Workload(
    name="matmul_lhs_rhs",
    input_specs={"lhs": ((M, K), "bfloat16"), "rhs": ((K, N), "bfloat16")},
    f_numpy=f_lhs_rhs_numpy,
    f_nkigym=f_lhs_rhs_nkigym,
    rollout_seeds=(3726201086334714209, 8914180450183696697, 1799233449652390075),
    max_steps=40,
)
WORKLOADS = {workload.name: workload for workload in (LHS_T_RHS, LHS_RHS)}
TRANSFORMS = [
    LoadTranspose(),
    MatmulTranspose(),
    Split(),
    Fuse(),
    Reorder(),
    CodeMotion(),
    RFactor(),
    SoftwarePipeline(),
    BufferLayout(),
    BufferCompaction(),
]


def _dump_kernel(state: KernelIR, cache_dir: Path, rollout: int, step: int) -> KernelArtifact:
    """Dump one rollout state and return its label and generated kernel path."""
    label = f"rollout_{rollout}/step_{step}"
    state_dir = cache_dir / label
    state.dump(state_dir)
    return label, state_dir / "kernel.py"


def _gather_kernels(workload: Workload, cache_dir: Path) -> list[KernelArtifact]:
    """Run every seeded rollout for one workload and gather dumped kernels."""
    kernels: list[KernelArtifact] = []
    environment = KernelMDP(workload.f_nkigym, workload.input_specs, transforms=TRANSFORMS)
    for rollout, seed in enumerate(workload.rollout_seeds):
        rng = random.Random(seed)
        state = environment.reset()
        kernels.append(_dump_kernel(state, cache_dir, rollout, 0))
        for step in range(1, workload.max_steps + 1):
            actions = environment.legal_actions(state)
            if not actions:
                break
            action = rng.choice(actions)
            print(
                f"[{workload.name} rollout {rollout} seed {seed}] step {step}: {type(action[0]).__name__} {action[1]}"
            )
            state = environment.step(state, action)
            kernels.append(_dump_kernel(state, cache_dir, rollout, step))
    print(f"[{workload.name}] gathered {len(kernels)} kernel(s)")
    return kernels


def _check_numerics(workload: Workload, kernels: list[KernelArtifact]) -> None:
    """Simulate one workload's kernels against a shared NumPy reference."""
    rng = np.random.default_rng(VALIDATION_SEED)
    inputs = {
        name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dtype) in workload.input_specs.items()
    }
    expected = workload.f_numpy(**inputs)
    generated_name = f"nki_{workload.f_nkigym.__name__}"
    for index, (label, kernel_path) in enumerate(kernels):
        module_name = f"dumped_{workload.name}_{index}"
        spec = importlib.util.spec_from_file_location(module_name, kernel_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load {label} from {kernel_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        actual = np.asarray(simulate_fp32(getattr(module, generated_name))(**inputs))
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL, err_msg=label)
        print(f"[numerics] {workload.name}/{label}: PASS (atol={ATOL}, rtol={RTOL})")
    print(f"[numerics] {workload.name}: all {len(kernels)} kernel(s) PASS")


def _run_workload(workload: Workload, cache_root: Path) -> None:
    """Generate and validate all configured rollouts for one workload."""
    cache_dir = cache_root / workload.name / "random_rollout"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    kernels = _gather_kernels(workload, cache_dir)
    _check_numerics(workload, kernels)


def _parse_args() -> argparse.Namespace:
    """Parse cache and workload selection arguments."""
    parser = argparse.ArgumentParser(description="Run random transform rollouts for canonical matmul workloads.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--workload", choices=("all", *WORKLOADS), default="all")
    return parser.parse_args()


def _main() -> None:
    """Run the selected workload or both workloads."""
    args = _parse_args()
    cache_root = Path(args.cache).expanduser().resolve()
    selected = list(WORKLOADS.values()) if args.workload == "all" else [WORKLOADS[args.workload]]
    for workload in selected:
        _run_workload(workload, cache_root)


if __name__ == "__main__":
    _main()
